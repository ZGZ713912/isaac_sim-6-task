#!/usr/bin/env python3
"""CPU-only local actor response diagnostic; not a closed-loop stability test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_actor_class(actor_name, run, infos):
    """Import authenticated actor dependencies in an isolated archive package."""
    if actor_name == "MLPModel":
        return MLPModel
    names = {
        "wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel": (
            "wheel_velocity_increment", "WheelVelocityIncrementModel"),
        "wheeled_algo.leg_velocity_separated:LegVelocitySeparatedModel": (
            "leg_velocity_separated", "LegVelocitySeparatedModel"),
    }
    if actor_name not in names:
        raise ValueError("Unsupported scratch actor action ABI")
    module_name, class_name = names[actor_name]
    dependencies = ["wheel_velocity_increment"]
    if module_name != dependencies[0]:
        dependencies.append(module_name)
    directory = run / "source_snapshot/src/wheeled_algo"
    for name in dependencies:
        relative = f"src/wheeled_algo/{name}.py"
        if sha(directory / f"{name}.py") != infos["source_hashes"].get(relative):
            raise ValueError("Archived actor dependency changed: " + relative)
    package_name = "_frozen_scratch_actor_" + hashlib.sha256(str(run.resolve()).encode()).hexdigest()
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    for name in dependencies:
        qualified = package_name + "." + name
        spec = importlib.util.spec_from_file_location(qualified, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[package_name + "." + module_name], class_name)


def inspect(path):
    run = path.parent
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    infos = checkpoint["infos"]
    if infos.get("schema") != "scratch-standing-checkpoint-v1" or infos.get("training_origin") != "scratch":
        raise ValueError("Only the new from-zero policies are supported")
    sensor_file = run / "source_snapshot/src/wheeled_tasks/chassis/scut_observation.py"
    if sha(sensor_file) != infos["source_hashes"]["src/wheeled_tasks/chassis/scut_observation.py"]:
        raise ValueError("Archived observation ABI changed")
    spec = importlib.util.spec_from_file_location("frozen_scut_observation", sensor_file)
    sensors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sensors)
    cfg = json.loads((run / "config/contract.json").read_text())
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / cfg["asset_directory"] / "manifest.json").read_text())
    bank_file = run / "pose_bank.json"
    if not bank_file.exists():
        bank_file = Path(json.loads((run / "task_contract.json").read_text())["bank"])
    bank = json.loads(bank_file.read_text())
    pose = bank["poses"][bank["diagnostic_height_starts"]["mapping"]["stand"]]
    names = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
    nominal = torch.tensor([manifest["nominal_joint_pos"][name] for name in names])
    q = torch.tensor([[pose["full_joint_positions"][name] for name in names]])
    action = torch.atan2((q - nominal).sin(), (q - nominal).cos()) / .25
    action[:, [2, 5]] = 0.
    observation = sensors.build_scut35(torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
        torch.tensor([[0., 0., .305]]), q, torch.zeros(1, 6), action, nominal,
        torch.zeros(1, dtype=torch.bool), torch.zeros(1), torch.zeros(1))
    agent = json.loads((run / "agent_config.json").read_text())
    actor_cfg = dict(agent["actor"])
    actor_name = actor_cfg.pop("class_name")
    agent_hash = infos.get("agent_config_sha256")
    if (agent_hash is not None and sha(run / "agent_config.json") != agent_hash
            or actor_name != infos.get("actor_class", "MLPModel")
            or actor_name != "MLPModel" and agent_hash is None):
        raise ValueError("Actor class/configuration is not authenticated")
    actor_class = frozen_actor_class(actor_name, run, infos)
    actor = actor_class(TensorDict(dict(policy=observation), batch_size=[1]), agent["obs_groups"], "actor", 6, **actor_cfg)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    def mean(frame):
        return actor(TensorDict(dict(policy=frame[None]), batch_size=[1]))[0]

    frame = observation[0].clone().requires_grad_(True)
    output = mean(frame)
    jacobian = torch.autograd.functional.jacobian(mean, frame)
    recurrence = jacobian[:, 22:28]
    eigenvalues = torch.linalg.eigvals(recurrence)
    previous_policy = action[:, list(sensors.POLICY_FROM_CONTROL)][0]
    return dict(checkpoint=str(path), checkpoint_sha256=sha(path), completed_updates=infos["completed_updates"],
        physical_nominal_input=observation[0].tolist(), output_mean_raw=output.detach().tolist(),
        initial_leg_target_error_rad=((output[:4]-previous_policy[:4])*.25).detach().tolist(),
        wheel_velocity_targets_rad_s=(10.*output[4:]).detach().tolist(),
        wheel_target_gain_per_pitch_rad=(10.*jacobian[4:, 7]).detach().tolist(),
        wheel_target_gain_per_pitch_rate_rad_s=(5.*jacobian[4:, 5]).detach().tolist(),
        leg_target_gain_per_leg_velocity_rad_s=(.025*jacobian[:4, 16:20]).detach().tolist(),
        previous_action_spectral_radius=float(eigenvalues.abs().max()),
        previous_action_jacobian=recurrence.detach().tolist(),
        gaussian_std_raw=actor.distribution.log_std_param.detach().exp().tolist(),
        actor_input_jacobian=jacobian.detach().tolist(), new_PPO=0,
        scope="Local actor derivative at ideal mid-height geometry with all other input features held fixed; not physical closed-loop poles or causal proof")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = [inspect(path) for path in args.checkpoints]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema="scratch-actor-local-response-v1", rows=rows), indent=2, allow_nan=False)+"\n")
    print(json.dumps([{key: row[key] for key in ("completed_updates", "initial_leg_target_error_rad", "wheel_velocity_targets_rad_s",
        "wheel_target_gain_per_pitch_rad", "wheel_target_gain_per_pitch_rate_rad_s", "previous_action_spectral_radius")} for row in rows]))
