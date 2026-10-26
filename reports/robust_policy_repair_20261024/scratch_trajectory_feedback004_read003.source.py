#!/usr/bin/env python3
"""CPU-only local action derivatives at recorded native decision states.

The PD matrices are algebraic command derivatives, not physical closed-loop
poles. No optimizer, rollout, simulator or policy modification is constructed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(directory, times):
    receipt = json.loads((directory / "receipt.json").read_text())
    if (receipt["status"] != "completed_unqualified" or receipt["new_PPO"] != 0
            or receipt["optimizer_constructed"]):
        raise ValueError("Need a completed read-only native feedback evaluation")
    checkpoint_path = Path(receipt["checkpoint"])
    if sha(checkpoint_path) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA changed")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run = checkpoint_path.parent
    infos = state["infos"]
    agent_path = run / "agent_config.json"
    if infos.get("agent_config_sha256") != sha(agent_path):
        raise ValueError("Actor configuration SHA changed")
    config_path = run / "config/contract.json"
    if sha(config_path) != infos["mechanism_config_sha256"]:
        raise ValueError("Mechanism SHA changed")
    config = json.loads(config_path.read_text())
    control_path = Path(__file__).resolve().parents[1] / config["control_math_source"]
    if sha(control_path) != config["control_math_sha256"]:
        raise ValueError("Motor-control prior SHA changed")
    control = json.loads(control_path.read_text())
    response = config.get("actuator_response", {})
    if response.get("enabled") and (response["gain"] != 1. or any(response["time_constant_s"])):
        raise ValueError("This algebraic PD diagnostic requires direct nominal torque response")
    agent = json.loads(agent_path.read_text())
    actor_cfg = dict(agent["actor"])
    actor_name = actor_cfg.pop("class_name")
    actor_class = MLPModel
    if actor_name == "wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel":
        name = "src/wheeled_algo/wheel_velocity_increment.py"
        model_file = run / "source_snapshot" / name
        if sha(model_file) != infos["source_hashes"][name]:
            raise ValueError("Frozen actor action model SHA changed")
        spec = importlib.util.spec_from_file_location("frozen_trajectory_increment_actor", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actor_class = module.WheelVelocityIncrementModel
    elif actor_name != "MLPModel":
        raise ValueError("Unsupported actor action ABI")
    dummy = TensorDict(dict(policy=torch.zeros(1, 35)), batch_size=[1])
    actor = actor_class(dummy, agent["obs_groups"], "actor", 6, **actor_cfg)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    actor.eval()
    result = receipt["evaluation"]
    feedback_path = directory / "evaluation/feedback_trace.npz"
    physical_path = directory / "evaluation/native_trace.npz"
    if (sha(feedback_path) != result["feedback_trace"]["sha256"]
            or sha(physical_path) != result["trace_sha256"]):
        raise ValueError("Native trace SHA changed")

    def mean(frame):
        return actor(TensorDict(dict(policy=frame[None]), batch_size=[1]))[0]

    rows = []
    with np.load(feedback_path) as feedback, np.load(physical_path) as physical:
        for lane, case in enumerate(result["cases"]):
            if case["height_m"] != .305 or case["pitch_deg"] not in (0., -.5):
                continue
            for time in times:
                index = round(time / .005)
                if not physical["valid"][index, lane]:
                    continue
                frame = torch.tensor(feedback["observation_policy"][index, lane])
                output = mean(frame)
                recorded = torch.tensor(feedback["action_raw_policy"][index, lane])
                error = float((output.detach() - recorded).abs().max())
                if error > 2e-6:
                    raise ValueError("CPU actor mean differs from the recorded native decision")
                jac = torch.autograd.functional.jacobian(mean, frame)
                finite_difference = torch.empty(6, 6)
                for axis in range(6):
                    plus, minus = frame.clone(), frame.clone()
                    plus[16 + axis] += .0001  # +/-0.001 rad/s in raw0.1*dq units.
                    minus[16 + axis] -= .0001
                    finite_difference[:, axis] = (mean(plus).detach() - mean(minus).detach()) / .002
                finite_difference_error = float((finite_difference - jac[:, 16:22]*.1).abs().max())
                if finite_difference_error > .002:
                    raise ValueError("Finite differences do not confirm the encoder derivative units")
                # Policy axes: four legs then two wheels. Raw speeds are0.1*dq.
                kp, kd = config["v5_control"]["leg_kp"], config["v5_control"]["leg_kd"]
                leg_velocity_gain = kp * .25 * .1 * jac[:4, 16:20] - kd * torch.eye(4)
                wheel_kd = control["actuators"]["wheel"]["kd"]
                wheel_velocity_gain = wheel_kd * (jac[4:, 20:22] - torch.eye(2))
                previous = jac[:, 22:28]
                torque = physical["torque"][index, :, lane]
                force = physical["force"][index, :, lane]
                row = dict(pitch_deg=case["pitch_deg"], decision_time_s=index*.005,
                    CPU_mean_max_abs_difference=error,
                    encoder_derivative_finite_difference_max_abs_error=finite_difference_error,
                    leg_command_torque_velocity_gain_nm_per_rad_s=leg_velocity_gain.tolist(),
                    leg_command_gain_symmetric_eigenvalues=torch.linalg.eigvalsh(
                        (leg_velocity_gain + leg_velocity_gain.T)*.5).tolist(),
                    wheel_command_torque_velocity_gain_nm_per_rad_s=wheel_velocity_gain.tolist(),
                    wheel_command_gain_symmetric_eigenvalues=torch.linalg.eigvalsh(
                        (wheel_velocity_gain + wheel_velocity_gain.T)*.5).tolist(),
                    previous_action_spectral_radius=float(torch.linalg.eigvals(previous).abs().max()),
                    native_leg_effort_peak_next5ms_nm=float(np.abs(torque).max()),
                    native_dual_support_next5ms=(np.linalg.norm(force, axis=-1) > 2.).all(-1).tolist(),
                    actor_input_jacobian=jac.tolist())
                rows.append(row)
    return dict(schema="scratch-native-state-local-feedback-v1", checkpoint=str(checkpoint_path),
        checkpoint_sha256=receipt["checkpoint_sha256"], feedback_trace_sha256=sha(feedback_path),
        evaluation_receipt_sha256=sha(directory / "receipt.json"), rows=rows,
        motor_control_prior_sha256=sha(control_path),
        new_PPO=0, optimizer_constructed=False, hardware_ready=False,
        source_sha256=sha(__file__),
        scope="Unsaturated algebraic V5 PD action derivatives at recorded states, with other actor inputs fixed. "
              "Positive symmetric gain can inject velocity-aligned command torque at a decision boundary. "
              "Targets remain held for5 physical steps; this is not a physical damping matrix, closed-loop pole "
              "test, isolated cause or behavior qualification.")


def record_events(report, directory):
    from rsl_rl.utils.logger import Logger
    run = Path(report["checkpoint"]).parent
    agent = json.loads((run / "agent_config.json").read_text())
    config = json.loads((run / "config/contract.json").read_text())
    directory.mkdir(parents=True, exist_ok=False)
    logger = Logger(log_dir=str(directory), cfg=agent, env_cfg=config, num_envs=1,
        is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
    logger.init_logging_writer()
    try:
        for row in report["rows"]:
            metrics = dict(
                leg_command_gain_max_nm_per_rad_s=max(row["leg_command_gain_symmetric_eigenvalues"]),
                wheel_command_gain_max_nm_per_rad_s=max(row["wheel_command_gain_symmetric_eigenvalues"]),
                previous_action_spectral_radius=row["previous_action_spectral_radius"],
                CPU_actor_mean_max_abs_difference=row["CPU_mean_max_abs_difference"],
                encoder_derivative_finite_difference_max_abs_error=row["encoder_derivative_finite_difference_max_abs_error"])
            step = round(row["decision_time_s"] / .005)
            for name, value in metrics.items():
                logger.writer.add_scalar(
                    f"DiagnosticScratchLocalFeedback200Hz/pitch{row['pitch_deg']:+.2f}/{name}", value, step)
        logger.writer.flush()
    finally:
        logger.writer.close()
    return dict(directory=str(directory.resolve()),
        events_sha256={str(path): sha(path) for path in directory.glob("events.out.tfevents.*")},
        step="Sparse original200Hz decision index; divide by200 for native seconds, not PPO updates",
        scope="CPU local derivative diagnostics from sealed native states; no new simulation or qualification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError("Preserve old diagnostic reports")
    times = (0., .1, .5, 1., 2., 3., 3.7, 3.79, 3.8, 3.805, 4., 4.2, 4.5, 5., 5.5, 6., 7.)
    report = inspect(args.evaluation_run, times)
    report["tensorboard"] = record_events(report, args.output.with_suffix(".events"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps([dict(pitch_deg=row["pitch_deg"], time_s=row["decision_time_s"],
        leg_command_gain_max_eigenvalue=max(row["leg_command_gain_symmetric_eigenvalues"]),
        wheel_command_gain_max_eigenvalue=max(row["wheel_command_gain_symmetric_eigenvalues"]),
        previous_action_radius=row["previous_action_spectral_radius"])
        for row in report["rows"]]))
