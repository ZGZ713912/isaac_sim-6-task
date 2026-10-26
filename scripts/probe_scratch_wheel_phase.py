#!/usr/bin/env python3
"""Short zero-PPO phase test of the axisymmetric native wheel cylinders.

Both arms use the same constant static-load leg targets and zero wheel-speed
targets. Only reset wheel angles vary. No actor controls these trajectories.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASES = (0., math.pi/12, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("identical", "sweep"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    reference_path = ROOT / "reports/robust_policy_repair_20261024/scratch_nominal_static_load001.json"
    reference = json.loads(reference_path.read_text())
    for name, expected in reference["sources_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Static target source changed: " + name)
    spec = importlib.util.spec_from_file_location("scratch_phase_probe_runner", ROOT / "scripts/train_scratch_standing.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    native_prepare, nominal = runner.prepare, runner.nominal_action
    short_source = inspect.getsource(runner.evaluate)
    anchor = "horizon = round(8. / env.policy_dt)"
    if short_source.count(anchor) != 1:
        raise ValueError("Frozen evaluation horizon changed")
    short_source = short_source.replace(anchor, "horizon = round(.2 / env.policy_dt)")
    exec(compile(ast.parse(short_source), str(Path(__file__).resolve()) + ":short_harness", "exec"), runner.__dict__)
    short_evaluate = runner.evaluate
    phases = PHASES if args.mode == "sweep" else (0.,)*7
    evidence = dict(schema="scratch-wheel-phase-probe-v1", mode=args.mode,
        phases_rad=list(phases), native_duration_s=.2, new_PPO=0, learned_policy=False,
        state_writes="episode reset only; wheel angles after native grounded placement",
        actor_controls_trajectory=False, standing_qualification=False, hardware_ready=False,
        source_sha256=sha(__file__), reference_sha256=sha(reference_path),
        scope="Axisymmetric collision/COM/inertia phase sensitivity; not policy or standing acceptance")

    def archive(phase):
        folder = args.output / "diagnostic_sources"
        folder.mkdir(exist_ok=True)
        for file in (Path(__file__), ROOT / "scripts/record_scratch_physics.py", reference_path):
            (folder / file.name).write_bytes(file.read_bytes())
        evidence["phase"] = phase
        runner.write_json(args.output / "diagnostic_receipt.json", evidence)

    def prepare(options):
        if options.updates or not options.fixed_target_probe:
            raise ValueError("Wheel phase diagnostic must not run PPO")
        config, settings = native_prepare(options)
        task_file = args.output / "task_contract.json"
        task_contract = json.loads(task_file.read_text())
        task_contract["wheel_phase_physics_probe"] = dict(evidence, all_actual_initial_pitch_deg=0.)
        runner.write_json(task_file, task_contract)
        archive("prepared_before_native_app")
        return config, settings

    def constant(env, height=.305):
        from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
        value = nominal(env, height)
        if getattr(env, "wheel_phase_probe_active", False):
            offset = value.new_tensor(reference["hypothetical_static_load_offset_rad"])
            value = value + offset[list(POLICY_FROM_CONTROL)]/.25
        return value

    def evaluate(env, actor, output, writer, completed, *, fixed_target=False):
        import torch
        import numpy as np
        from wheeled_tasks.chassis.root_kinematics import update_robot_data
        if completed or not fixed_target:
            raise ValueError("A fixed diagnostic cannot execute learning")
        reset = env.reset
        initial = {}

        def phase_reset(ids):
            if env.evaluation_cases is not None:
                for case in env.evaluation_cases:
                    case["pitch_deg"] = 0.
            result = reset(ids)
            root = env.robot.data.root_link_pose_w.torch[ids].clone()
            centers = env.wheel_centers()[ids].clone()
            positions = env.robot.data.joint_pos.torch[ids].clone()
            angle = positions.new_tensor([phases[int(i) % 7] for i in ids.cpu().tolist()])
            positions[:, env.ids[2]] += angle
            positions[:, env.ids[5]] -= angle
            env.robot.write_joint_position_to_sim_index(position=positions, env_ids=ids)
            update_robot_data(env, env.dt)
            if not torch.equal(root, env.robot.data.root_link_pose_w.torch[ids]):
                raise ValueError("Wheel phase must not change the root reset pose")
            center_change = float((centers-env.wheel_centers()[ids]).abs().max())
            if center_change > 2e-7:
                raise ValueError("Axial wheel phase changed the cylinder center")
            initial.update(root_pose=root[:7].cpu().tolist(), joints=positions[:7].cpu().tolist(),
                cylinder_center_max_change_m=center_change,
                motor_velocities=env.robot.data.joint_vel.torch[ids][:7, env.ids].cpu().tolist())
            return result

        env.reset, env.wheel_phase_probe_active = phase_reset, True
        try:
            result = short_evaluate(env, actor, output, writer, completed, fixed_target=True, heights=(.305,))
        finally:
            env.reset, env.wheel_phase_probe_active = reset, False
        for lane, case in enumerate(result["cases"]):
            case.update(original_harness_pitch_label_deg=case["pitch_deg"], pitch_deg=0.,
                left_wheel_phase_rad=phases[lane], right_wheel_phase_rad=-phases[lane],
                standing_pass=False, standing_qualification=False)
        result.update(protocol="200ms constant static-load targets; identical0deg pitch;7 wheel phase cases;every1ms",
            standing_pass=0, standing_qualification=False, initial_native_state=initial,
            diagnostic_only=True, learned_policy=False)
        from record_scratch_physics import record
        result["physics_runtime"] = record(env)
        with np.load(output / "native_trace.npz") as trace:
            for lane, case in enumerate(result["cases"]):
                force = trace["force"][:, :, lane].reshape(-1, 2, 3)
                writer.add_scalar(f"ProbeWheelPhase/{args.mode}/case{lane}/minimum_normal_force_n", float(force[..., 2].min()), 0)
                writer.add_scalar(f"ProbeWheelPhase/{args.mode}/case{lane}/missing_physical_ms", case["missing_physical_ms"], 0)
                writer.add_scalar(f"ProbeWheelPhase/{args.mode}/case{lane}/maximum_clearance_m", case["maximum_wheel_clearance_m"], 0)
        writer.flush()
        runner.write_json(output / "result.json", result)
        evidence.update(native_evaluation_completed=True, initial_native_state=initial)
        archive("evaluated_before_app_close")
        return result

    runner.prepare, runner.nominal_action, runner.evaluate = prepare, constant, evaluate
    sys.argv = [str(ROOT / "scripts/train_scratch_standing.py"), "--output", str(args.output),
                "--updates", "0", "--num-envs", "32", "--fixed-target-probe"]
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    runner.main()


if __name__ == "__main__":
    main()
