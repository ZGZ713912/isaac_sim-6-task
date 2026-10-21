#!/usr/bin/env python3
"""Export an exact behavior-reviewed 35D->6D deterministic candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mean(checkpoint):
    import torch
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.robust_getup import RobustGetupActor
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = state["infos"]
    if info["schema"] != "robust-getup-checkpoint-v2":
        raise ValueError("Expected a v2 training checkpoint")
    actor_source = "src/wheeled_algo/robust_getup.py"
    if digest(ROOT / actor_source) != info["source_sha256"][actor_source]:
        raise ValueError("Actor source differs from the trained source")
    parent_file = Path(info["base_checkpoint"])
    if digest(parent_file) != info["base_checkpoint_sha256"]:
        raise ValueError("Locomotion parent changed")
    parent = torch.load(parent_file, map_location="cpu", weights_only=True)
    plan = json.loads((ROOT / parent["infos"]["plan_file"]).read_text())
    mean = RobustGetupActor(mean_from_checkpoint(parent, ROOT / plan["actor_bundle"] / "source.onnx"),
                           quiet_alpha=info["settings"]["quiet_actor_alpha"])
    weights = {k.removeprefix("mlp.mean."): v for k, v in state["actor_state_dict"].items() if k.startswith("mlp.mean.")}
    mean.load_state_dict(weights, strict=True)
    expected = {k.removeprefix("mlp."): v for k, v in parent["actor_state_dict"].items() if k.startswith("mlp.")}
    if any(not torch.equal(value, mean.parent.state_dict()[key]) for key, value in expected.items()):
        raise ValueError("Frozen locomotion parent was modified")
    if not all(torch.isfinite(v).all() for v in mean.state_dict().values()):
        raise ValueError("Nonfinite policy state")
    return mean.eval(), info


def assess(reports, baselines, checkpoint, protocol):
    from analyze_robust_getup import analyze
    checkpoint_sha = digest(checkpoint)
    failures, results = [], []
    if len(reports) != 2 or len(baselines) != 2:
        raise ValueError("Two matched confirmation seeds are required")
    for evaluation, baseline in zip(reports, baselines, strict=True):
        data = json.loads((evaluation / "report.json").read_text())
        parent = json.loads((baseline / "report.json").read_text())
        if (data["checkpoint_sha256"] != checkpoint_sha or data["actor_variant"] != "robust"
                or parent["actor_variant"] != "parent" or data["seed"] != parent["seed"]):
            raise ValueError("Evaluation/checkpoint/baseline identity mismatch")
        analysis, original = analyze(evaluation), analyze(baseline)
        if analysis["passive_motor_peak_nm"] != 0. or analysis["max_gap_m"] > protocol["closure_gap_max_m"]:
            failures.append("passive torque or closed-chain protection")
        for family, rate in protocol["family_min_complete_rate"].items():
            row = analysis["actual_settled_attitudes"].get(family, {})
            successes = row.get("recovery_and_motion_success", 0)
            if successes < protocol["minimum_successful_episodes_per_family"] or successes < rate * row.get("episodes", 0):
                failures.append(f"seed {data['seed']} actual {family} recovery and resumed motion")
        if analysis["motion_survived"] < protocol["ordinary_motion_survival_rate_min"] * analysis["motion_episodes"]:
            failures.append(f"seed {data['seed']} ordinary motion survival")
        for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right"):
            row = analysis["cases"].get(name, {})
            if (row.get("vx_mae_m_s") is None or row["vx_mae_m_s"] > protocol["ordinary_arc_vx_mae_max_m_s"]
                    or row["yaw_mae_rad_s"] > protocol["ordinary_arc_yaw_mae_max_rad_s"]):
                failures.append(f"seed {data['seed']} {name} simultaneous tracking")
        quiet_gains = []
        for name in ("stand", "stand_low", "stand_high"):
            row, old = analysis["cases"][name], original["cases"][name]
            quiet, old_quiet = row.get("quiet", {}), old.get("quiet", {})
            if (row["survived"] < protocol["stand_survival_rate_min"] * row["episodes"]
                    or quiet.get("vx_rms_m_s", float("inf")) > protocol["stand_vx_rms_max_m_s"]
                    or quiet.get("yaw_rms_rad_s", float("inf")) > protocol["stand_yaw_rms_max_rad_s"]):
                failures.append(f"seed {data['seed']} {name} standing")
            for field in ("leg_target_hf_rms_rad", "wheel_target_hf_rms_rad_s"):
                if old_quiet.get(field, 0.) > 1e-6 and field in quiet:
                    quiet_gains.append(1. - quiet[field] / old_quiet[field])
        if not quiet_gains or sum(quiet_gains) / len(quiet_gains) < protocol["nominal_quiet_hf_target_rms_improvement_fraction"]:
            failures.append(f"seed {data['seed']} quiet target vibration reduction")
        if not any(row.get("push") and row.get("in_motion_fall") and row.get("recovered_after_push")
                   and row["followup_success"] and row["survived"] for row in data["lanes"]):
            failures.append(f"seed {data['seed']} actual in-motion fall, recovery and resumed command")
        results.append(dict(evaluation=str(evaluation), evaluation_sha256=digest(evaluation / "report.json"),
            baseline=str(baseline), baseline_sha256=digest(baseline / "report.json"),
            mean_quiet_hf_improvement_fraction=sum(quiet_gains) / len(quiet_gains) if quiet_gains else None,
            actual_settled_attitudes=analysis["actual_settled_attitudes"]))
    if sorted(r["seed"] for r in (json.loads((p / "report.json").read_text()) for p in reports)) != sorted(protocol["required_confirmation_seeds"]):
        failures.append("confirmation seeds")
    return dict(adequate=not failures, failures=failures, comparisons=results)


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs=2, required=True)
    parser.add_argument("--baseline", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/unified_robust_20261021/acceptance_protocol001.json")
    args = parser.parse_args()
    behavior = assess(args.evaluation, args.baseline, args.checkpoint, json.loads(args.protocol.read_text()))
    if not behavior["adequate"]:
        raise ValueError("Candidate still needs iteration: " + "; ".join(behavior["failures"]))
    mean, info = load_mean(args.checkpoint)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    policy = args.output / "policy.onnx"
    torch.onnx.export(mean, torch.zeros(1, 35), str(policy), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(policy)
    onnx.helper.set_model_props(model, dict(actor_adaptation="wheel-leg-robust-getup-v2", policy_hz="200",
        observation_dim="35", action_dim="6", motor_control="full_RL_after_passive_enable"))
    onnx.save(model, policy)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    worst = 0.
    with torch.inference_mode():
        for directory in args.evaluation:
            obs = np.load(directory / "trajectory.npz")["observations"].reshape(-1, 35)
            for value in obs[np.linspace(0, len(obs) - 1, min(2048, len(obs)), dtype=int)]:
                expected = mean(torch.from_numpy(value[None])).numpy()
                actual = session.run(None, {"obs": value[None]})[0]
                if not np.isfinite(actual).all():
                    raise ValueError("Nonfinite ONNX output")
                worst = max(worst, float(np.abs(actual - expected).max()))
    if worst > 5e-4:
        raise ValueError(f"ONNX parity error: {worst}")
    contract = dict(schema="robust-getup-deployment-v2", model_sha256=digest(policy),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        actor_dim=35, action_dim=6, history_length=1, policy_hz=200, physical_pd_hz=1000,
        deterministic_mean=True, new_robust_updates=info["completed_updates"],
        prior_recovery_updates=info["prior_recovery_updates"], prior_locomotion_updates=1346,
        parent_export_pretraining_updates="unknown", leg_action_clip=13., wheel_action_clip=9.,
        leg_position_scale=.25, wheel_velocity_scale=10., leg_kp=160., leg_kd=2.5, wheel_kp=.6,
        policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        nominal_model_pos=[-.42,.13742282595395358,.42,-.1374155762580851,0.,0.],
        required_runtime_changes=["bypass scripted recovery; policy owns all six motors from enable",
            "zero all six motor efforts before enable; retain gas springs",
            "at first enable populate four previous leg actions from wrapped measured nominal offsets / .25",
            "use exported actions as the next previous-action observation in policy order",
            "retain paired_branch107 projection, torque limits and sensor freshness protections",
            "retain yaw and reduce vx at abs(vx*yaw)>2.06; hold planar request during recovery"],
        quiet_filter="inside ONNX, using the previous action; no additional deployment filter",
        ort_max_abs_error=worst, behavior=behavior, evidence_scope="Isaac Sim; user performs vehicle testing",
        hardware_tested=False)
    (args.output / "deployment_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
