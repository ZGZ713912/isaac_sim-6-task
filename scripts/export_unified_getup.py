#!/usr/bin/env python3
"""Export a behavior-reviewed unified policy, with a separate deployment ABI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mean(checkpoint):
    import torch
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.unified_getup import UnifiedGetupActor

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = state["infos"]
    if info["schema"] != "unified-getup-checkpoint-v1":
        raise ValueError("Expected a unified get-up learning checkpoint")
    parent_path = Path(info["base_checkpoint"])
    if digest(parent_path) != info["base_checkpoint_sha256"]:
        raise ValueError("Frozen locomotion parent changed")
    actor_source = "src/wheeled_algo/unified_getup.py"
    if digest(ROOT / actor_source) != info["source_sha256"][actor_source]:
        raise ValueError("Actor implementation changed since this checkpoint")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    plan = json.loads((ROOT / parent["infos"]["plan_file"]).read_text())
    mean = UnifiedGetupActor(mean_from_checkpoint(parent, ROOT / plan["actor_bundle"] / "source.onnx"))
    mean.load_state_dict({key.removeprefix("mlp."): value for key, value in state["actor_state_dict"].items()
                          if key.startswith("mlp.")}, strict=True)
    expected = {key.removeprefix("mlp."): value for key, value in parent["actor_state_dict"].items()
                if key.startswith("mlp.")}
    if any(not torch.equal(value, mean.parent.state_dict()[key]) for key, value in expected.items()):
        raise ValueError("Frozen locomotion weights changed during self-righting learning")
    if not all(torch.isfinite(value).all() for value in mean.state_dict().values()):
        raise ValueError("Nonfinite policy state")
    return mean.cpu().eval(), info


def check_behavior(report, checkpoint_sha):
    if report.get("ablation_variants"):
        raise ValueError("A paired ablation report cannot qualify a deployed policy")
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Evaluation does not authenticate this exact checkpoint")
    recovery = [row for row in report["lanes"] if row["recovery"]]
    actual = {}
    for row in recovery:
        family = row["post_settle_family"]
        bucket = actual.setdefault(family, dict(episodes=0, successful=0))
        bucket["episodes"] += 1
        bucket["successful"] += int(row["followup_success"] and row["survived"])
    # This is a candidate gate, not a hardware acceptance claim. Broad recovery
    # must be demonstrated; upright births cannot substitute for fallen ones.
    missing = [family for family in ("left", "right", "inverted")
               if actual.get(family, {}).get("successful", 0) == 0]
    arcs = [row for row in report["lanes"] if row["name"].startswith("arc_")]
    arc_pass = arcs and all(row["survived"] and row["vx_mae_m_s"] is not None
        and row["vx_mae_m_s"] < .25 and row["yaw_mae_rad_s"] < .4 for row in arcs)
    pushes = [row for row in report["lanes"] if row.get("push")]
    push_pass = pushes and any(row.get("in_motion_fall") and row.get("recovered_after_push")
        and row["survived"] and row["followup_success"] for row in pushes)
    if missing or not arc_pass or not push_pass or report["passive_motor_peak_nm"] != 0.:
        raise ValueError(f"Behavior not yet adequate for this candidate: missing recovery {missing}, "
                         f"arcs={bool(arc_pass)}, moving-fall recovery={bool(push_pass)}")
    return dict(actual_settled_attitudes=actual, arcs_passed=True, moving_fall_recovery_passed=True)


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = json.loads((args.evaluation / "report.json").read_text())
    behavior = check_behavior(report, digest(args.checkpoint))
    mean, info = load_mean(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    policy = args.output / "policy.onnx"
    torch.onnx.export(mean, torch.zeros(1, 35), str(policy), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False, dynamic_axes=None)
    onnx.checker.check_model(onnx.load(policy))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 35] or session.get_outputs()[0].shape != [1, 6]:
        raise ValueError("ONNX input/output dimensions changed")
    observations = np.load(args.evaluation / "trajectory.npz")["observations"].reshape(-1, 35)
    observations = observations[np.linspace(0, len(observations)-1, min(2048, len(observations)), dtype=int)]
    worst = 0.
    with torch.inference_mode():
        for obs in observations:
            expected = mean(torch.from_numpy(obs[None])).numpy()
            actual = session.run(None, {"obs": obs[None]})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Nonfinite ONNX output")
            worst = max(worst, float(np.max(np.abs(actual - expected))))
    if worst > 5e-4:
        raise ValueError(f"ONNX parity failed: {worst}")
    contract = dict(schema="unified-getup-deploy-v1", checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), policy_sha256=digest(policy),
        completed_new_updates=info["completed_updates"], input=dict(name="obs", shape=[1, 35], dtype="float32"),
        output=dict(name="actions", shape=[1, 6], dtype="float32"), policy_hz=200,
        policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        action_clip=[13., 13., 13., 13., 9., 9.], leg_position_scale=.25, wheel_velocity_scale=10.,
        deterministic_mean=True, passive_start="zero all six motor efforts; gas spring remains active",
        previous_action_at_enable="measured wrapped motor offset / 0.25 for four legs; zero wheels",
        rmcs_required_mode="full RL from fallen state; bypass scripted leg prepare and recovery torque mixing",
        paired_leg_projection="same current 40..107 degree paired mechanism and shared winding",
        command_projection="retain yaw and cap abs(vx*yaw) at 2.06 m/s^2",
        source_training_updates_before_parent_export="unknown", ort_max_abs_error=worst,
        evaluation=str(args.evaluation.resolve()), evaluation_sha256=digest(args.evaluation / "report.json"),
        behavior=behavior, evidence_scope="Isaac Sim only", hardware_ready=False)
    (args.output / "deployment_contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False)+"\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
