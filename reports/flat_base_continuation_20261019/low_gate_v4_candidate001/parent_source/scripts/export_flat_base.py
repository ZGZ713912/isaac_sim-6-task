#!/usr/bin/env python3
"""Export a descendant of the supplied actor with its paired-107 RMCS contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate_release(primary, manifest, plan, checkpoint_sha, root=ROOT):
    """An intermediate replay cannot qualify a deployment-clock release."""
    from wheeled_tasks.chassis.flat_base_continuation import digest
    if manifest.get("schema") != "flat-base-behavior-release-v4" or manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Release evidence does not identify this exact checkpoint")
    basic = set(plan["pilot_case_names"])
    extended = basic | {"forward_1", "backward_1", "forward_2", "backward_2", "forward_3", "backward_3",
        "rotate_4", "rotate_4_reverse", "curve_low", "curve_low_reverse", "height_scan", "start_stop_1",
        "start_stop_2", "start_stop_3"}
    default_height = manifest.get("default_height_contract", {})
    height = default_height.get("height_m")
    if (default_height.get("schema") != "flat-rmcs-packaged-default-height-v1"
            or type(height) not in (int, float) or not math.isfinite(height) or not .23 <= height <= .43):
        raise ValueError("Release evidence lacks the authenticated packaged default height")
    maximum_height = manifest.get("maximum_height_contract", {})
    high = maximum_height.get("height_m")
    bounds = default_height.get("command_height_bounds_m")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
            or not .23 <= bounds[0] <= height <= bounds[1] <= .43
            or maximum_height.get("schema") != "flat-rmcs-packaged-maximum-height-v1"
            or type(high) not in (int, float) or not math.isfinite(high) or high != bounds[1]
            or any(maximum_height.get(k) != default_height.get(k)
                   for k in ("command_height_bounds_m", "source_file", "source_sha256"))):
        raise ValueError("Release evidence lacks the authenticated packaged maximum height")
    catalogue = json.loads((ROOT / plan["source_worker_file"]).read_text())["evaluation"]["cases"]
    commands = {case["name"]: case["command"] for case in catalogue}
    cohorts = [("training_basic", primary, 200, plan["evaluation_seed"], basic)]
    receipts = {}
    for name, hz, seed, cases in (
            ("training_envelope", 200, plan["evaluation_seed"], extended),
            ("deployment_basic", 1000, plan["evaluation_seed"], basic),
            ("deployment_envelope", 1000, plan["evaluation_seed"], extended),
            ("confirmation_envelope", 1000, plan["confirmation_seed"], extended),
            ("rmcs_reference_envelope", 1000, plan["evaluation_seed"], extended),
            ("rmcs_reference_confirmation", 1000, plan["confirmation_seed"], extended),
            ("rmcs_default_height_envelope", 1000, plan["evaluation_seed"], extended),
            ("rmcs_default_height_confirmation", 1000, plan["confirmation_seed"], extended),
            ("rmcs_maximum_height_envelope", 1000, plan["evaluation_seed"], extended),
            ("rmcs_maximum_height_confirmation", 1000, plan["confirmation_seed"], extended)):
        entry = manifest["evaluations"].get(name)
        if not entry:
            raise ValueError(f"Release evidence is missing a required cohort: {name}")
        path = root / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Release evidence changed: {name}")
        cohorts.append((name, json.loads(path.read_text()), hz, seed, cases))
        receipts[name] = dict(file=str(path.resolve()), sha256=entry["sha256"])
    for name, evaluation, hz, seed, cases in cohorts:
        if (evaluation.get("status") != "evaluated" or evaluation.get("checkpoint_sha256") != checkpoint_sha
                or evaluation.get("source_onnx_sha256") != plan["source_onnx_sha256"]
                or evaluation.get("asset_manifest_sha256") != plan["asset_manifest_sha256"]
                or evaluation.get("model_spec_sha256") != plan["model_spec_sha256"]
                or evaluation.get("policy_hz") != 50 or evaluation.get("physics_hz") != hz
                or evaluation.get("seed") != seed or evaluation.get("new_optimizer_updates") != 0):
            raise ValueError(f"Release cohort has a different model, clock or seed: {name}")
        result = evaluation["result"]
        protocol = "flat-base-paired107-fixed-v1" if cases == basic else "flat-base-paired107-envelope-v2"
        if name.startswith("rmcs_"):
            protocol = "flat-base-paired107-rmcs-reference-envelope-v1"
            reference = manifest.get("command_reference")
            if (not reference or evaluation.get("rmcs_command_reference") != reference
                    or reference.get("schema") != "flat-rmcs-command-reference-v1"
                    or reference.get("planar_slew") != "none; next policy tick"
                    or reference.get("height_command_is_reference") is not True):
                raise ValueError(f"Release cohort does not bind the RMCS command-reference path: {name}")
            if name.startswith("rmcs_default_height_"):
                protocol = "flat-base-paired107-rmcs-default-height-envelope-v1"
                if evaluation.get("packaged_default_height") != default_height:
                    raise ValueError(f"Release cohort does not bind the packaged default height: {name}")
            if name.startswith("rmcs_maximum_height_"):
                protocol = "flat-base-paired107-rmcs-maximum-height-envelope-v1"
                if evaluation.get("packaged_maximum_height") != maximum_height:
                    raise ValueError(f"Release cohort does not bind the packaged maximum height: {name}")
        if (not result["passed"] or set(result["cases"]) != cases
                or evaluation["settings"]["protocol_id"] != protocol):
            raise ValueError(f"Release cohort has incomplete behavior coverage: {name}")
        settings_cases = {case["name"]:case for case in evaluation["settings"]["cases"]}
        for case_name, case in result["cases"].items():
            expected = list(commands[case_name])
            if name.startswith("rmcs_default_height_") and any(abs(v) > 0 for v in expected[:2]):
                expected[2] = height
            if name.startswith("rmcs_maximum_height_") and any(abs(v) > 0 for v in expected[:2]):
                expected[2] = high
            if case["command"] != expected or settings_cases.get(case_name, {}).get("command") != expected:
                raise ValueError(f"Release cohort has a different command or body height: {name}/{case_name}")
            stationary = case["command"][:2] == [0., 0.]
            acceptance = plan["acceptance"]
            numeric = ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "closure_gap_max_m",
                       "tilt_max_deg", "stand_drift_max_m")
            if (case["episodes"] != 4 or case["requested_episodes"] != 4 or case["full_horizon_episodes"] != 4
                    or not case["passed"] or not all(case["checks"].values())
                    or not all(math.isfinite(case[key]) and case[key] >= 0 for key in numeric)
                    or case["vx_mae_m_s"] > acceptance["stand_velocity_mae_m_s_max" if stationary else "velocity_mae_m_s_max"]
                    or case["yaw_mae_rad_s"] > acceptance["yaw_mae_rad_s_max"]
                    or case["height_mae_m"] > acceptance["height_mae_m_max"]
                    or case["closure_gap_max_m"] > acceptance["closure_gap_max_m"]
                    or case["tilt_max_deg"] > acceptance["tilt_max_deg"]
                    or stationary and case["stand_drift_max_m"] > acceptance["stand_drift_m_max"]):
                raise ValueError(f"Release cohort has an unqualified or censored case: {name}")
    return receipts



def read_checkpoint_plan(root, info):
    """Bind export timing and source provenance to the checkpoint's actual plan."""
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, digest
    plan_file = info.get("plan_file")
    if not isinstance(plan_file, str) or not plan_file:
        raise ValueError("Checkpoint is missing its authenticated training plan")
    plan = read_plan(root, plan_file)
    if digest(Path(root) / plan_file) != info.get("plan_sha256"):
        raise ValueError("Saved checkpoint plan changed before export")
    return plan

def main():
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    import torch
    import yaml
    from wheeled_algo.onnx_actor_import import recover_actor
    from wheeled_algo.command_residual import KIND, mean_from_checkpoint
    from wheeled_algo.guarded_feedback_residual import KIND as FEEDBACK_KIND, TRACKING_KIND
    from wheeled_algo.separated_feedback import KIND as SEPARATED_KIND
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, digest, encode

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-evidence", type=Path,
                        help="Exact passing 200/1000Hz and RMCS-reference development/heldout evidence; otherwise intermediate")
    parser.add_argument("--rmcs", type=Path, default=Path("/home/noir/Documents/workspace/RMCS"))
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    info = state["infos"]
    plan = read_checkpoint_plan(ROOT, info)
    if (info["flat_base_source_onnx_sha256"] != plan["source_onnx_sha256"]
            or info["asset_manifest_sha256"] != plan["asset_manifest_sha256"]
            or info["control_math_sha256"] != plan["control_sha256"]):
        raise ValueError("Checkpoint does not belong to the matched flat actor lineage")
    evaluation = json.loads(args.evaluation.read_text())
    if (evaluation["status"] != "evaluated" or evaluation["checkpoint_sha256"] != digest(args.checkpoint)
            or evaluation["model_spec_sha256"] != plan["model_spec_sha256"]):
        raise ValueError("The exact candidate must have completed a matching behavior evaluation")
    release = None
    if args.release_evidence:
        from flat_rmcs_command_reference import read_reference_contract
        from flat_default_height_profile import read_default_height_contract
        from flat_maximum_height_profile import read_maximum_height_contract
        release_manifest = json.loads(args.release_evidence.read_text())
        reference_contract = read_reference_contract(args.rmcs)
        if release_manifest.get("command_reference") != reference_contract:
            raise ValueError("Release manifest differs from the inspected current RMCS command-reference contract")
        if release_manifest.get("default_height_contract") != read_default_height_contract(args.rmcs, reference_contract):
            raise ValueError("Release manifest differs from the current RMCS packaged default height")
        if release_manifest.get("maximum_height_contract") != read_maximum_height_contract(args.rmcs, reference_contract):
            raise ValueError("Release manifest differs from the current RMCS packaged maximum height")
        release = validate_release(evaluation, release_manifest, plan, digest(args.checkpoint))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / plan["actor_bundle"] / "source.onnx"
    _, recovered = recover_actor(source)
    mean = {k: v for k, v in state["actor_state_dict"].items() if k.startswith("mlp.")}
    actor = mean_from_checkpoint(state, source)
    if info.get("actor_adaptation") in (KIND, FEEDBACK_KIND, TRACKING_KIND, SEPARATED_KIND):
        graph = actor.export_graph(source)
    else:
        graph = onnx.load(str(source))
        replacements = {p["onnx_name"]: mean[p["torch_name"]].numpy() for p in recovered["parameters"]}
        for initializer in graph.graph.initializer:
            initializer.CopyFrom(numpy_helper.from_array(replacements[initializer.name], name=initializer.name))
    rmcs_rl = args.rmcs / "rmcs_ws/src/rmcs_rl"
    yaml_path = args.rmcs / "rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-rl.yaml"
    yaml_text = yaml_path.read_text()
    parameters = yaml.safe_load(yaml_text)["rl_controller"]["ros__parameters"]
    if (parameters["policy_rate"], parameters["rl_obs_size"], parameters["rl_action_size"],
            parameters["history_length"], parameters["leg_winding_version"], parameters["expected_model_spec_sha256"]) != (
            50., 35, 6, 1, plan["leg_target_winding"], plan["model_spec_sha256"]):
        raise ValueError("Current RMCS YAML interface changed")
    observation = parameters["observation_terms"]
    action = parameters["action_terms"]
    layout_json = json.dumps(dict(observation=observation, action=action), separators=(",", ":"), sort_keys=True)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update(leg_winding_version=plan["leg_target_winding"], leg_target_model_spec_sha256=plan["model_spec_sha256"],
        rmcs_model_spec_sha256=plan["model_spec_sha256"], rmcs_policy_hz="50", rmcs_history_length="1",
        rmcs_obs_frame_size="35", rmcs_obs_clip="100", rmcs_obs_layout=json.dumps(observation, separators=(",", ":")),
        rmcs_actions_layout=json.dumps(action, separators=(",", ":")),
        rmcs_layout_hash=hashlib.sha256(layout_json.encode()).hexdigest(),
        source_onnx_sha256=plan["source_onnx_sha256"], source_checkpoint_sha256=digest(args.checkpoint),
        actor_transfer_optimizer_updates=str(info["successful_updates_total"]),
        actor_transfer_actor_updates=str(info["actor_updates_total"]))
    if info.get("actor_adaptation"):
        metadata.update(actor_adaptation=info["actor_adaptation"],
                        active_branch_actor_updates=str(info["active_branch_actor_updates"]),
                        local_update_count_semantics="all local trial branches; active_branch_actor_updates counts this residual branch only",
                        original_feedback_source_onnx_sha256=plan["source_onnx_sha256"])
    if info.get("feedback_branch_actor_updates") is not None:
        metadata["feedback_branch_actor_updates"] = str(info["feedback_branch_actor_updates"])
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    output = args.output / "policy.onnx"
    onnx.save(graph, str(output))
    traces = np.load(args.evaluation.parent / "traces.npz")
    values = traces["observations"][::10].reshape(-1, 35)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=session_options, providers=["CPUExecutionProvider"])
    expected = actor(torch.from_numpy(values)).detach().numpy()
    actual = session.run(None, {"obs": values})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    # Compile the actual current C++ loader; no ROS or motor code is run.
    ort_root = args.rmcs / "rmcs_ws/build/rmcs_rl/onnxruntime-linux-x64-1.20.0"
    binary = args.output / "rmcs_policy_check"
    source_files = [ROOT / "scripts/cpp/flat_base_rmcs_check.cpp", rmcs_rl / "src/policy/onnx_runtime.cpp",
                    rmcs_rl / "src/policy/policy_model.cpp"]
    subprocess.run(["g++", "-std=c++23", "-O2", f"-I{rmcs_rl / 'src'}", f"-I{rmcs_rl / 'include'}",
        f"-I{ort_root / 'include'}", *map(str, source_files), f"-L{ort_root / 'lib'}", f"-Wl,-rpath,{ort_root / 'lib'}",
        "-lonnxruntime", "-o", str(binary)], check=True)
    command = [str(binary), str(output), "35", plan["model_spec_sha256"], plan["leg_target_winding"]]
    fixture = "\n".join(" ".join(format(float(v), ".9g") for v in row) for row in values)
    native = subprocess.run(command, input=fixture, text=True, capture_output=True, check=True)
    cpp = np.fromstring(native.stdout, sep=" ").reshape(-1, 6)
    np.testing.assert_allclose(cpp, actual, rtol=2e-5, atol=2e-5)
    negative = {}
    for name, index, value in (("wrong_obs_size", 2, "34"), ("wrong_model_spec", 3, "0" * 64),
                               ("wrong_winding", 4, "independent_nearest_v1")):
        bad = command.copy()
        bad[index] = value
        test = subprocess.run(bad, input="", text=True, capture_output=True)
        negative[name] = dict(rejected=test.returncode != 0, error=test.stderr.strip())
    if not all(row["rejected"] for row in negative.values()):
        raise RuntimeError("Current RMCS loader accepted an invalid contract")
    # Keep the commissioning state and cap every command producer to the
    # assessed envelope in this separate package; the RMCS tree is untouched.
    target_yaml, count = re.subn(r'rl_model_path: "[^"]+"', 'rl_model_path: "models/wheel_leg/flat_finetuned.onnx"', yaml_text)
    if count != 1:
        raise ValueError("Unexpected RMCS model-path declaration")
    for key, value, expected_count in (("vx_max", 3., 2), ("yaw_rate_max", 4., 2),
            ("heading_hold_yaw_rate_max", 4., 1), ("spin_yaw_rate", 4., 1)):
        target_yaml, count = re.subn(rf'(?m)^(    {key}:) [^\n]+$', rf'\g<1> {value}', target_yaml)
        if count != expected_count:
            raise ValueError(f"Unexpected RMCS {key} declarations")
    target_yaml = target_yaml.replace("# Use dev's V7 real-robot command envelope.",
        "# Flat continuation package: |vx| <= 3 m/s and |yaw| <= 4 rad/s.")
    (args.output / "wheel-leg-flat-finetuned.yaml").write_text(target_yaml)
    report = dict(schema="flat-base-rmcs-export-v1", model_sha256=digest(output),
        source_onnx_sha256=plan["source_onnx_sha256"], checkpoint_sha256=digest(args.checkpoint),
        successful_updates_total=info["successful_updates_total"], actor_updates_total=info["actor_updates_total"],
        actor_adaptation=info.get("actor_adaptation"), active_branch_actor_updates=info.get("active_branch_actor_updates"),
        feedback_branch_actor_updates=info.get("feedback_branch_actor_updates"),
        local_count_semantics="all local trial branches; original training count remains unknown",
        consumed_optimizer_updates_total=info["consumed_optimizer_updates_total"],
        evaluation_sha256=digest(args.evaluation), behavior_passed=evaluation["result"]["passed"],
        evaluated_case_names=list(evaluation["result"]["cases"]), input_shape=["batch", 35], output_shape=["batch", 6],
        layout_sha256=metadata["rmcs_layout_hash"], source_rmcs_yaml_sha256=digest(yaml_path),
        source_loader_sha256={str(p): digest(p) for p in source_files}, verification_samples=len(values),
        pytorch_onnx_max_abs_error=float(np.abs(actual - expected).max()),
        rmcs_cpp_python_ort_max_abs_error=float(np.abs(cpp - actual).max()), negative_contract_tests=negative,
        deployment_interface_verified=True, behavior_qualified=release is not None,
        export_kind="behavior_qualified_simulation_candidate" if release is not None else "intermediate",
        release_evaluation_receipts=release, release_evidence_sha256=digest(args.release_evidence) if release is not None else None,
        packaged_command_limits=dict(vx_m_s=3., yaw_rad_s=4., height_m=[.23, .42]),
        physical_robot_tested=False, hardware_motion_started=False,
        training_plan_file=info["plan_file"], training_plan_sha256=info["plan_sha256"],
        training_physics_hz=round(1 / plan["physics_dt"]),
        control_timing_scope=f"50Hz actor; {round(1 / plan['physics_dt'])}Hz training; frozen200Hz baseline and1000Hz deployment cohorts required for behavior qualification")
    (args.output / "verification.json").write_bytes(encode(report))
    (args.output / "evaluation.json").write_bytes(args.evaluation.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
