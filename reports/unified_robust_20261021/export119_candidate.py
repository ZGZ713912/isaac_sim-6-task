"""Explicit user-requested 115 trial export. Formal admission remains unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from runtime116_binding_audit import verify_configuration
from verify_unified_test_export import verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(1)
    checkpoint = ROOT / "runs/unified_robust_20261022/composed_getup115/model_migrated.pt"
    if sha(checkpoint) != "11c4451ab73c4935051fc0187a5fc3a70a3534cee67f22d393500f623bd9f121":
        raise ValueError("Selected115 checkpoint changed")
    mean, info = load_mean(checkpoint)
    observation_contract = info["settings"]["standing_position_observation"]
    runtime = Path("/home/noir/Documents/workspace/RMCS/.codex/worktrees/unified-standing-v3")
    evidence = json.loads((REPORTS / "runtime085_final.json").read_text())
    for relative, expected in evidence["source_files"].items():
        if sha(runtime / relative) != expected:
            raise ValueError("Runtime source changed: " + relative)
    for path, expected in ((runtime, evidence["rmcs_commit"]),
                           (runtime / "rmcs_ws/src/rmcs_rl", evidence["child_commit"])):
        if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != expected:
            raise ValueError("Runtime commit changed")
        if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "-uno"], text=True):
            raise ValueError("Runtime tracked files are dirty")
    yaml_source = runtime / "rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-unified-rl.yaml"
    configuration = yaml.safe_load(yaml_source.read_text())
    observation_sha = verify_configuration(configuration, observation_contract)
    if observation_sha != evidence["model_contract_sha256"] or not info["settings"]["paired_feedback_winding"]:
        raise ValueError("Runtime observation or paired-feedback binding differs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence").mkdir()
    (output / "runtime").mkdir()
    policy = output / "policy.onnx"
    torch.onnx.export(mean, torch.zeros(1, 35), str(policy), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(policy)
    onnx.helper.set_model_props(model, dict(policy_hz="200", observation_dim="35", action_dim="6",
        actor_adaptation="wheel-leg-robust-getup-v2", motor_control="full_RL_after_passive_enable",
        source_checkpoint_sha256=sha(checkpoint), leg_winding_version="paired_branch107_v1",
        leg_target_model_spec_sha256=observation_contract["model_spec_sha256"],
        rmcs_observation_contract_sha256=observation_sha,
        usage="USER_AUTHORIZED_EXPERIMENTAL_VEHICLE_TEST", behavior_qualified="false",
        standing_position_slots="14:16", standing_position_scale="20_per_metre",
        recovery_coordinates="smooth_absolute_knee_circular_common"))
    onnx.save(model, policy)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    checked, worst, vectors, rows = 0, 0., [], []
    sources = []
    for name in ("composed_static_original115", "composed_static_holdout115", "dynamic_recovery_full079"):
        directory = ROOT / "runs/unified_robust_20261022" / name / "robust"
        report = json.loads((directory / "report.json").read_text())
        path = directory / "trajectory.npz"
        if sha(path) != report["trajectory_sha256"]:
            raise ValueError("Recorded input trajectory changed")
        with np.load(path) as data:
            observations = data["observations"]
            sample = observations[np.linspace(0, len(observations) - 1, 48, dtype=int)].reshape(-1, 35)
        rows.append((name, np.ascontiguousarray(sample[np.isfinite(sample).all(-1)], dtype=np.float32)))
        sources.append(dict(name=name, trajectory_sha256=sha(path), checkpoint_sha256=report["checkpoint_sha256"],
            interpretation="native115 standing" if name.endswith("115") else "079 sensor inputs only; not native115 recovery evidence"))
    parity = ROOT / "runs/unified_robust_20261022/runtime085_frames_formatted/parity.npz"
    frame_evidence = json.loads((REPORTS / "runtime085_frame_formatted.json").read_text())
    if sha(parity) != frame_evidence["files"][str(parity)]:
        raise ValueError("C++ frame fixture changed")
    with np.load(parity) as data:
        rows.append(("cpp_full35_frames", np.ascontiguousarray(data["actual"][data["accepted"], :35], dtype=np.float32)))
    for name, observations in rows:
        with torch.inference_mode():
            expected = mean(torch.from_numpy(observations)).numpy()
        local_worst = 0.
        choose = set(np.linspace(0, len(observations) - 1, min(64, len(observations)), dtype=int).tolist())
        for index, obs in enumerate(observations):
            actual = session.run(None, {"obs": obs[None]})[0][0]
            if not np.isfinite(actual).all() or (np.abs(actual) > np.asarray([16.] * 4 + [9.] * 2) + 1e-5).any():
                raise ValueError("Nonfinite or out-of-contract exported action")
            error = float(np.max(np.abs(actual - expected[index])))
            local_worst = max(local_worst, error)
            if index in choose:
                vectors.append(dict(source=name, observation=obs.tolist(), expected_raw_action=expected[index].tolist()))
        if local_worst > 1e-4:
            raise ValueError(f"ONNX/Torch parity failed: {name} {local_worst}")
        checked += len(observations)
        worst = max(worst, local_worst)
        sources.append(dict(parity_input=name, observations=len(observations), maximum_raw_action_error=local_worst))
        print(json.dumps(sources[-1]), flush=True)
    for _ in range(20):
        session.run(None, {"obs": rows[0][1][0:1]})
    timings = []
    for _ in range(512):
        start = time.perf_counter()
        session.run(None, {"obs": rows[0][1][0:1]})
        timings.append((time.perf_counter() - start) * 1000.)
    write(output / "inference_vectors.json", dict(model_sha256=sha(policy), max_abs_tolerance=1e-4, vectors=vectors))
    shutil.copy2(checkpoint, output / "checkpoint.pt")
    shutil.copy2(ROOT / "scripts/verify_unified_test_export.py", output / "verify_export.py")
    profile = yaml_source.read_text()
    profile = profile.replace("# Candidate full-RL 200 Hz profile; not the active vehicle configuration.",
        "# User-requested115 experimental vehicle-test profile; behavior remains unqualified.")
    profile = profile.replace("# Requires a qualified unified_candidate.onnx with the exact observation hash.",
        "# Requires matching runtime commits and the exact115 observation hash.")
    profile = profile.replace("# Full-RL candidate is delivered only after behavior and runtime qualification.",
        "# Explicit user trial; see deployment_contract.json for incomplete behavior evidence.")
    profile = profile.replace("models/wheel_leg/unified_candidate.onnx", "models/wheel_leg/unified115_test.onnx")
    write_contract_sha = verify_configuration(yaml.safe_load(profile), observation_contract)
    assert write_contract_sha == observation_sha
    (output / "wheel-leg-infantry-unified115-test.yaml").write_text(profile)
    for label, path, base, commit in (
        ("rmcs-parent", runtime, "5a7867ec", evidence["rmcs_commit"]),
        ("rmcs_rl-child", runtime / "rmcs_ws/src/rmcs_rl", "22ac9e5", evidence["child_commit"])):
        patch = subprocess.check_output(["git", "-C", str(path), "format-patch", "--stdout", "--binary", base + ".." + commit])
        (output / "runtime" / (label + ".patch")).write_bytes(patch)
    for relative in ("iteration115_final.json", "iteration115_migration.json", "iteration079_final.json",
                     "runtime085_final.json", "runtime116_binding_audit.json", "gui118_visible003.json",
                     "standing117_composed_static_original115.json", "standing117_composed_static_holdout115.json",
                     "acceptance_protocol004.json"):
        shutil.copy2(REPORTS / relative, output / "evidence" / relative)
    shutil.copy2(runtime / "docs/WHEEL_LEG_UNIFIED_STANDING_V3.md", output / "runtime" / "observation_runtime_v3.md")
    layout = [(0, 3, "body vx,vy,yaw commands", 1.), (3, 1, "requested height metres", 5.),
        (4, 3, "body gyro rad/s", .5), (7, 3, "body projected gravity", 1.),
        (10, 4, "wrapped leg q-nominal in policy order", 1.),
        (14, 2, "filtered relative standing wheel-coordinate displacement metres", 20.),
        (16, 6, "joint velocity rad/s in policy order", .1),
        (22, 6, "previous APPLIED raw actions in policy order", 1.),
        (28, 7, "fixed ordinary context [1,0,0,0,0,0,0]", 1.)]
    contract = dict(schema="unified-getup-user-trial-v1", model_sha256=sha(policy), checkpoint_sha256=sha(checkpoint),
        authorization="2026-10-23 user explicitly requested ONNX for their vehicle testing",
        usage="experimental vehicle test performed by user", behavior_qualified=False, hardware_tested=False,
        full_training_admitted=False, formal_exporter_gate_unchanged=True,
        actor_dim=35, action_dim=6, history_length=1, policy_hz=200, physical_pd_hz=1000,
        deterministic_mean=True, standing_position_observation=observation_contract,
        observation_contract_sha256=observation_sha, leg_target_model_spec_sha256=observation_contract["model_spec_sha256"],
        observation_layout=[dict(start=a, count=b, value=c, scale=d) for a,b,c,d in layout],
        policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        nominal_model_pos=[-.42, .13742282595395358, .42, -.1374155762580851, 0., 0.],
        leg_action_clip=16., wheel_action_clip=9., leg_position_scale=.25, wheel_velocity_scale=10.,
        leg_kp=160., leg_kd=2.5, wheel_kp=.6, leg_torque_cap_nm=40., wheel_torque_cap_nm=4.5,
        leg_winding_version="paired_branch107_v1", quiet_filter="inside graph; no additional output filter",
        standing_position_input_to_actor=True, quiet_output_wheel_alpha=info["settings"]["quiet_output_wheel_alpha"],
        runtime=dict(worktree=str(runtime), parent_commit=evidence["rmcs_commit"], child_commit=evidence["child_commit"],
            parent_base="5a7867ec", child_base="22ac9e5", source_files=evidence["source_files"],
            checked_current_sources=True, model_destination="rmcs_ws/src/rmcs_rl/models/wheel_leg/unified115_test.onnx",
            profile="wheel-leg-infantry-unified115-test", original_yaml_sha256=sha(yaml_source)),
        behavior_limits=["96/96 standing episodes completed30s under tested random domains; not infinite-time or hardware evidence",
            "Heldout low stance maximum6.592cm exceeds existing5cm gate; mid3.586cm/high5.679cm",
            "Latest115 GUI left symmetric trial terminated at0.815s without recovery",
            "115 full random recovery, two matched nominal suites and stationary lifecycle were not run after static failure",
            "079 recovery parent nominal strong right moving push0/16; parent success counts do not qualify115",
            "Self-collision disabled; randomized parameters not identified on vehicle"],
        suggested_initial_height_m=.305, long_term_stability_guaranteed=False,
        candidate_ppo_lineage=569, experiment_ppo_consumed=1384, experiment_offline_steps=128,
        new_ppo_updates=0, new_offline_steps=0, training_domain_settings=info["settings"], source_sha256=info["source_sha256"],
        parity=dict(observations=checked, maximum_raw_action_error=worst, sources=sources,
            description="Graph fidelity on recorded115 standing,079 dynamic inputs and C++ fixtures; not new dynamics"),
        cpu_inference_ms=dict(samples=512, threads=1, p50=float(np.median(timings)), p99=float(np.quantile(timings,.99))),
        exporter_sha256=sha(__file__), portable_verifier_sha256=sha(output / "verify_export.py"))
    write(output / "deployment_contract.json", contract)
    write(output / "offline_validation.json", verify(output))
    print(json.dumps(dict(output=str(output), model_sha256=sha(policy), observations=checked,
        maximum_raw_action_error=worst, behavior_qualified=False, hardware_tested=False)), flush=True)


if __name__ == "__main__":
    main()
