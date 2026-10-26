#!/usr/bin/env python3
"""Authenticate a serial native leg-speed response diagnostic against its baseline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from rsl_rl.utils.logger import Logger

from inspect_scratch_support_breaks import inspect as inspect_original_support


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def process_matches(identity):
    try:
        value = Path(f"/proc/{identity['pid']}/stat").read_text()
    except FileNotFoundError:
        return False
    return int(value[value.rfind(")") + 2:].split()[19]) == identity["start_ticks"]


def physical_pair_checks(baseline, intervention):
    """Reject changed initial physics or any changed unselected native lane."""
    if set(baseline) != set(intervention):
        raise ValueError("Native pair has different recorded channels")
    other = list(range(7)) + list(range(14, 21))
    for key in baseline:
        if baseline[key].shape != intervention[key].shape:
            raise ValueError("Native pair has different recorded shapes: " + key)
    return dict(
        first_five_physical_samples_exact=all(np.array_equal(baseline[key][:1], intervention[key][:1])
            for key in baseline),
        all_nonselected_native_trajectories_exact=all(np.array_equal(
            np.take(baseline[key], other, axis=2 if key in ("force", "clearance", "dq", "torque") else 1),
            np.take(intervention[key], other, axis=2 if key in ("force", "clearance", "dq", "torque") else 1))
            for key in baseline))


def leg_response_scale(receipt, archived_evaluator):
    if "leg_speed_response_scale" in receipt:
        return receipt["leg_speed_response_scale"]
    if "policy_counterfactual" not in receipt and "--leg-speed-response-scale" not in archived_evaluator:
        # Older authenticated native baselines predate this diagnostic option.
        return 1.
    raise ValueError("Missing leg-response provenance in a diagnostic-capable evaluator")


def compare(baseline, intervention, identity_path):
    a, b, identity = read(baseline / "receipt.json"), read(intervention / "receipt.json"), read(identity_path)
    if Path(identity.get("run", identity.get("output", ""))).resolve() != intervention:
        raise ValueError("Native process identity belongs to another evaluation")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    counter = b["policy_counterfactual"]
    native_leg_scale = leg_response_scale(a, (baseline / "diagnostic_sources/evaluate_scratch_standing.py").read_text())
    checks = dict(
        both_native_runs_completed=a["status"] == b["status"] == "completed_unqualified",
        intervention_original_process_exited=not process_matches(identity),
        intervention_original_GPU_absent=str(identity["pid"]) not in gpu.splitlines(),
        zero_new_PPO_and_no_optimizer=a["new_PPO"] == b["new_PPO"] == 0
            and not a["optimizer_constructed"] and not b["optimizer_constructed"],
        exact_checkpoint_and_seed=a["checkpoint_sha256"] == b["checkpoint_sha256"] and a["seed"] == b["seed"],
        exact_native_physics=a["physics_runtime"] == b["physics_runtime"],
        original8s_protocol=a["evaluation_duration_s"] == b["evaluation_duration_s"] == 8,
        intervention_only_direct_leg_speed_response=native_leg_scale == 1.
            and b["leg_speed_response_scale"] in (0., .5)
            and a["wheel_response_scale"] == b["wheel_response_scale"] == 1.
            and a["external_force_mode"] == b["external_force_mode"] == "native"
            and a["reset_observation_mode"] == b["reset_observation_mode"] == "native"
            and not a["hold_initial_leg_targets"] and not b["hold_initial_leg_targets"]
            and "solver_counterfactual" not in a and "solver_counterfactual" not in b,
        all1600_policy_calls=counter["actor_calls"] == 1600 and counter["native_actor_calls_per_decision"] == 2,
        declared_selection_and_no_adoption=counter["selected_case_indices"] == list(range(7, 14))
            and counter["neutral_features"] == [16, 17, 18, 19] and counter["selected_height_m"] == .305
            and not counter["learned_policy"] and not counter["adoption_by_training"] and counter["export_forbidden"])
    paths = {}
    support_audits = {}
    for root, receipt in ((baseline, a), (intervention, b)):
        result = read(root / "evaluation/result.json")
        if result != receipt["evaluation"]:
            raise ValueError("Evaluation result and terminal receipt disagree")
        for file, expected in (("native_trace.npz", result["trace_sha256"]),
                               ("feedback_trace.npz", result["feedback_trace"]["sha256"])):
            path = root / "evaluation" / file
            if sha(path) != expected:
                raise ValueError("Original native evidence changed: " + str(path))
            paths[str(path)] = expected
        for name, expected in receipt["diagnostic_source_hashes"].items():
            path = root / "diagnostic_sources" / name
            if sha(path) != expected:
                raise ValueError("Frozen diagnostic changed: " + str(path))
            paths[str(path)] = expected
        # Recompute strict support, valid denominators and physical leg-speed
        # peaks; a pair of matching receipt copies alone is not acceptance proof.
        support_audits[str(root)] = inspect_original_support(root / "evaluation", .305)
    checks["both_original_strict_support_readouts_recomputed"] = len(support_audits) == 2
    sidecar = intervention / "evaluation/unmodified_leg_encoder_actions.npz"
    if Path(counter["original_actions"]["path"]).resolve() != sidecar or sha(sidecar) != counter["original_actions"]["sha256"]:
        raise ValueError("Original/neutral actor action sidecar changed")
    paths[str(sidecar)] = sha(sidecar)
    for field in ("capture_scratch_feedback.py", "record_scratch_physics.py"):
        checks["same_" + field.replace(".py", "")] = a["diagnostic_source_hashes"][field] == b["diagnostic_source_hashes"][field]
    other = list(range(7)) + list(range(14, 21))
    with np.load(baseline / "evaluation/feedback_trace.npz") as x, \
            np.load(intervention / "evaluation/feedback_trace.npz") as y, np.load(sidecar) as native:
        if x["observation_policy"].shape != (1600, 21, 35) or y["observation_policy"].shape != (1600, 21, 35):
            raise ValueError("Require every original8s actor observation")
        checks["all_initial_native_state_exact"] = all(np.array_equal(x[key], y[key])
            for key in x.files if key.startswith("initial_"))
        checks["first_actor_input_and_output_exact"] = all(np.array_equal(x[key][0], y[key][0])
            for key in ("observation_policy", "action_raw_policy"))
        expected_selected = np.zeros(21, dtype=bool)
        expected_selected[7:14] = True
        checks["sidecar_selection_exact"] = np.array_equal(native["selected"], expected_selected)
        full, anchor = native["native_action_policy"], native["neutral_leg_action_policy"]
        if full.shape != (1600, 21, 6) or anchor.shape != (1600, 21, 4):
            raise ValueError("Incomplete actual native/neutral action sidecar")
        expected = full.copy()
        expected[:, 7:14, :4] = anchor[:, 7:14] + b["leg_speed_response_scale"] * (full[:, 7:14, :4] - anchor[:, 7:14])
        checks["all_applied_policy_outputs_match_declared_formula"] = np.array_equal(y["action_raw_policy"], expected)
        checks["all_wheels_are_current_native_actor_outputs"] = np.array_equal(y["action_raw_policy"][:, :, 4:], full[:, :, 4:])
        checks["all_nonselected_feedback_trajectories_exact"] = all(
            np.array_equal(x[key][:, other], y[key][:, other]) for key in x.files if not key.startswith("initial_"))
        changed = int(np.any(expected[:, 7:14, :4] != full[:, 7:14, :4], axis=(1, 2)).sum())
        checks["diagnostic_actually_changes_selected_leg_outputs"] = changed > 0
    with np.load(baseline / "evaluation/native_trace.npz") as x, np.load(intervention / "evaluation/native_trace.npz") as y:
        checks.update(physical_pair_checks({key: x[key] for key in x.files}, {key: y[key] for key in y.files}))
    labels = lambda r: [(c["height_m"], c["pitch_deg"]) for c in r["evaluation"]["cases"]]
    checks["same21_original_case_labels"] = labels(a) == labels(b) and len(labels(a)) == 21

    def group(receipt):
        rows = receipt["evaluation"]["cases"][7:14]
        return dict(total_missing_physical_ms=sum(r["missing_physical_ms"] for r in rows),
            standing_pass=sum(r["standing_pass"] for r in rows),
            mean_position_vibration_3_95Hz_mrad=sum(r["leg_vibration_3_95hz_rms_mrad"] for r in rows) / 7,
            maximum_leg_speed_rad_s=max(r["leg_speed_peak_rad_s"] for r in rows),
            mean_forward_speed_rms_m_s=sum(r["forward_speed_rms_m_s"] for r in rows) / 7)

    return dict(schema="scratch-leg-encoder-native-comparison-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
        baseline=str(baseline), intervention=str(intervention), identity=str(identity_path),
        checks=checks, all_checks_pass=all(checks.values()), physical_attribution_allowed=all(checks.values()),
        baseline_mid=group(a), intervention_mid=group(b), changed_selected_leg_output_frames=changed,
        completed_updates=b["completed_updates"], response_scale=b["leg_speed_response_scale"],
        sources_sha256=dict(paths, **{str(identity_path): sha(identity_path),
            str(baseline / "receipt.json"): sha(baseline / "receipt.json"),
            str(intervention / "receipt.json"): sha(intervention / "receipt.json")}),
        original_strict_acceptance_retained=True, new_PPO=0, native_steps_by_reader=0,
        optimizer_constructed=False, adoption_by_training=False, behavior_qualification=False, hardware_ready=False,
        scope="An intervention is comparable only if all source, initial-state and untouched-population checks pass. "
            "Outcomes retain the original strict force>2N from first1ms; an ablated function never becomes a qualified policy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, events = args.output.resolve(), args.output.resolve().with_suffix(".events")
    if output.exists() or events.exists() or output.with_suffix(".source.py").exists():
        raise ValueError("Preserve prior comparison, exact reader and RSL events")
    baseline, intervention = args.baseline.resolve(), args.intervention.resolve()
    report = compare(baseline, intervention, args.identity.resolve())
    receipt = read(intervention / "receipt.json")
    run = Path(receipt["checkpoint"]).parent
    logger = Logger(log_dir=str(events), cfg=read(run / "agent_config.json"),
        env_cfg=read(intervention / "config/contract.json"), num_envs=1, is_distributed=False,
        gpu_world_size=1, gpu_global_rank=0, device="cpu")
    logger.init_logging_writer()
    try:
        for group in ("baseline_mid", "intervention_mid"):
            for key, value in report[group].items():
                logger.writer.add_scalar("DiagnosticLegEncoderNative/" + group + "/" + key,
                                         value, report["completed_updates"])
        logger.writer.add_scalar("DiagnosticLegEncoderNative/all_provenance_checks_pass",
                                 int(report["all_checks_pass"]), report["completed_updates"])
        logger.writer.flush()
    finally:
        logger.writer.close()
    source = Path(__file__).read_bytes()
    report.update(reader_sha256=hashlib.sha256(source).hexdigest(), tensorboard=dict(directory=str(events),
        files_sha256={event.name: sha(event) for event in events.glob("events.out.tfevents.*")}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".source.py").write_bytes(source)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("all_checks_pass", "checks", "baseline_mid", "intervention_mid")}))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
