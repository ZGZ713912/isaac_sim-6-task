#!/usr/bin/env python3
"""Authenticate existing native results and measure task coverage; no simulation or learning."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/robust_policy_repair_20261024"
OUTPUT = REPORTS / "capability_status_20261026_read001.json"
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def authenticate(path, expected):
    digest = sha(path)
    if digest != expected:
        raise ValueError(f"Changed original evidence: {path}")
    INPUTS[str(path.relative_to(ROOT))] = digest


def runs_of_true(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return len(starts), int((ends - starts).max(initial=0))


def scratch(run):
    result = read(run / "evaluation/result.json")
    trace = run / "evaluation/native_trace.npz"
    authenticate(trace, result["trace_sha256"])
    extras, field_checks = [], []
    with np.load(trace, allow_pickle=False) as z:
        for i, c in enumerate(result["cases"]):
            valid = z["valid"][:, i]
            force = z["force"][:, :, i][valid]
            support = (np.linalg.norm(force, axis=-1) > 2.).all(-1).reshape(-1)
            dq = z["dq"][:, :, i][valid]
            q = z["q"][:, i][valid]
            late = q[600:] if len(q) > 600 else q
            frequency = np.fft.rfftfreq(len(late), .005)
            spectrum = np.fft.rfft(late - late.mean(0), axis=0)
            spectrum[(frequency < 3.) | (frequency > 95.)] = 0.
            band = np.fft.irfft(spectrum, n=len(late), axis=0)
            position = z["position"][:, i][valid]
            velocity = z["velocity"][:, i][valid]
            tilt = np.degrees(np.arccos(np.clip(-z["gravity"][:, i, 2][valid], -1., 1.)))
            measured = dict(
                full_horizon=len(q) == 1600, physical_ms=len(support),
                missing_physical_ms=int((~support).sum()),
                dual_support_fraction=float(support.mean()),
                first20ms_continuous=bool(support[:20].all()),
                maximum_wheel_clearance_m=float(z["clearance"][:, :, i][valid].max()),
                leg_speed_peak_rad_s=float(np.abs(dq).max()),
                leg_speed_rms_rad_s=float(np.sqrt(np.mean(dq ** 2))),
                leg_vibration_3_95hz_rms_mrad=float(np.sqrt(np.mean(band ** 2)) * 1000),
                height_mae_m=float(np.abs(position[:, 2] - c["height_m"]).mean()),
                tilt_peak_deg=float(tilt.max()),
                closure_gap_peak_m=float(z["gap"][:, i][valid].max()),
                forward_speed_rms_m_s=float(np.sqrt(np.mean(velocity[:, 0] ** 2))))
            measured["continuous_support_pass"] = measured["full_horizon"] and not measured["missing_physical_ms"]
            measured["standing_pass"] = bool(measured["continuous_support_pass"]
                and measured["tilt_peak_deg"] < 10. and measured["height_mae_m"] < .01
                and measured["forward_speed_rms_m_s"] < .03
                and measured["leg_speed_peak_rad_s"] < 3. and measured["closure_gap_peak_m"] < .003)
            for k, v in measured.items():
                ok = c[k] == v if isinstance(v, (bool, int)) else math.isclose(c[k], v, rel_tol=2e-6, abs_tol=2e-8)
                if not ok:
                    raise ValueError(f"Native result mismatch: {run.name} case{i} {k}: {c[k]} vs {v}")
                field_checks.append(ok)
            delta = position[:, :2] - np.asarray(result["initial_root_positions"])[i, :2]
            radius = np.linalg.norm(delta, axis=1)
            events, longest = runs_of_true(~support)
            extras.append(dict(height_m=c["height_m"], pitch_deg=c["pitch_deg"],
                support_loss_events=events, longest_support_loss_ms=longest,
                max_xy_excursion_m=float(radius.max()), terminal_xy_offset_m=float(radius[-1]),
                position_sampling_hz=200, support_and_dq_sampling_hz=1000))
    groups = []
    for height in sorted({c["height_m"] for c in result["cases"]}):
        cases = [c for c in result["cases"] if c["height_m"] == height]
        extra = [c for c in extras if c["height_m"] == height]
        groups.append(dict(height_m=height, cases=len(cases),
            standing_pass=sum(c["standing_pass"] for c in cases),
            total_missing_ms=sum(c["missing_physical_ms"] for c in cases),
            worst_dual_support_fraction=min(c["dual_support_fraction"] for c in cases),
            mean_leg_vibration_mrad=float(np.mean([c["leg_vibration_3_95hz_rms_mrad"] for c in cases])),
            maximum_leg_speed_rad_s=max(c["leg_speed_peak_rad_s"] for c in cases),
            mean_height_mae_m=float(np.mean([c["height_mae_m"] for c in cases])),
            maximum_tilt_deg=max(c["tilt_peak_deg"] for c in cases),
            mean_forward_speed_rms_m_s=float(np.mean([c["forward_speed_rms_m_s"] for c in cases])),
            maximum_xy_excursion_m=max(c["max_xy_excursion_m"] for c in extra),
            maximum_terminal_xy_offset_m=max(c["terminal_xy_offset_m"] for c in extra),
            support_loss_events=sum(c["support_loss_events"] for c in extra),
            longest_support_loss_ms=max(c["longest_support_loss_ms"] for c in extra)))
    return dict(run=str(run.relative_to(ROOT)), groups=groups, per_case_extra=extras,
                independently_recomputed_case_fields=len(field_checks), all_case_fields_match=all(field_checks))


if OUTPUT.exists():
    raise FileExistsError("Preserve the existing authenticated observation")

run = ROOT / "runs/scratch_standing_20261026/pilot009_leg_velocity_separated"
progress = read(run / "progress.json")
identity = read(REPORTS / "scratch_pilot009_launch_identity001.json")
seal = read(REPORTS / "scratch_pilot009_closed_seal001.json")
behavior = read(REPORTS / "scratch_pilot009_behavior001.json")
manifest = read(run / "source_hashes.json")
revision = read(run / "source_revision.json")
authenticate(run / "model_final.pt", seal["final_checkpoint_sha256"])
authenticate(run / "committed_source.tar", identity["source_archive_sha256"])
current = scratch(run)
previous = scratch(ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero")
pidstat = Path(f"/proc/{identity['pid']}/stat")
process_exited = not pidstat.exists() or int(pidstat.read_text().rsplit(") ", 1)[1].split()[19]) != identity["start_ticks"]
gpu_pids = [int(x) for x in subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines() if x.strip()]
checks = dict(original_process_exited=process_exited,
    original_GPU_absent=identity["pid"] not in gpu_pids,
    completed_fresh_512=progress["status"] == "completed_unqualified" and progress["completed_updates"] == 512,
    no_training_checkpoint_reads=progress["checkpoint_reads"] == [],
    integrity_seal_all_checks=seal["all_checks_pass"] and all(seal["checks"].values()),
    committed_source_identity=revision["commit"] == identity["commit"] == progress["committed_source"]["commit"],
    frozen_and_current_264_sources_match=len(manifest) == 264 and all(
        sha(ROOT / name) == sha(run / "source_snapshot" / name) == digest for name, digest in manifest.items()),
    final_result_embedded_in_progress=read(run / "evaluation/result.json") == progress["evaluation"],
    independent_native_field_recomputation=current["all_case_fields_match"] and previous["all_case_fields_match"],
    grouped_existing_analysis_matches=all(all(math.isclose(g[k], h[k], rel_tol=2e-6, abs_tol=2e-8)
        for k in ["total_missing_ms", "worst_dual_support_fraction", "mean_leg_vibration_mrad", "maximum_leg_speed_rad_s"])
        for g, h in zip(current["groups"], behavior["groups"])))

# Verify the already recorded fixed-evaluation TensorBoard values, without appending to an event.
event, = list(run.glob("events.out.tfevents.*"))
INPUTS[str(event.relative_to(ROOT))] = sha(event)
events = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
result = read(run / "evaluation/result.json")
matched = 0
for height in (.23, .305, .42):
    cases = [c for c in result["cases"] if c["height_m"] == height]
    for metric in ("dual_support_fraction", "missing_physical_ms", "leg_speed_peak_rad_s",
                   "leg_vibration_3_95hz_rms_mrad", "height_mae_m", "forward_speed_rms_m_s"):
        rows = events.Scalars(f"EvalScratchStanding/{height:.3f}m/{metric}")
        expected = sum(c[metric] for c in cases) / len(cases)
        if len(rows) != 1 or rows[0].step != 512 or not math.isclose(rows[0].value, expected, rel_tol=2e-6, abs_tol=2e-8):
            raise ValueError(f"Original fixed-evaluation RSL mismatch: {height} {metric}")
        matched += 1
checks["original_RSL_evaluation_scalars_match"] = matched == 18

historical_path = ROOT / "reports/unified_robust_20261021/behavior121_summary.json"
historical = read(historical_path)
native = Path(historical["report"])
authenticate(native, historical["report_sha256"])
native_report = read(native)
trace = native.with_name("trajectory.npz")
authenticate(trace, historical["trajectory_sha256"])
analysis = native.with_name("analysis.json")
authenticate(analysis, historical["analysis_sha256"])
checks["historical_115_checkpoint_exact"] = native_report["checkpoint_sha256"] == historical["checkpoint_sha256"]
selected_names = ["spin_left", "spin_right", "spin_fast_left", "spin_fast_right",
                  "forward", "backward", "forward_fast", "backward_fast"]
historical_motion = {}
with np.load(trace, allow_pickle=False) as z:
    for name in selected_names:
        measurements = []
        for i, lane in enumerate(native_report["lanes"]):
            if lane["name"] != name:
                continue
            mask = z["alive"][:, i] & ~z["passive"][:, i]
            pos = z["local_position"][:, i, :2][mask]
            delta = pos - pos[0]
            radius = np.linalg.norm(delta, axis=1)
            displacement = delta[-1]
            measurements.append(dict(lane=i, survived=lane["survived"],
                observed_active_seconds=float(z["time"][:, i][mask][-1] - z["time"][:, i][mask][0]),
                max_xy_from_first_active_m=float(radius.max()),
                terminal_xy_from_first_active_m=float(radius[-1]),
                max_absolute_world_y_from_first_active_m=float(np.abs(delta[:, 1]).max()),
                terminal_displacement_xy_m=displacement.tolist(),
                requested_command_first=z["requested_command"][:, i][mask][0].tolist()))
        existing = historical["cases"][name]
        historical_motion[name] = dict(
            vx_mean_m_s=existing["vx_mean_m_s"], yaw_mean_rad_s=existing["yaw_mean_rad_s"],
            actual_abs_yaw_rpm=abs(existing["yaw_mean_rad_s"]) * 60 / (2 * math.pi),
            vx_mae_m_s=existing["vx_mae_m_s"], yaw_mae_rad_s=existing["yaw_mae_rad_s"],
            episodes=existing["episodes"], survived=existing["survived"],
            per_episode_position_measurements=measurements,
            worst_max_xy_from_first_active_m=max(m["max_xy_from_first_active_m"] for m in measurements),
            worst_max_absolute_world_y_m=max(m["max_absolute_world_y_from_first_active_m"] for m in measurements))

static115 = read(ROOT / "reports/unified_robust_20261021/iteration115_final.json")
baseline_protocol = read(ROOT / "reports/unified_robust_20261021/acceptance_protocol004.json")
mid, old_mid = current["groups"][1], previous["groups"][1]
comparison = dict(
    total_missing_ms_reduction_fraction=1 - mid["total_missing_ms"] / old_mid["total_missing_ms"],
    leg_vibration_ratio=mid["mean_leg_vibration_mrad"] / old_mid["mean_leg_vibration_mrad"],
    forward_speed_rms_reduction_fraction=1-mid["mean_forward_speed_rms_m_s"]/old_mid["mean_forward_speed_rms_m_s"],
    same_final_evaluation_initial_root_positions=read(run / "evaluation/result.json")["initial_root_positions"]
       == read(ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero/evaluation/result.json")["initial_root_positions"],
    full_native_initial_state_exact_pairing_available=False,
    scope="Each run's original warm terminal evaluation; common protocol, not an exact paired standalone intervention.")
if not all(checks.values()):
    raise ValueError(f"Failed authentication: {checks}")
out = dict(schema="wheel-leg-capability-status-evidence-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    current_fresh_candidate="scratch009", current_source_commit=identity["commit"],
    current_checkpoint_sha256=seal["final_checkpoint_sha256"],
    checks=checks, all_checks_pass=True, current=current, previous007=previous,
    midheight_009_vs007=comparison, matched_original_RSL_evaluation_scalars=matched,
    historical_115=dict(checkpoint_sha256=historical["checkpoint_sha256"],
        evidence_scope="Historical 12s, seed211322, DR0.35 native run; not scratch009 or hardware success.",
        actual_settled_recovery=historical["actual_settled_recovery"],
        ordinary_without_push=historical["ordinary_without_push"],
        motion_with_push=historical["motion_with_push"],
        motion=historical_motion,
        arcs={k:v for k,v in historical["cases"].items() if k.startswith("arc_")},
        command_projection=historical["command_projection"], stop_tail=historical["stop_tail"],
        prior_static_admission_status={k:static115[k] for k in ["candidate_adopted", "full_training_admitted", "status"]},
        prior_standing_displacement_limits_m=baseline_protocol["stand_xy_excursion_max_m_by_case"]),
    permanent_spin_requirement=dict(target_rpm=100, minimum_rpm=80,
        target_rad_s=100*2*math.pi/60, minimum_rad_s=80*2*math.pi/60,
        must_measure_actual_body_yaw=True, require_both_directions=True),
    current_capability_coverage=dict(midheight_standing="failed:1/7 strict 8s cases",
        low_and_high_standing="untrained extrapolation failed:0/14",
        domain_randomization="disabled in scratch009; robustness not established",
        initial_fallen_recovery="not trained or evaluated in scratch009",
        in_motion_fall_recovery="not trained or evaluated in scratch009",
        low_and_high_speed_translation="not trained or evaluated in scratch009",
        straight_path="not trained or evaluated in scratch009",
        stationary_80_100rpm_spin="not trained or evaluated; historical115 reached about38rpm only",
        forward_reverse_arcs="not trained or evaluated in scratch009",
        stationary_moving_spinning_height_transitions="not trained or evaluated in scratch009",
        high_speed_abrupt_braking="not trained or evaluated in scratch009",
        all_task_smoothness="not established; static position band alone is insufficient"),
    total_completed_fresh_PPO_updates=2433+512, new_PPO_by_reader=0, native_steps_by_reader=0,
    optimizer_constructed=False, curve_source="Original RSL TensorBoard events; no event mutation or server started",
    input_sha256=INPUTS, reader_sha256=sha(Path(__file__)), qualified=False, hardware_ready=False)
OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
print(json.dumps(dict(all_checks_pass=True, checks=checks, current=current["groups"],
    midheight_009_vs007=comparison,
    historical_spin_rpm={k:v["actual_abs_yaw_rpm"] for k,v in historical_motion.items() if "spin_fast" in k}),
    ensure_ascii=False))
