#!/usr/bin/env python3
"""Reassess archived standing traces under the user's post-recovery contact tolerance.

This never starts a simulator, policy, optimizer or TensorBoard server. The old
strict result and all original event/trace bytes remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def support_review(supported, *, standing_start_sample, ignored_max_ms=2):
    """Use a declared completion/start event, never the best supported suffix.

    Samples before the event are excluded. A later loss/fall remains in the
    reviewed suffix. The function deliberately does not infer a completion.
    """
    values = np.asarray(supported)
    if values.ndim != 1 or values.dtype != np.bool_ or not len(values):
        raise ValueError("Need an actual1kHz Boolean contact sequence")
    if type(ignored_max_ms) is not int or ignored_max_ms < 0:
        raise ValueError("Need a nonnegative integer millisecond tolerance")
    if standing_start_sample is None:
        return dict(evaluated=False, reason="no logged standing/recovery-completed event", support_pass=False)
    if type(standing_start_sample) is not int or not 0 <= standing_start_sample < len(values):
        raise ValueError("Standing event is outside the observed episode")
    active = values[standing_start_sample:]
    edges = np.diff(np.r_[False, ~active, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    intervals = [dict(first_missing_sample=int(a + standing_start_sample),
                      end_sample_exclusive=int(b + standing_start_sample), duration_ms=int(b-a),
                      ignored_for_acceptance=bool(b-a <= ignored_max_ms)) for a, b in zip(starts, ends)]
    retained = [r for r in intervals if not r["ignored_for_acceptance"]]
    retained_ms = sum(r["duration_ms"] for r in retained)
    return dict(evaluated=True, standing_start_sample=standing_start_sample,
        excluded_pre_standing_samples=standing_start_sample, evaluated_physical_ms=len(active),
        raw_missing_ms=int((~active).sum()), raw_support_fraction=float(active.mean()),
        raw_loss_events=len(intervals), ignored_loss_events=len(intervals)-len(retained),
        ignored_loss_ms=int((~active).sum())-retained_ms,
        retained_loss_events=len(retained), retained_loss_ms=retained_ms,
        effective_support_fraction=1.-retained_ms/len(active), support_pass=not retained,
        longest_raw_loss_ms=max((r["duration_ms"] for r in intervals), default=0), intervals=intervals)


def review(run, requirements):
    run = Path(run).resolve()
    result_path = run / "evaluation/result.json"
    result = json.loads(result_path.read_text())
    trace = run / "evaluation/native_trace.npz"
    if sha(trace) != result["trace_sha256"] or result["schema"] != "scratch-standing-native-evaluation-v1":
        raise ValueError("Need the authenticated original standing-only trace")
    contract = json.loads(requirements.read_text())
    contact = contract["standing_contact"]
    if contract["schema"] != "wheel-leg-permanent-user-requirements-v2" or contact["current_native_sampling_hz"] != 1000:
        raise ValueError("Unsupported acceptance requirements")
    rows = []
    with np.load(trace, allow_pickle=False) as z:
        frames, count = z["valid"].shape
        if z["valid"].dtype != np.bool_ or count != len(result["cases"]) or not z["valid"][0].all():
            raise ValueError("Invalid original native population")
        if np.any(np.diff(z["valid"].astype(np.int8), axis=0) > 0):
            raise ValueError("Cannot merge separate episodes into a standing window")
        for name, shape in dict(force=(frames,5,count,2,3), clearance=(frames,5,count,2),
                                dq=(frames,5,count,4), position=(frames,count,3)).items():
            if z[name].shape != shape or not np.isfinite(z[name]).all():
                raise ValueError(f"Bad native samples: {name}")
        for i, case in enumerate(result["cases"]):
            valid = z["valid"][:,i]
            force = z["force"][valid,:,i].reshape(-1,2,3)
            clearance = z["clearance"][valid,:,i].reshape(-1,2)
            absent_wheel = np.linalg.norm(force, axis=-1) <= 2.
            supported = ~absent_wheel.any(-1)
            # The archived evaluate() explicitly initializes standing; no fallen
            # episode or get-up process is present in these scratch tests.
            support = support_review(supported, standing_start_sample=0,
                ignored_max_ms=contact["ignored_continuous_loss_max_ms"])
            if support["raw_missing_ms"] != case["missing_physical_ms"] or len(supported) != case["physical_ms"]:
                raise ValueError("Raw contact evidence differs from the old receipt")
            position = z["position"][valid,i,:2] - np.asarray(result["initial_root_positions"])[i,:2]
            radius = np.linalg.norm(position, axis=1)
            gaps = clearance[absent_wheel]
            speed_peak = float(np.abs(z["dq"][valid,:,i]).max())
            if speed_peak != case["leg_speed_peak_rad_s"]:
                raise ValueError("Original actual physical speed peak changed")
            other = bool(case["full_horizon"] and case["tilt_peak_deg"] < 10.
                and case["height_mae_m"] < .01 and case["forward_speed_rms_m_s"] < .03
                and speed_peak < 3. and case["closure_gap_peak_m"] < .003)
            rows.append(dict(case_index=i, height_m=case["height_m"], pitch_deg=case["pitch_deg"],
                phase="explicit initialized standing; no fallen/get-up samples", support=support,
                original_strict_standing_pass=case["standing_pass"],
                noncontact_foundation_conditions_pass=other,
                standing_pass_v2=bool(other and support["support_pass"]),
                absent_wheel_sample_count=len(gaps),
                absent_wheel_zero_force_samples=int((np.linalg.norm(force,axis=-1)[absent_wheel] == 0.).sum()),
                absent_wheel_positive_clearance_samples=int((gaps > 0.).sum()),
                absent_wheel_clearance_over_0p1mm_samples=int((gaps > .0001).sum()),
                absent_wheel_estimated_clearance_max_m=float(gaps.max()) if len(gaps) else None,
                absent_wheel_estimated_clearance_median_m=float(np.median(gaps)) if len(gaps) else None,
                max_xy_excursion_m=float(radius.max()), terminal_xy_offset_m=float(radius[-1]),
                leg_vibration_rms_mrad=case["leg_vibration_3_95hz_rms_mrad"],
                leg_speed_peak_rad_s=speed_peak))
    groups = []
    for height in sorted({r["height_m"] for r in rows}):
        selected = [r for r in rows if r["height_m"] == height]
        vib = float(np.mean([r["leg_vibration_rms_mrad"] for r in selected]))
        groups.append(dict(height_m=height, case_count=len(selected),
            old_strict_standing_pass_count=sum(r["original_strict_standing_pass"] for r in selected),
            standing_pass_v2_count=sum(r["standing_pass_v2"] for r in selected),
            contact_pass_v2_count=sum(r["support"]["support_pass"] for r in selected),
            raw_loss_ms=sum(r["support"]["raw_missing_ms"] for r in selected),
            raw_loss_events=sum(r["support"]["raw_loss_events"] for r in selected),
            ignored_loss_ms=sum(r["support"]["ignored_loss_ms"] for r in selected),
            retained_loss_ms=sum(r["support"]["retained_loss_ms"] for r in selected),
            retained_loss_events=sum(r["support"]["retained_loss_events"] for r in selected),
            longest_raw_loss_ms=max(r["support"]["longest_raw_loss_ms"] for r in selected),
            absent_wheel_sample_count=sum(r["absent_wheel_sample_count"] for r in selected),
            absent_wheel_zero_force_samples=sum(r["absent_wheel_zero_force_samples"] for r in selected),
            absent_wheel_positive_clearance_samples=sum(r["absent_wheel_positive_clearance_samples"] for r in selected),
            absent_wheel_clearance_over_0p1mm_samples=sum(r["absent_wheel_clearance_over_0p1mm_samples"] for r in selected),
            absent_wheel_estimated_clearance_max_m=max((r["absent_wheel_estimated_clearance_max_m"] for r in selected
                if r["absent_wheel_estimated_clearance_max_m"] is not None), default=None),
            max_xy_excursion_m=max(r["max_xy_excursion_m"] for r in selected),
            max_terminal_xy_offset_m=max(r["terminal_xy_offset_m"] for r in selected),
            max_excursion_within_1cm_cases=sum(r["max_xy_excursion_m"] <= .01 for r in selected),
            terminal_offset_within_1cm_cases=sum(r["terminal_xy_offset_m"] <= .01 for r in selected),
            mean_leg_angle_band_rms_mrad=vib, mean_leg_angle_band_rms_rad=vib*.001,
            mean_leg_angle_band_rms_deg=math.degrees(vib*.001)))
    return dict(schema="standing-user-acceptance-review-v2", observed_utc=datetime.now(timezone.utc).isoformat(),
        run=str(run), completed_updates=result["completed_updates"], requirements=str(requirements.resolve()),
        native_protocol=result.get("protocol"), domain_intervention=result.get("domain_intervention"),
        requirements_sha256=sha(requirements), original_result_sha256=sha(result_path), native_trace_sha256=sha(trace),
        review_source_sha256=sha(__file__), cases=rows, groups=groups,
        standing_pass_v2_count=sum(r["standing_pass_v2"] for r in rows),
        criterion_change_is_not_policy_improvement=True, old_artifacts_modified=False,
        new_PPO=0, new_native_steps=0, optimizer_constructed=False,
        complete_policy_qualified=False, hardware_ready=False,
        interpretation="Thresholded force absence is not equivalent to visible lift-off. Geometry estimates and pulse durations are diagnostics; no unique cause is assigned. Corrected nominal8s standing is distinct from robustness or all-task qualification.")


def write_events(report, directory):
    from rsl_rl.utils.logger import Logger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = Path(report["run"])
    agent = run / "agent_config.json"
    if not agent.exists():
        receipt = json.loads((run / "receipt.json").read_text())
        if (receipt.get("schema") != "scratch-standing-readonly-evaluation-v1"
                or receipt.get("new_PPO") != 0 or receipt.get("optimizer_constructed") is not False
                or sha(receipt["checkpoint"]) != receipt["checkpoint_sha256"]):
            raise ValueError("Missing authenticated read-only evaluation agent source")
        agent = Path(receipt["checkpoint"]).parent / "agent_config.json"
    logger = Logger(log_dir=str(directory), cfg=json.loads(agent.read_text()),
        env_cfg=json.loads((run/"config/contract.json").read_text()), num_envs=1,
        is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
    logger.init_logging_writer()
    expected = {}
    try:
        for group in report["groups"]:
            for key,value in group.items():
                if key != "height_m" and value is not None:
                    tag = f"StandingAcceptanceV2/{group['height_m']:.3f}m/{key}"
                    expected[tag] = value
                    logger.writer.add_scalar(tag, value, report["completed_updates"])
        logger.writer.flush()
    finally:
        logger.writer.close()
    events = EventAccumulator(str(directory), size_guidance={"scalars":0}).Reload()
    if set(events.Tags()["scalars"]) != set(expected):
        raise ValueError("Revised RSL metric tags differ")
    for tag,value in expected.items():
        rows=events.Scalars(tag)
        if len(rows)!=1 or rows[0].step!=report["completed_updates"] or not math.isclose(rows[0].value,value,rel_tol=2e-6,abs_tol=2e-8):
            raise ValueError(f"RSL revised metric readback failed: {tag}")
    return dict(verified_scalar_count=len(expected), all_values_read_back=True,
        step_definition="Actual completed fresh PPO updates of the archived evaluated checkpoint",
        event_files_sha256={str(p.resolve()):sha(p) for p in directory.glob("events.out.tfevents.*")})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--requirements",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.output.with_suffix(".source.py")
    events=args.output.with_suffix(".events")
    if args.output.exists() or source.exists() or events.exists():
        raise ValueError("Preserve earlier reviews and exact readers")
    report=review(args.run,args.requirements)
    report["RSL_events"]=write_events(report,events)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(groups=report["groups"],standing_pass_v2_count=report["standing_pass_v2_count"],
                         RSL_verified_values=report["RSL_events"]["verified_scalar_count"]),ensure_ascii=False))
