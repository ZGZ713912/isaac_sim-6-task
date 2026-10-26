#!/usr/bin/env python3
"""Read authentic 1ms leg velocity/effort traces without running a policy.

The existing3--95Hz position metric uses200Hz observations. These additional
1000Hz velocity/effort diagnostics expose short physical impulses separately.
They never change contact acceptance or supply a new standing pass predicate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rsl_rl.utils.logger import Logger


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band_rms(samples, lower, upper, *, upper_inclusive):
    """Real FFT normalization checked independently with Parseval's identity."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 1000:
        raise ValueError("Need at least one second of four-leg1kHz samples")
    centered = values - values.mean(0)
    frequency = np.fft.rfftfreq(len(values), .001)
    spectrum = np.fft.rfft(centered, axis=0)
    selected = (frequency >= lower) & ((frequency <= upper) if upper_inclusive else (frequency < upper))
    spectrum[~selected] = 0.
    filtered = np.fft.irfft(spectrum, n=len(values), axis=0)
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    weights = np.full(len(spectrum), 2., dtype=np.float64)
    weights[0] = 1.
    if len(values) % 2 == 0:
        weights[-1] = 1.
    spectral_power = float(np.sum(weights[:, None] * np.abs(spectrum) ** 2) / (len(values) ** 2 * 4))
    if not np.isclose(rms * rms, spectral_power, rtol=1e-11, atol=1e-15):
        raise ValueError("Physical-spectrum normalization mismatch")
    return rms


def inspect(evaluation, height):
    evaluation = evaluation.resolve()
    result_path, trace_path = evaluation / "result.json", evaluation / "native_trace.npz"
    result = json.loads(result_path.read_text())
    if sha(trace_path) != result["trace_sha256"]:
        raise ValueError("Original native trace changed")
    rows = []
    with np.load(trace_path) as trace:
        valid = trace["valid"]
        if (valid.ndim != 2 or valid.dtype != np.bool_ or not valid[0].all()
                or np.any(np.diff(valid.astype(np.int8), axis=0) > 0)
                or valid.shape[1] != len(result["cases"])):
            raise ValueError("Need one original valid prefix per case")
        frames, lanes = valid.shape
        for name, expected in dict(dq=(frames, 5, lanes, 4), torque=(frames, 5, lanes, 4),
                                   force=(frames, 5, lanes, 2, 3), q=(frames, lanes, 4)).items():
            if trace[name].shape != expected or not np.isfinite(trace[name]).all():
                raise ValueError("Missing or nonfinite physical trace: " + name)
        for lane, case in enumerate(result["cases"]):
            if case["height_m"] != height:
                continue
            active = valid[:, lane]
            dq = trace["dq"][active, :, lane].reshape(-1, 4)
            torque = trace["torque"][active, :, lane].reshape(-1, 4)
            force = trace["force"][active, :, lane].reshape(-1, 2, 3)
            q = trace["q"][active, lane]
            missing = ~(np.linalg.norm(force, axis=-1) > 2.).all(-1)
            if (len(dq) != case["physical_ms"] or int(missing.sum()) != case["missing_physical_ms"]
                    or float(np.abs(dq).max()) != case["leg_speed_peak_rad_s"]
                    or not np.isclose(float(np.sqrt(np.mean(dq ** 2))), case["leg_speed_rms_rad_s"], rtol=1e-6)):
                raise ValueError("Original physical metrics disagree")
            # Reproduce the pre-existing position metric without extending its
            # bandwidth or substituting integrated velocities for positions.
            late_q = q[600:] if len(q) > 600 else q
            frequency = np.fft.rfftfreq(len(late_q), .005)
            spectrum = np.fft.rfft(late_q - late_q.mean(0), axis=0)
            spectrum[(frequency < 3.) | (frequency > 95.)] = 0.
            old_vibration = float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(late_q), axis=0) ** 2)) * 1000.)
            if not np.isclose(old_vibration, case["leg_vibration_3_95hz_rms_mrad"], rtol=1e-6, atol=1e-9):
                raise ValueError("Original200Hz position metric disagrees")
            row = dict(height_m=height, pitch_deg=case["pitch_deg"], physical_ms=len(dq),
                original_missing_physical_ms=int(missing.sum()), original_standing_pass=case["standing_pass"],
                original_position_vibration_3_95Hz_rms_mrad=old_vibration,
                physical_leg_speed_peak_rad_s=float(np.abs(dq).max()),
                late_start_s=3., late_end_s=len(dq) * .001, late_samples=max(0, len(dq) - 3000),
                enough_late_exposure=len(dq) >= 4000)
            if row["enough_late_exposure"]:
                velocity = dq[3000:].astype(np.float64)
                effort = torque[3000:].astype(np.float64)
                row.update(
                    late_velocity_3_95Hz_rms_rad_s=band_rms(velocity, 3., 95., upper_inclusive=False),
                    late_velocity_95_450Hz_rms_rad_s=band_rms(velocity, 95., 450., upper_inclusive=True),
                    late_effort_95_450Hz_rms_Nm=band_rms(effort, 95., 450., upper_inclusive=True),
                    late_leg_speed_abs_p999_rad_s=float(np.quantile(np.abs(velocity), .999)),
                    late_any_leg_speed_over_0p2_rad_s_fraction=float((np.abs(velocity) > .2).any(-1).mean()),
                    late_leg_acceleration_rms_rad_s2=float(np.sqrt(np.mean((np.diff(velocity, axis=0) / .001) ** 2))),
                    late_leg_torque_delta_rms_Nm_per_ms=float(np.sqrt(np.mean(np.diff(effort, axis=0) ** 2))))
            rows.append(row)
    if len(rows) != 7:
        raise ValueError("Need all seven original pitch cases for the selected height")
    sufficient = [row for row in rows if row["enough_late_exposure"]]
    metrics = ("late_velocity_3_95Hz_rms_rad_s", "late_velocity_95_450Hz_rms_rad_s",
        "late_effort_95_450Hz_rms_Nm", "late_leg_speed_abs_p999_rad_s",
        "late_any_leg_speed_over_0p2_rad_s_fraction", "late_leg_acceleration_rms_rad_s2",
        "late_leg_torque_delta_rms_Nm_per_ms")
    return dict(evaluation=str(evaluation), completed_updates=result["completed_updates"],
        result_sha256=sha(result_path), trace_sha256=sha(trace_path), height_m=height,
        cases=rows, sufficient_late_cases=len(sufficient),
        means_of_sufficient_cases={key: sum(row[key] for row in sufficient) / len(sufficient)
            for key in metrics} if sufficient else {},
        original_standing_pass=sum(row["original_standing_pass"] for row in rows),
        original_missing_physical_ms=sum(row["original_missing_physical_ms"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--height", type=float, choices=(.23, .305, .42), default=.305)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, events = args.output, args.output.with_suffix(".events")
    archived_source = output.with_suffix(".source.py")
    if output.exists() or events.exists() or archived_source.exists():
        raise ValueError("Preserve existing evidence, RSL events and exact reader")
    names = [path.resolve().parent.name for path in args.evaluations]
    if len(names) != len(set(names)):
        raise ValueError("Use distinct evaluation run names")
    rows, event_info = [], {}
    for name, evaluation in zip(names, args.evaluations):
        row = inspect(evaluation, args.height)
        rows.append(row)
        run = evaluation.resolve().parent
        logger = Logger(log_dir=str(events / name), cfg=json.loads((run / "agent_config.json").read_text()),
            env_cfg=json.loads((run / "config/contract.json").read_text()), num_envs=1,
            is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
        logger.init_logging_writer()
        try:
            for key, value in row["means_of_sufficient_cases"].items():
                logger.writer.add_scalar(f"DiagnosticScratchPhysicalExcitation/{args.height:.3f}m/{key}",
                                         value, row["completed_updates"])
            logger.writer.add_scalar(f"DiagnosticScratchPhysicalExcitation/{args.height:.3f}m/sufficient_late_cases",
                                     row["sufficient_late_cases"], row["completed_updates"])
            logger.writer.flush()
        finally:
            logger.writer.close()
        event_info[name] = dict(directory=str((events / name).resolve()),
            files_sha256={path.name: sha(path) for path in (events / name).glob("events.out.tfevents.*")})
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    report = dict(schema="scratch-physical-excitation-evidence-v1", evaluations=rows, tensorboard=event_info,
        physical_sample_dt_s=.001, late_window_start_s=3.,
        band_definition="Velocity3<=f<95Hz; velocity/effort95<=f<=450Hz. Per-leg realFFT, mean-square over four legs. "
            "No taper; every band has independent Parseval verification. Position3--95Hz remains the original200Hz metric.",
        all_original_metrics_and_parseval_checks_pass=True, new_PPO=0, native_steps=0,
        optimizer_constructed=False, behavior_qualification=False, hardware_ready=False,
        reader_sha256=hashlib.sha256(source).hexdigest(),
        scope="Additional diagnostics from original valid1ms samples, never a new pass criterion. "
            "Insufficient late exposure is not zero vibration.95--450Hz velocity RMS is not angular displacement.")
    archived_source.write_bytes(source)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([dict(run=Path(row["evaluation"]).parent.name,
        sufficient_late_cases=row["sufficient_late_cases"], means=row["means_of_sufficient_cases"])
        for row in rows]))


if __name__ == "__main__":
    main()
