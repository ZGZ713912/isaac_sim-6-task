#!/usr/bin/env python3
"""New 15.62 gearbox identification; never rescales the September recordings.

Run ``prepare`` with pandas, then ``fit`` with NumPy/SciPy/Matplotlib. The
original CSVs are immutable and authenticated before using local parse caches.
All mechanical quantities are measured at the wheel output, not the rotor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/wheel_real2sim_1562_20261028"
DATA = Path("/home/noir/Documents/workspace/wheel_leg扫频")
CACHE = Path("/home/noir/.cache/rmcs-sweep-archive-20261027")
SOURCES = {
    "low": ("wheel_leg_轮子低速辨识数据.csv", "d602e0bf471e",
            "9435cda103468a8dd3ef88188da0d418dc0e6bf09b72eb52d893bd1551735c12"),
    "high": ("wheel_leg_轮子高速辨识数据.csv", "85583c71b537",
             "a0dd1035621e67f93f827adefaa2ca18cd106a9e3b3a8678c7811cf3f058896a"),
}
SW = "/wheel_leg/sweep/"
WHEELS = ("left", "right")
SPEED_KNOTS = np.array([.75, 2., 6., 12., 20., 35.75, 48.75, 55.25])


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(args):
    import pandas as pd

    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"gear_ratio": 15.62, "bus_voltage_v": 24., "coordinate": "wheel_output",
             "hardware_confirmation": "User confirmed BOTH CSVs are the new 15.62, 24V gearbox on 2026-10-28",
             "coordinate_conversion_applied": False, "sources": {}}
    for key, (filename, cache_key, expected) in SOURCES.items():
        path = args.data / filename
        digest = sha256(path)
        if digest != expected:
            raise ValueError(f"Source changed; review provenance before fitting: {path}")
        meta = json.loads((CACHE / f"{cache_key}.json").read_text())
        if meta["sha256"] != digest:
            raise ValueError("Parse cache source hash mismatch")
        frame = pd.read_pickle(CACHE / f"{cache_key}.pkl")
        if len(frame) != meta["records"]:
            raise ValueError("Parse cache record count mismatch")
        columns = {field: SW + field for field in (
            "state", "run_id", "stage", "mode", "side", "validation", "reference_active",
            "control_steady_ns", "stage_elapsed_s", "stage_duration_s", "run_elapsed_s")}
        columns.update({"kp": SW + "host/wheel_velocity_kp", "torque_limit": SW + "host/wheel_torque_limit_nm"})
        arrays = {field: frame[name].to_numpy() for field, name in columns.items() if name in frame}
        for field in ("angle", "velocity", "torque", "max_torque", "control_torque", "tau_frame_api",
                      "feedback_steady_ns", "feedback_sequence", "tx_queued_steady_ns", "tx_kind", "temperature_c"):
            names = [f"/wheel_leg/{side}_wheel/{field}" for side in WHEELS]
            if all(name in frame for name in names):
                arrays[field] = frame[names].to_numpy()
        for field in ("reference", "reference_velocity", "model_velocity", "model_acceleration", "torque_preclip",
                      "saturated", "probe_torque_model"):
            names = [f"{SW}{side}_wheel/{field}" for side in WHEELS]
            if all(name in frame for name in names):
                arrays[field] = frame[names].to_numpy()
        for field in ("angle", "velocity"):
            names = [f"/wheel_leg/{side}_{joint}_joint/{field}" for side in WHEELS for joint in ("hip", "knee")]
            arrays["carrier_" + field] = frame[names].to_numpy()
        np.savez_compressed(args.output / f"{key}.npz", **arrays)
        source_audit = {**meta, "confirmed_new_gearbox": True,
            "wheel_max_torque_nm": [np.unique(arrays["max_torque"][:, k]).tolist() for k in range(2)],
            "declared_kp": np.unique(arrays.get("kp", [])).tolist(), "stages": []}
        for run, stage in frame[[SW + "run_id", SW + "stage"]].drop_duplicates().to_numpy():
            ix = np.flatnonzero((arrays["run_id"] == run) & (arrays["stage"] == stage) & (arrays["state"] == 3))
            if len(ix) < 2:
                continue
            t = (arrays["control_steady_ns"][ix] - arrays["control_steady_ns"][ix[0]]) * 1e-9
            delta = np.diff(t)
            row = {"run": int(run), "stage": int(stage), "side": int(np.median(arrays["side"][ix])),
                   "validation": bool(np.median(arrays["validation"][ix])), "n": len(ix),
                   "duration_s": float(t[-1]), "dt_median_s": float(np.median(delta)),
                   "dt_max_s": float(delta.max()), "speed_min_max": [], "reference_min_max": [],
                   "tau_median_nm": [], "feedback_tau_median_nm": [], "temperature_min_max": []}
            for k in range(2):
                tail = ix[len(ix)//2:]
                row["speed_min_max"].append([float(arrays["velocity"][ix, k].min()), float(arrays["velocity"][ix, k].max())])
                row["reference_min_max"].append([float(arrays["reference_velocity"][ix, k].min()), float(arrays["reference_velocity"][ix, k].max())])
                row["tau_median_nm"].append(float(np.median(arrays["tau_frame_api"][tail, k])))
                row["feedback_tau_median_nm"].append(float(np.median(arrays["torque"][tail, k])))
                if "temperature_c" in arrays:
                    row["temperature_min_max"].append([float(np.min(arrays["temperature_c"][ix, k])), float(np.max(arrays["temperature_c"][ix, k]))])
            source_audit["stages"].append(row)
        audit["sources"][key] = source_audit
        print(json.dumps({"source": key, "rows": len(frame), "stages": len(source_audit["stages"]),
                          "kp": source_audit["declared_kp"], "max_torque": source_audit["wheel_max_torque_nm"]}), flush=True)
        del frame, arrays
    write_json(args.output / "source_audit.json", audit)


def interpolation_basis(speed, knots=SPEED_KNOTS):
    """Partition of unity for a clamped, piecewise linear resistance curve."""
    x = np.clip(np.abs(speed), knots[0], knots[-1])
    left = np.minimum(np.searchsorted(knots, x, side="right") - 1, len(knots) - 2)
    alpha = (x - knots[left]) / (knots[left + 1] - knots[left])
    basis = np.zeros((len(x), len(knots)))
    basis[np.arange(len(x)), left] = 1. - alpha
    basis[np.arange(len(x)), left + 1] = alpha
    return basis


def frequency_response(data, side):
    """Probe acts as an instrument, avoiding naive closed-loop U/W regression.

    The estimated delay is the *aggregate* frame-command to reported-RPM delay;
    it cannot separate motor current loop, transport, and velocity estimation.
    Only training probe stages enter this estimate, never validation stages.
    """
    from scipy.signal import csd, welch, coherence
    from scipy.optimize import least_squares

    rows = []
    for stage in (11 + 30 * side, 26 + 30 * side):
        ii = np.flatnonzero((data["state"] == 3) & (data["run_id"] == 2) & (data["stage"] == stage))
        t = (data["control_steady_ns"][ii] - data["control_steady_ns"][ii[0]]) * 1e-9
        grid = np.arange(0, t[-1], .001)
        u, w, probe = [np.interp(grid, t, data[key][ii, side])
                       for key in ("tau_frame_api", "velocity", "probe_torque_model")]
        f, power = welch(probe, fs=1000, nperseg=4096)
        _, cross_u = csd(probe, u, fs=1000, nperseg=4096)
        _, cross_w = csd(probe, w, fs=1000, nperseg=4096)
        _, coh = coherence(probe, w, fs=1000, nperseg=4096)
        keep = (f >= 2) & (f <= 40) & (power > power.max() * .015) & (coh > .75)
        if keep.sum() < 5:
            raise ValueError("Too few independently excited/coherent wheel frequencies")
        jw, measured = 2j * np.pi * f[keep], cross_w[keep] / cross_u[keep]

        def error(x):
            inertia, damping, delay = x
            predicted = np.exp(-jw * delay) / (inertia * jw + damping)
            relative = predicted / measured - 1.
            return np.r_[relative.real, relative.imag]

        result = least_squares(error, [.0042, .001, .004],
            bounds=([.000575, 0., 0.], [.02, .03, .015]))
        rows.append({"stage": stage, "inertia_kg_m2": float(result.x[0]),
                     "local_damping_nm_per_rad_s": float(result.x[1]),
                     "aggregate_delay_s": float(result.x[2]),
                     "complex_relative_rmse": float(np.sqrt(np.mean(error(result.x) ** 2))),
                     "frequencies_hz": f[keep].tolist(), "coherence": coh[keep].tolist()})
    return {"training_probe_fits": rows,
            "aggregate_delay_s": float(np.median([r["aggregate_delay_s"] for r in rows])),
            "delay_components_separately_identified": False,
            "additional_first_order_time_constant_identified": False}


def integral_equations(sources, side, delay):
    """100ms non-overlapping windows; torque is shifted before integration."""
    x, y, groups = [], [], []
    for source, data in sources.items():
        t = (data["control_steady_ns"] - data["control_steady_ns"][0]) * 1e-9
        w = data["velocity"][:, side]
        u = np.interp(t - delay, t, data["tau_frame_api"][:, side])
        age = (data["control_steady_ns"] - data["feedback_steady_ns"][:, side]) * 1e-9
        eligible = ((data["state"] == 3) & (data["validation"] == 0) & (abs(w) >= .8)
            & (age >= 0) & (age < .002) & (np.max(abs(data["carrier_velocity"]), axis=1) < .15))
        if source == "high":
            # The 65rad/s voltage plateau is not a torque-calibration point.
            eligible &= ((data["run_id"] == 2) & (abs(data["reference_velocity"][:, side]) < 60.)
                         & ~np.isin(data["stage"], (14, 29, 44, 59)))
        basis = interpolation_basis(w)
        sign = np.sign(w)
        for start in range(0, len(t) - 100, 100):
            sl, end = slice(start, start + 101), start + 100
            dt = np.diff(t[sl])
            if (not eligible[sl].all() or w[start] * w[end] <= 0 or dt.min() <= 0 or dt.max() > .002
                    or data["stage"][start] != data["stage"][end]
                    or data["run_id"][start] != data["run_id"][end]):
                continue
            duration = t[end] - t[start]
            # Trapezoids reduce RPM sample-and-hold phase error. Input uses ZOH.
            wb = basis[sl] * sign[sl, None]
            x.append(np.r_[(w[end] - w[start]) / duration,
                          np.sum(.5 * (wb[:-1] + wb[1:]) * dt[:, None], axis=0) / duration, 1.])
            y.append(dt @ u[start:end] / duration)
            groups.append((source, int(data["stage"][start])))
    return np.asarray(x), np.asarray(y), groups


def robust_fit(x, y, groups):
    from scipy.optimize import lsq_linear
    # Smooth the speed curve, without forcing identical left/right resistance.
    regularizer = np.zeros((len(SPEED_KNOTS) - 2, x.shape[1]))
    for k in range(len(regularizer)):
        regularizer[k, k + 1:k + 4] = (1., -2., 1.)
    scale = np.maximum(np.linalg.norm(x, axis=0), 1e-9)
    # Free decay carries the passive-loss information. Its shorter duration
    # must not be swamped by hundreds of seconds of powered steady holds.
    coast = np.array([source == "high" and stage in (3, 18, 33, 48) for source, stage in groups])
    coast_weight = min(15., float((~coast).sum() / max(1, coast.sum())))
    base_weights = np.where(coast, coast_weight, 1.)
    weights = base_weights.copy()
    lower = np.r_[.000575, np.full(len(SPEED_KNOTS), .005), -.08]
    upper = np.r_[.02, np.full(len(SPEED_KNOTS), 1.), .08]
    for _ in range(10):
        xx = np.vstack([x * np.sqrt(weights[:, None]), 2 * regularizer])
        yy = np.r_[y * np.sqrt(weights), np.zeros(len(regularizer))]
        result = lsq_linear(xx / scale, yy, bounds=(lower * scale, upper * scale), tol=1e-12)
        parameters = result.x / scale
        residual = x @ parameters - y
        weights = base_weights * np.minimum(1., .025 / np.maximum(abs(residual), 1e-12))
    return parameters, {"windows": len(y), "mean_torque_rmse_nm": float(np.sqrt(np.mean(residual ** 2))),
                        "mean_torque_mae_nm": float(np.mean(abs(residual))),
                        "normalized_condition_number": float(np.linalg.cond(x / scale)),
                        "training_coast_windows": int(coast.sum()), "coast_objective_weight": coast_weight}


def friction(w, axis):
    f = float(np.interp(abs(w), axis["friction_speed_knots_rad_s"], axis["moving_friction_nm"]))
    return max(.001, f + (axis["direction_bias_nm"] if w >= 0 else -axis["direction_bias_nm"]))


def passive_decay_fit(x, y, groups, powered_parameters):
    """Isolate passive resistance from zero-current decays with fixed inertia.

    Long closed-loop holds contain controller/current-conversion residuals.
    They must not define a mechanical loss merely because they have more rows.
    Two training decays of opposite sign per wheel provide the passive fit;
    the later two decays and the entire third run remain excluded.
    """
    from scipy.optimize import lsq_linear
    mask = np.array([source == "high" and stage in (3, 18, 33, 48) for source, stage in groups])
    xx = x[mask, 1:]
    yy = y[mask] - x[mask, 0] * powered_parameters[0]
    reg = np.zeros((len(SPEED_KNOTS)-2, xx.shape[1]))
    for k in range(len(reg)):
        reg[k, k:k+3] = (1., -2., 1.)
    prior = np.eye(xx.shape[1]) * .12
    a = np.vstack([xx, .25 * reg, prior])
    b = np.r_[yy, np.zeros(len(reg)), prior @ powered_parameters[1:]]
    result = lsq_linear(a, b, bounds=(np.r_[np.full(len(SPEED_KNOTS), .005), -.08],
                                    np.r_[np.ones(len(SPEED_KNOTS)), .08]), tol=1e-12)
    passive = np.r_[powered_parameters[0], result.x]
    # The final fraction of a quantized RPM decay cannot resolve the moving
    # loss below 2rad/s. Preserve the separately excited low-speed fit there.
    passive[1:3] = powered_parameters[1:3]
    return passive, {
        "training_coast_windows": int(mask.sum()),
        "passive_force_rmse_nm": float(np.sqrt(np.mean((xx @ result.x - yy)**2))),
        "method": "zero-current training decays; fixed command-calibrated effective inertia; weak curvature/low-speed prior"}


def replay_trace(t, reference, probe, command, initial, axis, *, closed_loop, coast):
    """No measured state is injected after the initial condition of a stage."""
    n = len(t)
    prediction, generated = np.empty(n), np.zeros(n)
    prediction[0] = initial
    delay, inertia = axis["aggregate_delay_s"], axis["effective_total_inertia_kg_m2"]
    pending = 0
    for i in range(n - 1):
        w, dt = prediction[i], t[i + 1] - t[i]
        request = (0. if coast else .5 * (reference[i] - w) + probe[i]) if closed_loop else command[i]
        generated[i] = np.clip(request, -4.5, 4.5)
        while pending < i and t[pending + 1] <= t[i] - delay:
            pending += 1
        torque = generated[pending] if t[i] >= delay else command[0]
        env = axis["high_speed_envelope"]
        if w * torque > 0:
            sign_index = 0 if w >= 0 else 1
            allowed = min(4.5, max(0., env["rolloff_slope_nm_per_rad_s"][sign_index]
                * (env["zero_torque_speed_rad_s"][sign_index] - abs(w))))
            torque = np.clip(torque, -allowed, allowed)
        fs = axis["static_breakaway_nominal_nm"]
        if abs(w) < 1e-10 and abs(torque) <= fs:
            prediction[i + 1] = 0.
            continue
        free = w + dt * torque / inertia
        f = friction(w if abs(w) > 1e-10 else torque, axis)
        step = dt * f / inertia
        prediction[i + 1] = max(0., free - step) if free >= 0 else min(0., free + step)
    return prediction


def validation_traces(sources, side):
    for source, data in sources.items():
        for run in np.unique(data["run_id"]):
            for stage in np.unique(data["stage"][data["run_id"] == run]):
                ii = np.flatnonzero((data["state"] == 3) & (data["run_id"] == run) & (data["stage"] == stage))
                if len(ii) < 100:
                    continue
                val = bool(np.median(data["validation"][ii]))
                coast = bool(source == "high" and (run == 2 and stage in (14 + side * 30, 29 + side * 30)
                    or run == 3 and side == 0 and stage == 3))
                extra_run = source == "high" and run == 3 and side == 0
                ref = data["reference_velocity"][ii, side]
                if not (val or coast or extra_run) or (max(abs(ref)) < .1 and not coast):
                    continue
                # Reject a partial/interrupt stage, never stitch across missing time.
                t = (data["control_steady_ns"][ii] - data["control_steady_ns"][ii[0]]) * 1e-9
                if np.diff(t).min() <= 0:
                    raise ValueError("Nonmonotone validation timestamps")
                complete = (bool(t[-1] > .95 * np.median(data["stage_duration_s"][ii]))
                    if "stage_duration_s" in data else
                    bool(ii[-1] + 1 < len(data["stage"]) and data["stage"][ii[-1] + 1] != stage
                         and data["run_id"][ii[-1] + 1] == run))
                yield {"source": source, "run": int(run), "stage": int(stage), "indices": ii,
                       "coast": coast, "time_s": t, "complete": complete,
                       "max_gap_s": float(np.diff(t).max())}


def fit(args):
    sources = {key: dict(np.load(args.output / f"{key}.npz")) for key in SOURCES}
    report = {"schema": "identified-wheel-1562-fit-v1", "coordinate": "wheel_output",
              "data_audit_sha256": sha256(args.output / "source_audit.json"),
              "training_split": "Low run1 non-validation; high run2 non-validation excluding |target|>=60 and coast14/29/44/59. High run3 entirely held out.",
              "axes": {}, "validation": []}
    model = {"schema": "identified-wheel-response-v1", "gear_ratio": 15.62, "bus_voltage_v": 24.,
             "coordinate": "wheel_output", "controller_velocity_p_nm_per_rad_s": .5,
             "software_torque_cap_nm": 4.5, "current_conversion_peak_nm": 4.880434419381787,
             "axes": {}, "source_files": [
                 {"path": str(args.data / filename), "sha256": digest, "new_gearbox_confirmed": True}
                 for filename, _, digest in SOURCES.values()]}
    high = sources["high"]
    all_traces = {}
    for side, name in enumerate(WHEELS):
        joint = ("L_joint3", "R_joint3")[side]
        response = frequency_response(high, side)
        delay = round(response["aggregate_delay_s"] / .0005) * .0005
        x, y, groups = integral_equations(sources, side, delay)
        powered_parameters, score = robust_fit(x, y, groups)
        parameters, passive_score = passive_decay_fit(x, y, groups, powered_parameters)
        axis = {"side": name, "effective_total_inertia_kg_m2": float(parameters[0]),
                "cad_inertia_kg_m2": .000575,
                "additional_armature_kg_m2": float(parameters[0] - .000575),
                "friction_speed_knots_rad_s": SPEED_KNOTS.tolist(), "moving_friction_nm": parameters[1:-1].tolist(),
                "direction_bias_nm": float(parameters[-1]), "aggregate_delay_s": delay,
                "powered_resistance_diagnostic_nm": powered_parameters[1:-1].tolist(),
                "moving_friction_source": "powered low-speed excitation up to 2rad/s; zero-current training coasts above 6rad/s; interpolate between",
                # These are bounded start/stop observations, not uniquely identified presliding parameters.
                "static_breakaway_nominal_nm": (.19, .375)[side],
                "static_breakaway_observed_interval_nm": ([.15, .23], [.30, .45])[side],
                "static_breakaway_exactly_identified": False}
        plateau, zeros, slopes, spread = [], [], [], []
        for sign, stage in ((1., 2 + side * 30), (-1., 17 + side * 30)):
            mask = ((high["state"] == 3) & (high["run_id"] == 2) & (high["stage"] == stage)
                    & (high["stage_elapsed_s"] > 3.))
            speed = sign * high["velocity"][mask, side]
            p = float(np.median(speed)); plateau.append(p)
            spread.append(np.quantile(speed, [.05, .95]).tolist())
            # Only no-load plateau is measured. Rolloff WIDTH remains a prior.
            slope = 4.5 / (.10 * p)
            slopes.append(slope)
            zeros.append(p + friction(sign * p, axis) / slope)
        axis["high_speed_envelope"] = {"measured_airborne_plateau_rad_s": plateau,
            "measured_plateau_p05_p95_rad_s": spread, "zero_torque_speed_rad_s": zeros,
            "rolloff_slope_nm_per_rad_s": slopes, "rolloff_width_fraction_prior": .10,
            "rolloff_width_randomization_recommendation": [.05, .20],
            "loaded_torque_speed_curve_identified": False,
            "braking": "retain independent 4.5Nm software cap; do not collapse braking with motoring speed limit"}
        model["axes"][joint] = axis
        report["axes"][joint] = {"fit": axis, "integral_score": score, "passive_decay_score": passive_score, "frequency_response": response,
                                  "training_stage_count": len(set(groups))}
        for trace in validation_traces(sources, side):
            data, ii, t = sources[trace["source"]], trace["indices"], trace["time_s"]
            reference, measured, command = [data[key][ii, side] for key in ("reference_velocity", "velocity", "tau_frame_api")]
            probe = data.get("probe_torque_model", np.zeros_like(data["velocity"]))[ii, side]
            predicted = replay_trace(t, reference, probe, command, measured[0], axis,
                                     closed_loop=True, coast=trace["coast"])
            torque_replay = replay_trace(t, reference, probe, command, measured[0], axis,
                                        closed_loop=False, coast=trace["coast"])
            error = predicted - measured
            label = f"{name}_{trace['source']}_r{trace['run']}_s{trace['stage']}"
            row = {k: trace[k] for k in ("source", "run", "stage", "coast", "complete", "max_gap_s")}
            row.update(side=name, label=label, duration_s=float(t[-1]),
                       speed_rmse_rad_s=float(np.sqrt(np.mean(error ** 2))),
                       speed_mae_rad_s=float(np.mean(abs(error))),
                       speed_abs_error_p95_rad_s=float(np.quantile(abs(error), .95)),
                       torque_input_speed_rmse_rad_s=float(np.sqrt(np.mean((torque_replay - measured) ** 2))),
                       measured_abs_speed_p95_rad_s=float(np.quantile(abs(measured), .95)))
            report["validation"].append(row)
            all_traces[label + "_time_s"] = t
            all_traces[label + "_measured"] = measured
            all_traces[label + "_predicted"] = predicted
            all_traces[label + "_command"] = command
            all_traces[label + "_torque_input_predicted"] = torque_replay
        print(json.dumps({"side": name, "J": parameters[0], "delay_ms": delay * 1000,
                          "friction": parameters[1:-1].tolist(), "integral": score}), flush=True)
    model["status"] = "airborne_identified_candidate"
    model["scope"] = "new gearbox wheel-output effective dynamics; independent left/right; research real2sim"
    model["uncertainty"] = {"inertia_scale_training_prior": [.94, 1.06],
        "friction_variation": "Use retained zero-current decays and temperature variation; do not confuse hold tracking error with a full plant validation.",
        "delay": "3.0-6.0ms between training directions; aggregate effective I/O, not isolated shaft-torque latency.",
        "loaded_capacity": "No dynamometer/ground-load data; high-speed rolloff shape remains a randomized prior."}
    model["limitations"] = ["Airborne data identifies effective output inertia/losses, not tire-ground contact or traction.",
        "Torque is nominal C620 current conversion, not a shaft torque sensor; inertia is relative to that scale.",
        "Aggregate command-to-reported-RPM delay must replace overlapping transport/feedback delays, not be added twice.",
        "No-load maximum speeds are measured. High-speed rolloff width and loaded torque-speed capacity are explicit priors.",
        "Static breakaway is bounded by these records; position-dependent cogging/presliding is not uniquely identified."]
    write_json(args.output / "fit.json", report)
    model_path = ROOT / "model/wheel_identified_1562_20261028/model.json"
    write_json(model_path, model)
    np.savez_compressed(args.output / "heldout_replay.npz", **all_traces)
    print(json.dumps({"model": str(model_path), "validation_traces": len(report["validation"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
