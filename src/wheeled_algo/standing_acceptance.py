"""Versioned, truth-position standing review; never part of policy inference."""
import math

import numpy as np

CASES = ("stand_low", "stand", "stand_high")
ACTUAL_HF_FIELDS = ("leg_motion_fast_rms_rad_3_95", "wheel_motion_fast_rms_rad_s_3_95")


def excursion_limit(protocol, case):
    limits = protocol.get("stand_xy_excursion_max_m_by_case")
    if limits is None:
        return protocol.get("stand_xy_excursion_max_m", float("inf"))
    if set(limits) != set(CASES) or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.
        for v in limits.values()
    ):
        raise ValueError("Standing limits require three finite positive case bounds")
    return limits[case]


def high_recentering(time, xy, spec):
    """Allow bounded offsets and inward motion without requiring zero drift."""
    time, xy = np.asarray(time, dtype=np.float64), np.asarray(xy, dtype=np.float64)
    if (time.ndim != 1 or xy.shape != (len(time), 2) or len(time) < 3
            or not np.isfinite(time).all() or not np.isfinite(xy).all()
            or not np.all(np.diff(time) > 0)):
        raise ValueError("Recentering review needs a finite, ordered complete XY trace")
    radius = np.linalg.norm(xy - xy[:1], axis=1)
    tail = time >= time[-1] - spec["tail_seconds"]
    if np.ptp(time[tail]) < spec["tail_seconds"] - .02:
        raise ValueError("Recentering review needs the complete declared tail window")
    centered_time = time[tail] - time[tail].mean()
    slope = float(centered_time @ radius[tail] / (centered_time @ centered_time))
    peak, tail_radius = float(radius.max()), float(radius[tail].mean())
    return_limit = max(spec["near_reference_radius_m"], peak * (1. - spec["minimum_peak_return_fraction"]))
    return dict(peak_excursion_m=peak, tail_mean_radius_m=tail_radius,
        tail_outward_radial_slope_m_s=slope, return_limit_m=return_limit,
        returned=tail_radius <= return_limit,
        no_sustained_outward_drift=slope <= spec["outward_radial_slope_max_m_s"])


def review_extended_standing(report, analysis, data, protocol):
    """Review all original active XY frames, actual vibration and wheel lift."""
    spec = protocol["long_standing_confirmation"]
    baseline = spec["actual_hf_baseline"]["by_case"]
    failures, rows = [], {}
    if analysis["passive_motor_peak_nm"] != 0. or analysis["max_gap_m"] > protocol["closure_gap_max_m"]:
        failures.append("closed-chain or passive torque")
    for case in CASES:
        lanes = [r["lane"] for r in report["lanes"] if r["name"] == case]
        if len(lanes) != spec["replicas_per_case"]:
            raise ValueError("Extended standing population differs from the frozen protocol")
        if analysis["cases"][case]["survived"] != len(lanes):
            failures.append(case + ": termination")
        limit, episodes = excursion_limit(protocol, case), []
        for lane in lanes:
            selected = data["alive"][:, lane] & ~data["passive"][:, lane]
            times = data["time"][selected, lane]
            xy = data["local_position"][selected, lane, :2]
            if len(times) < 3 or np.ptp(times) < spec["minimum_seconds"] - .02:
                failures.append(f"{case}: lane{lane} incomplete standing window")
                continue
            if not np.isfinite(xy).all():
                raise ValueError("Nonfinite truth XY in standing evidence")
            excursion = float(np.linalg.norm(xy - xy[:1], axis=1).max())
            air_mask = data["alive"][:, lane] & (data["time"][:, lane] >= .2)
            clearance = float(data["wheel_clearance_peak"][air_mask, lane].max())
            detail = dict(lane=lane, full_window_excursion_m=excursion, post_02s_clearance_peak_m=clearance)
            if excursion > limit:
                failures.append(f"{case}: lane{lane} excursion exceeds{limit * 100:g}cm")
            if not math.isfinite(clearance) or clearance > spec["wheel_clearance_after_02s_max_m"]:
                failures.append(f"{case}: lane{lane} wheel lift")
            if case == "stand_high":
                recent = high_recentering(times, xy, spec["high_recentering"])
                detail["recentering"] = recent
                if not recent["returned"] or not recent["no_sustained_outward_drift"]:
                    failures.append(f"{case}: lane{lane} sustained offset or outward drift")
            episodes.append(detail)
        quiet = analysis["cases"][case]["quiet"]
        for key in ACTUAL_HF_FIELDS:
            value = quiet.get(key, float("nan"))
            if not math.isfinite(value) or value > baseline[case][key]:
                failures.append(case + ": actual vibration regressed " + key)
        rows[case] = dict(episodes=episodes, excursion_limit_m=limit,
            survived=analysis["cases"][case]["survived"],
            actual_hf={key: quiet.get(key) for key in ACTUAL_HF_FIELDS})
    return dict(adequate=not failures, failures=failures, standing=rows)
