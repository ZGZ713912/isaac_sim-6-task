"""Seal matched extended-static and full-suite evidence after native completion."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/unified_robust_20261022"
REPORTS = ROOT / "reports/unified_robust_20261021"
CASES = ("stand_low", "stand", "stand_high")


def read(name):
    directory = RUNS / name / "robust"
    report = json.loads((directory / "report.json").read_text())
    analysis = json.loads((directory / "analysis.json").read_text())
    trajectory = directory / "trajectory.npz"
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    if digest != report["trajectory_sha256"]:
        raise ValueError("Trajectory hash mismatch: " + name)
    return directory, report, analysis, np.load(trajectory)


def paired(before, after):
    checks = {}
    for key in ("name", "pose_index", "command", "recovery", "world_origin_m",
                "first_reset_domain", "material_coefficients", "sequence", "push", "push_sign"):
        checks[key] = [x.get(key) for x in before[1]["lanes"]] == [x.get(key) for x in after[1]["lanes"]]
    checks["first_observation"] = np.array_equal(before[3]["observations"][0], after[3]["observations"][0])
    checks["native_origin_mode"] = before[1]["native_origin_mode"] == after[1]["native_origin_mode"]
    if not all(checks.values()):
        raise ValueError("Unmatched native pair: " + str(checks))
    return checks


def standing(evidence):
    directory, report, analysis, data = evidence
    result = {}
    for case in CASES:
        row = analysis["cases"][case]
        quiet = row["quiet"]
        lanes = [r["lane"] for r in report["lanes"] if r["name"] == case]
        clearances = [float(data["wheel_clearance_peak"][data["alive"][:, i] & (data["time"][:, i] >= .2), i].max())
                      for i in lanes]
        excursions = []
        for i in lanes:
            selected = data["alive"][:, i] & ~data["passive"][:, i]
            xy = data["local_position"][selected, i, :2]
            excursions.append(float(np.linalg.norm(xy - xy[:1], axis=-1).max()))
        result[case] = dict(episodes=row["episodes"], survived=row["survived"],
            full_window_max_excursion_m=max(excursions),
            full_window_within_5cm=sum(x <= .05 for x in excursions),
            steady_window_max_excursion_m=quiet["body_xy_worst_episode_excursion_m"],
            leg_motion_fast_rms_rad_3_95=quiet["leg_motion_fast_rms_rad_3_95"],
            wheel_motion_fast_rms_rad_s_3_95=quiet["wheel_motion_fast_rms_rad_s_3_95"],
            leg_target_fast_rms_rad_3_95=quiet["leg_target_fast_rms_rad_3_95"],
            wheel_target_fast_rms_rad_s_3_95=quiet["wheel_target_fast_rms_rad_s_3_95"],
            wheel_clearance_after_02s_peak_m=max(clearances))
    return result


def main():
    pipeline = json.loads((REPORTS / "iteration036_pipeline.json").read_text())
    if pipeline["status"] != "completed":
        raise ValueError("Wait for all serial native phases to complete")
    baseline = read("quiet_velocity_baseline036")
    candidate = read("quiet_velocity_confirmation036")
    original_full = read("side_confirmation032")
    full = read("velocity_full_confirmation036")
    extended_pair, full_pair = paired(baseline, candidate), paired(original_full, full)
    before_static, after_static = standing(baseline), standing(candidate)
    failures = []
    for case, row in after_static.items():
        if row["survived"] != row["episodes"]:
            failures.append(case + " static termination")
        if row["full_window_max_excursion_m"] > .05:
            failures.append(case + " full-window excursion exceeds5cm")
        if row["wheel_clearance_after_02s_peak_m"] > .003:
            failures.append(case + " wheel clearance exceeds3mm after0.2s")
    train = json.loads((REPORTS / "iteration036_training.json").read_text())
    result = dict(schema="quiet036-complete-comparison-v1", training=train,
        extended_pairing=extended_pair, full_pairing=full_pair,
        extended_static_before=before_static, extended_static_after=after_static,
        full_static=standing(full),
        recovered_by_actual_attitude=full[2]["actual_settled_attitudes"],
        motion=dict(survived=full[2]["motion_survived"], episodes=full[2]["motion_episodes"],
                    practical_moving_fall_recoveries=full[2]["practical_moving_fall_recoveries"]),
        motion_cases={name: {k: row[k] for k in ("vx_mean_m_s", "yaw_mean_rad_s", "vx_mae_m_s", "yaw_mae_rad_s")}
            for name, row in full[2]["cases"].items() if name.startswith(("arc_", "forward", "reverse", "fast_"))},
        passive_motor_peak_nm=full[2]["passive_motor_peak_nm"], closure_peak_m=full[2]["max_gap_m"],
        static_failures=failures, static_extended_pass=not failures,
        long_training_admitted=False, deployment_package_replaced=False,
        scope="One matched randomized seed; full fresh nominal two-seed admission remains separate",
        reports={name: str(e[0].relative_to(ROOT)) for name, e in
                 (("baseline", baseline), ("extended_candidate", candidate), ("full_candidate", full))})
    path = REPORTS / "iteration036_final.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("extended_static_before", "extended_static_after", "motion", "static_failures")}, indent=2))


if __name__ == "__main__":
    main()
