#!/usr/bin/env python3
"""Compare complete native gap runs; retain survival and control-quality failures."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from train_robust_getup import write_json, digest


def read(path):
    import numpy as np
    path = Path(path)
    progress = json.loads((path/"progress.json").read_text())
    if progress["status"] != "completed":
        raise ValueError("Require a complete native evaluation: " + str(path))
    report = json.loads((path/"evaluation/report.json").read_text())
    summary = json.loads((path/"evaluation/gap_summary.json").read_text())
    trace = np.load(path/"evaluation/trajectory.npz")
    assert digest(path/"evaluation/trajectory.npz") == summary["trajectory_sha256"]
    return progress, report, summary, trace


def aggregate(rows):
    import numpy as np
    result = dict(episodes=len(rows), survived=sum(r["survived"] for r in rows),
        ever_recovered=sum(r["recovered"] for r in rows),
        tail_stable_at_least90pct=sum(r.get("tail_stable_fraction", 0.)>=.9 for r in rows))
    for key in ("first_200ms_leg_effort_peak_nm", "max_leg_speed_rad_s", "saturation_fraction",
                "longest_saturation_s", "tail_leg_motion_3_95hz_rms_rad", "tail_leg_speed_rms_rad_s",
                "tail_leg_effort_rms_nm", "tail_saturation_fraction", "max_position_excursion_m",
                "vx_mae_m_s", "yaw_mae_rad_s"):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = dict(mean=float(np.mean(values)), median=float(np.median(values)), max=float(max(values)), count=len(values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    before, rb, sb, tb = read(args.before)
    after, ra, sa, ta = read(args.after)
    if before.get("evaluation_overrides", {}) != after.get("evaluation_overrides", {}):
        raise ValueError("Counterfactual actuator interventions cannot be compared as a policy-only change")
    if before.get("actuator_timing_profile","stress") != after.get("actuator_timing_profile","stress"):
        raise ValueError("Actuator timing profiles differ")
    if (before["seed"], before["seconds"], before["severity"], before.get("actuator_model"), before.get("nominal_dynamics",False)) != (
            after["seed"], after["seconds"], after["severity"], after.get("actuator_model"), after.get("nominal_dynamics",False)):
        raise ValueError("Require identical seed, duration, severity and actuator interpretation")
    if [r["name"] for r in sb["rows"]] != [r["name"] for r in sa["rows"]]:
        raise ValueError("Case populations differ")
    if not all(x["first_reset_domain"] == y["first_reset_domain"] and x["material_coefficients"] == y["material_coefficients"]
               for x,y in zip(rb["lanes"], ra["lanes"])):
        raise ValueError("Initial sensor/actuator/material domains differ")
    # motor_position/velocity in the trajectory are POST action. They cannot be
    # required to match after different policies have controlled active lanes.
    passive = tb["passive"][0] & ta["passive"][0]
    initial = dict(actor_inputs_exact=bool(np.array_equal(tb["observations"][0], ta["observations"][0])),
        passive_mask_exact=bool(np.array_equal(tb["passive"][0], ta["passive"][0])))
    initial.update({"passive_poststep_"+key+"_exact": bool(np.array_equal(tb[key][0, passive],ta[key][0, passive]))
                    for key in ("motor_position", "motor_velocity")})
    if not all(initial.values()):
        raise ValueError("Native initial observations or mechanical states differ")
    precontrol_captured = all((p/"evaluation_precontrol_state.npz").exists() for p in (args.before,args.after))
    if precontrol_captured:
        xb, xa = [np.load(p/"evaluation_precontrol_state.npz") for p in (args.before,args.after)]
        for key in xb.files:
            initial["precontrol_"+key+"_exact"] = bool(np.array_equal(xb[key],xa[key]))
        if not all(initial.values()):
            raise ValueError("Native precontrol physical states differ")
    groups = dict(all=lambda r: True, entry=lambda r:r["behavior_case"].startswith("real_start"),
        standing=lambda r:r["behavior_case"].startswith("stand"),
        motion=lambda r:r["behavior_case"] in ("forward", "backward", "arc_forward_left", "arc_forward_right"),
        fallen=lambda r:r["behavior_case"] in ("left_symmetric", "right_symmetric", "inverted_symmetric"))
    result = dict(schema="real-gap-native-comparison-v1", before_checkpoint_sha256=before["checkpoint_sha256"],
        after_checkpoint_sha256=after["checkpoint_sha256"], seed=before["seed"], severity=before["severity"],
        initial_exact=initial, precontrol_physical_state_captured=precontrol_captured,
        historical_pairing_limit=(None if precontrol_captured else "Initial actor inputs and passive physical states match; active physical state before action was not captured"),
        sensor_actuator_material_exact=True, groups={}, full_training_admitted=False,
        qualification=None, hardware_ready=False)
    for name, predicate in groups.items():
        result["groups"][name] = dict(before=aggregate([r for r in sb["rows"] if predicate(r)]),
                                     after=aggregate([r for r in sa["rows"] if predicate(r)]))
    startup_before=json.loads((args.before/"startup.json").read_text())
    startup_after=json.loads((args.after/"startup.json").read_text())
    result["rigid_parameters_exact"]=(startup_before["dynamics_randomization"]["sampled_properties_sha256"]
                                       ==startup_after["dynamics_randomization"]["sampled_properties_sha256"])
    if not result["rigid_parameters_exact"]:
        raise ValueError("Rigid-body sampled parameters differ")
    write_json(args.output,result)
    for name, row in result["groups"].items():
        print(name, json.dumps(row,ensure_ascii=False))


if __name__ == "__main__":
    main()
