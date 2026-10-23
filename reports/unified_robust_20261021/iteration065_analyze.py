"""Audit actor-input reference state through native stationary push recovery."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from wheeled_algo.robust_getup import quiet_gate, tilt_gate


def load(name, overrides):
    directory = RUNS / name / "robust"
    report = json.loads((directory / "report.json").read_text())
    trajectory = directory / "trajectory.npz"
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == report["trajectory_sha256"]
    assert json.loads((directory.parent / "progress.json").read_text())["status"] == "completed"
    assert report["evaluation_overrides"] == overrides
    return directory, report, dict(np.load(trajectory))


def reference_trace_violations(obs, active, count, age, filtered, anchor, alive, task):
    """Validate actual input-aligned state, including deliberately bad traces."""
    norm = np.linalg.norm(obs[:, 7:10], axis=-1)
    finite = np.isfinite(obs[:, 7:10]).all(-1) & (norm > .5) & (norm < 1.5)
    severe = finite & (obs[:, 9] / np.maximum(norm, 1e-9) > -.5) & alive
    violations = []
    if np.any(active[severe]):
        violations.append("active reference on severe sensor tilt")
    if np.any(filtered[severe] != 0.) or np.any(anchor[severe] != 0.):
        violations.append("stale reference/filter on severe sensor tilt")
    increments = np.diff(count, prepend=0)
    previous_active = np.r_[False, active[:-1]]
    if np.any(((increments < 0) | (increments > 1) | ((increments == 1) & ~previous_active)) & alive):
        violations.append("invalid severe-reset count transition")
    captures = active & ~previous_active
    for tick in np.flatnonzero(captures & (count > 0) & alive):
        gravity_valid = finite[tick] and obs[tick, 9] / norm[tick] < -.94
        age_valid = age[tick] >= .05 - 1e-7
        x = torch.from_numpy(obs[tick:tick + 1].copy())
        eligible = (quiet_gate(x)[0, 0] >= .999 and
            tilt_gate(x, task["stance_aware_recovery"], task["recovery_shape_aware"])[0, 0] <= .05)
        if not (gravity_valid and age_valid and eligible):
            violations.append(f"ineligible recapture at frame {tick}")
        if np.any(filtered[tick] != 0.) or not np.isfinite(anchor[tick]).all():
            violations.append(f"invalid fresh reference at frame {tick}")
    return violations


def main(iteration=65):
    torch.set_num_threads(1)
    assert iteration in (65, 69)
    overrides = dict(stationary_push_cases=True)
    if iteration == 69:
        overrides["stationary_push_scale"] = .5
    before, after = [load(f"stationary_reference_{side}{iteration:03d}", overrides) for side in ("before", "after")]
    for field in ("name", "pose_index", "command", "push", "push_sign", "world_origin_m",
                  "first_reset_domain", "material_coefficients"):
        assert [r.get(field) for r in before[1]["lanes"]] == [r.get(field) for r in after[1]["lanes"]], field
    first_push = int(np.flatnonzero(after[2]["pushing"].any(-1))[0])
    for field in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive"):
        assert np.array_equal(before[2][field][:first_push], after[2][field][:first_push]), field
    result = dict(schema=f"reference{iteration:03d}-native-lifecycle-audit-v1", before={}, after={},
                  new_macro_updates=0, robust_consumed=1160, long_training_admitted=False,
                  matched_first_states_domains_materials=True, prepush_traces_bitwise_exact=True,
                  prepush_frames=first_push, diagnostic_only=True, hardware_ready=False)
    result["evaluation_overrides"] = overrides
    violations = []
    task = json.loads((after[0].parent / "task_contract.json").read_text())["settings"]
    for label, (directory, report, data) in (("before", before), ("after", after)):
        rows = []
        for row in report["lanes"]:
            i = row["lane"]
            t, alive = data["time"][:, i], data["alive"][:, i]
            obs = data["observations"][:, i]
            norm = np.linalg.norm(obs[:, 7:10], axis=-1)
            finite = np.isfinite(obs[:, 7:10]).all(-1) & (norm > .5) & (norm < 1.5)
            severe = finite & (obs[:, 9] / np.maximum(norm, 1e-9) > -.5) & alive
            active = data["standing_reference_active"][:, i]
            count = data["standing_reference_severe_reset_count"][:, i]
            captures = active & ~np.r_[False, active[:-1]]
            recapture_indices = np.flatnonzero(captures & (count > 0) & alive)
            last = alive & (t > 10.)
            good = data["stable"][last, i] & (np.abs(data["velocity"][last, i, 0]) < .1)
            good &= np.abs(data["omega"][last, i, 2]) < .2
            xy = data["local_position"][last, i, :2]
            excursion = float(np.linalg.norm(xy - xy[:1], axis=-1).max()) if len(xy) else None
            stable_fraction = float(good.mean()) if len(good) else 0.
            completed = bool(row["survived"] and len(good) >= 399)
            practical = completed and row["in_motion_fall"] and stable_fraction >= .95 and excursion <= .05
            entry = dict(lane=i, case=row["name"], survived=row["survived"],
                prepush_reference_active=bool(active[max(0, first_push - 20):first_push].all()),
                severe_input_frames=int(severe.sum()), severe_reference_resets=int(count[alive].max()),
                recapture_times_s=t[recapture_indices].tolist(), actual_fall=row["in_motion_fall"],
                recovered_after_push=row["recovered_after_push"], practical_terminal_standing=practical,
                terminal_stable_fraction=stable_fraction, terminal_excursion_m=excursion,
                termination_causes=row["termination_causes"], max_gap_m=row["max_gap_m"])
            if label == "after":
                if not entry["prepush_reference_active"]:
                    violations.append(f"lane {i}: no stable active reference before push")
                violations.extend(f"lane {i}: {message}" for message in reference_trace_violations(
                    obs, active, count, data["standing_reference_age_s"][:, i],
                    data["standing_reference_filtered_m"][:, i], data["standing_reference_anchor_m"][:, i],
                    alive, task))
            rows.append(entry)
        summary = {}
        for case in ("stationary_push_left", "stationary_push_right"):
            selected = [row for row in rows if row["case"] == case]
            summary[case] = dict(episodes=len(selected), survived=sum(r["survived"] for r in selected),
                actual_falls=sum(r["actual_fall"] for r in selected),
                severe_reference_resets=sum(r["severe_reference_resets"] > 0 for r in selected),
                recaptured=sum(bool(r["recapture_times_s"]) for r in selected),
                practical_terminal_standing=sum(r["practical_terminal_standing"] for r in selected),
                exercised_reset_recapture_stand=sum(r["severe_reference_resets"] > 0 and
                    bool(r["recapture_times_s"]) and r["practical_terminal_standing"] for r in selected))
        result[label] = dict(summary=summary, lanes=rows, trajectory_sha256=report["trajectory_sha256"],
                             checkpoint_sha256=report["checkpoint_sha256"])
        step = json.loads((directory.parent / "progress.json").read_text())["completed_updates"]
        writer = SummaryWriter(str(directory.parent), filename_suffix=f".stationary_reference{iteration:03d}")
        for case, metrics in summary.items():
            for key, value in metrics.items():
                writer.add_scalar(f"StationaryReference/{case}/{key}", value, step)
        writer.close()
    result["lifecycle_violations"] = violations
    exercised = all(row["exercised_reset_recapture_stand"] >= 1 for row in result["after"]["summary"].values())
    result["native_lifecycle_exercised_and_valid"] = exercised and not violations
    result["mechanical_gate_passed"] = all(row["max_gap_m"] <= .003 and not
        set(row["termination_causes"]) & {"mechanical", "closure_gap"} for row in result["after"]["lanes"])
    result["scope"] = "One matched randomized stationary push pair; does not replace full-suite/long-standing admission"
    (REPORTS / f"iteration{iteration:03d}_final.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("native_lifecycle_exercised_and_valid", "lifecycle_violations")}))
    print(json.dumps(result["after"]["summary"]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=[65, 69], default=65)
    main(parser.parse_args().iteration)
