"""Read-only timing evidence from032; no new policy, simulation or PPO."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import tilt_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    run = RUNS / "side_confirmation032/robust"
    checkpoint = RUNS / "pilot020_side_recovery032/model_final.pt"
    report = json.loads((run / "report.json").read_text())
    analysis = json.loads((run / "analysis.json").read_text())
    assert sha(run / "trajectory.npz") == report["trajectory_sha256"]
    assert sha(checkpoint) == report["checkpoint_sha256"] == analysis["checkpoint_sha256"]
    assert sha(run / "report.json") == analysis["report_sha256"]
    mean, info = load_mean(checkpoint, migration_source_root=checkpoint.parent / "source")
    settings = info["settings"]
    assert settings["stance_aware_recovery"] and settings["recovery_shape_aware"]
    data = np.load(run / "trajectory.npz")
    rows, errors = [], []
    for case in report["lanes"]:
        if not case.get("push"):
            continue
        lane = case["lane"]
        time = data["time"][:, lane].astype(np.float64)
        alive = data["alive"][:, lane]
        post_push = alive & (time >= settings["push_start_seconds"])
        obs = torch.from_numpy(data["observations"][:, lane].copy())
        with torch.inference_mode():
            gate = tilt_gate(obs, True, True).numpy().ravel()
            predicted = mean(obs).numpy()
        recorded = data["actions"][:, lane][:, [0, 1, 3, 4, 2, 5]]
        error = float(np.abs(predicted[alive] - recorded[alive]).max())
        assert error < 2e-4, (lane, error)
        errors.append(error)
        tilt = np.rad2deg(np.arccos(np.clip(-data["gravity"][:, lane, 2], -1., 1.)))
        observed_tilt = np.rad2deg(np.arccos(np.clip(-obs[:, 9].numpy(), -1., 1.)))

        def first(mask):
            indices = np.flatnonzero(post_push & mask)
            return (int(indices[0]), float(time[indices[0]])) if len(indices) else (None, None)

        events = {}
        for name, mask in (("external_push", data["pushing"][:, lane]),
                           ("true_tilt45", tilt >= 45.), ("true_tilt60", tilt >= 60.),
                           ("observed_tilt60", observed_tilt >= 60.),
                           ("gate01", gate >= .1), ("gate05", gate >= .5), ("gate09", gate >= .9),
                           ("both_wheels_no_contact", ~data["wheel_contact"][:, lane].any(-1)),
                           ("environment_fall", data["in_motion_fall"][:, lane]),
                           ("impact_guard", data["impact_bad"][:, lane])):
            _, events[name] = first(mask)
        at_fall, _ = first(data["in_motion_fall"][:, lane])
        at_impact, _ = first(data["impact_bad"][:, lane])
        lead = (events["impact_guard"] - events["gate09"]
                if events["impact_guard"] is not None and events["gate09"] is not None else None)
        rows.append(dict(lane=lane, name=case["name"], domain_enabled=case["domain_enabled"],
            effective_wheel_mu=case["material_coefficients"]["effective_wheel_dynamic"],
            survived=case["survived"], recovered_after_push=case["recovered_after_push"],
            practical_followup=analysis["practical_followup_by_lane"][lane], termination=case["termination_causes"],
            events_post_step_timestamp_s=events,
            gate_at_first_environment_fall=float(gate[at_fall]) if at_fall is not None else None,
            gate_at_impact_guard=float(gate[at_impact]) if at_impact is not None else None,
            true_tilt_at_first_environment_fall_deg=float(tilt[at_fall]) if at_fall is not None else None,
            observed_xy_gyro_post_push_peak_rad_s=float((obs[post_push, 4:6].norm(dim=-1) * 2.).max()),
            gate09_to_impact_guard_seconds=lead))
    ordinary = np.array([c["lane"] for c in report["lanes"] if not c["recovery"] and not c.get("push")])
    observation = data["observations"][:, ordinary]
    valid = data["alive"][:, ordinary]
    ordinary_tilt = np.rad2deg(np.arccos(np.clip(-observation[:, :, 9], -1., 1.)))
    ordinary_gyro = np.linalg.norm(observation[:, :, 4:6] * 2., axis=-1)
    result = dict(schema="moving-fall046-readonly-timing-v1", source_report=str((run / "report.json").relative_to(ROOT)),
        source_report_sha256=sha(run / "report.json"), trajectory_sha256=sha(run / "trajectory.npz"),
        checkpoint_sha256=sha(checkpoint), optimizer_updates=0, native_runs=0,
        action_replay_max_raw_error=max(errors), rows=rows,
        ordinary_nonpush_observed_tilt_peak_deg=float(ordinary_tilt[valid].max()),
        ordinary_nonpush_observed_xy_gyro_peak_rad_s=float(ordinary_gyro[valid].max()),
        scope="Recorded action-input gate and post-step physical events. Timestamp bins are5ms; sensors may be delayed. Descriptive timing does not establish a causal policy intervention or physical safety limit.",
        interpretation="Check whether full recovery authority already precedes impact before assuming a late tilt gate is the only cause. No gate, reward, impact threshold, material or policy was changed.",
        hardware_ready=False)
    # Derived behavior metrics use the existing RSL-RL run's event stream.
    # This is a finite offline writer; no TensorBoard server is started.
    from torch.utils.tensorboard import SummaryWriter
    leads = [r["gate09_to_impact_guard_seconds"] for r in rows if r["gate09_to_impact_guard_seconds"] is not None]
    metrics = {"push_episodes": len(rows), "impact_terminated_episodes": len(leads),
               "gate09_to_impact_guard_min_s": min(leads),
               "gate09_to_impact_guard_median_s": float(np.median(leads)),
               "gate09_to_impact_guard_max_s": max(leads),
               "ordinary_nonpush_observed_tilt_peak_deg": result["ordinary_nonpush_observed_tilt_peak_deg"],
               "ordinary_nonpush_observed_xy_gyro_peak_rad_s": result["ordinary_nonpush_observed_xy_gyro_peak_rad_s"]}
    with SummaryWriter(log_dir=str(run.parent), filename_suffix=".fall_timing046") as writer:
        for tag, value in metrics.items():
            writer.add_scalar("Diagnosis/moving_fall/" + tag, value, info["completed_updates"])
    result["tensorboard"] = dict(run=str(run.parent.relative_to(ROOT)), step=info["completed_updates"],
                                 tag_prefix="Diagnosis/moving_fall/", scalar_values=metrics)
    target = REPORTS / "moving_fall_timing046.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
