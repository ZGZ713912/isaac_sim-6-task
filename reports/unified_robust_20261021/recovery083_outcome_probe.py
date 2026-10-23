"""Readonly retention and impact timing after both083 native evaluations."""
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read, paired
from iteration045_runner import write
from wheeled_algo.robust_getup import tilt_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert json.loads((REPORTS / "iteration083_pipeline.json").read_text())["status"] == "completed"
    torch.set_num_threads(1)
    before, after = read("dynamic_recovery_full079"), read("dynamic_recovery_full083")
    matching = paired(before, after)
    ordinary = [r["lane"] for r in after[1]["lanes"] if not r["recovery"] and not r.get("push", False)]
    static = [r["lane"] for r in after[1]["lanes"] if r["name"] in ("stand_low", "stand", "stand_high")]
    keys = ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")
    static_exact = {key: np.array_equal(before[3][key][:, static], after[3][key][:, static]) for key in keys}
    assert len(ordinary) == 84 and len(static) == 12
    movement = {name: {field: dict(before=before[2]["cases"][name][field], after=row[field])
        for field in ("vx_mean_m_s", "yaw_mean_rad_s", "vx_mae_m_s", "yaw_mae_rad_s")}
        for name, row in after[2]["cases"].items() if name.startswith(("arc_", "fast_", "forward", "backward"))}
    timings = {}
    for variant, model in (("reference", "pilot033_dynamic_recovery079"), ("robust", "pilot034_dynamic_recovery083")):
        mean, _ = load_mean(RUNS / model / "model_final.pt")
        directory = RUNS / "dynamic_recovery_target083" / variant
        report = json.loads((directory / "report.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        data = np.load(directory / "trajectory.npz")
        rows = {}
        for case in ("moving_push_left", "moving_push_right"):
            row = next(r for r in report["lanes"] if r["name"] == case)
            lane = row["lane"]
            obs = torch.from_numpy(data["observations"][:, lane].copy())
            gate = tilt_gate(obs, mean.stance_aware_recovery, mean.recovery_shape_aware).numpy().ravel()
            time = data["time"][:, lane]
            # Mask at first termination, since the task keeps simulating dead
            # lanes. These timestamps are post-step labels for input frames.
            selected = data["alive"][:, lane] & (time >= 5.)
            if row["termination_time_s"] is not None:
                selected |= np.abs(time - row["termination_time_s"]) < 1e-4
            def first(mask):
                times = time[selected & mask]
                return float(times[0]) if len(times) else None
            terminal = row["termination_time_s"] or 12.
            window = selected & (time >= terminal - .2)
            rows[case] = dict(survived=row["survived"], recovered_after_push=row["recovered_after_push"],
                termination_time_s=row["termination_time_s"], termination_causes=row["termination_causes"],
                first_actual_fall_s=first(data["in_motion_fall"][:, lane]),
                first_authority90_s=first(gate >= .9), peak_contact_impulse_ns=float(data["impulse"][selected, lane].max()),
                preterminal200ms_leg_speed_peak_rad_s=float(np.abs(data["motor_velocity"][window, lane][:, [0, 1, 3, 4]]).max()),
                peak_gyro_x_rad_s=float(np.abs(data["omega"][selected, lane, 0]).max()))
        timings[variant] = dict(cases=rows, trajectory_sha256=report["trajectory_sha256"])
        writer = SummaryWriter(str(RUNS / "dynamic_recovery_target083"), filename_suffix=".outcome083")
        for case, row in rows.items():
            for key, value in row.items():
                if isinstance(value, (int, float, bool)):
                    writer.add_scalar("Diagnosis/Recovery083/" + variant + "/" + case + "/" + key, value, 697)
        writer.close()
    value = dict(schema="recovery083-retention-impact-v1", status="completed", initial_pairing=matching,
        ordinary_without_strong_push=dict(episodes=84, before=sum(before[1]["lanes"][i]["survived"] for i in ordinary),
            after=sum(after[1]["lanes"][i]["survived"] for i in ordinary)),
        static12_trajectories_bitwise_exact=static_exact, motion=movement, targeted_impact_timing=timings,
        scope="Representative nominal lanes; timing is not a unique causal explanation; dead-lane aftermath excluded",
        new_macro_updates=0, robust_consumed=1384, full_training_admitted=False)
    write(REPORTS / "recovery083_outcome.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
