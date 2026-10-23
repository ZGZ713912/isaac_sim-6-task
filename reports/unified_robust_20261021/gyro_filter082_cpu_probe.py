"""Scope and recorded-transient exposure only; no changed-action native rollout."""
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
from gyro_filter082_prototype import GyroReleasedWheelFilter
from iteration045_runner import write
from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter
from wheeled_algo.robust_getup import quiet_gate


def main():
    torch.set_num_threads(1)
    mean, _ = load_mean(RUNS / "quiet_severe_reference065/model_migrated.pt")
    source = RUNS / "high_recenter_full066/robust/trajectory.npz"
    data = np.load(source)
    obs = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    rng = torch.get_rng_state()
    checks = {}
    with torch.no_grad():
        action = mean(obs)
        filtered = QuietWheelOutputFilter(mean, .25)(obs)
        outside = quiet_gate(obs).ravel() == 0.
        for start, full in ((.05, .2), (.03, .1)):
            model = GyroReleasedWheelFilter(mean, start, full)
            actual = model(obs)
            speed = obs[:, 4:6].norm(dim=-1) * 2.
            assert torch.equal(actual[:, :4], action[:, :4])
            assert torch.equal(actual[outside], action[outside])
            assert torch.equal(actual[speed <= start], filtered[speed <= start])
            assert torch.equal(actual[speed >= full], action[speed >= full])
            checks[str((start, full))] = dict(legs_exact=True, nonquiet_exact=True,
                low_speed_matches_alpha025_exact=True, high_speed_original_exact=True,
                state_keys_unchanged=set(model.state_dict()) == {"mean." + key for key in mean.state_dict()})
        assert torch.equal(rng, torch.get_rng_state())
    exposure = {}
    for run, variant in (("quiet_whole_static074", "robust"), ("fresh_nominal067_211131", "robust")):
        directory = RUNS / run / variant
        report = json.loads((directory / "report.json").read_text())
        path = directory / "trajectory.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["trajectory_sha256"]
        d = np.load(path)
        rows = {}
        for case in ("stand_low", "stand", "stand_high"):
            lanes = [row["lane"] for row in report["lanes"] if row["name"] == case]
            speed = np.linalg.norm(d["observations"][:, lanes, 4:6] * 2., axis=-1)
            time = d["time"][:, lanes]
            rows[case] = {str((start, full)): {window: dict(
                gyro_quantiles_rad_s=np.quantile(speed[mask], [.5, .95, .99, 1]).tolist(),
                retained_full_filter_fraction=float((speed[mask] <= start).mean()),
                released_filter_fraction=float((speed[mask] >= full).mean()),
                mean_alpha=float((.25 + .75 * np.clip((speed[mask] - start) / (full - start), 0., 1.)).mean()))
                for window, mask in (("startup_0_1s", time < 1.), ("steady_3_12s", time >= 3.))}
                for start, full in ((.05, .2), (.03, .1))}
        exposure[run] = dict(cases=rows, trajectory_sha256=report["trajectory_sha256"])
        with SummaryWriter(str(directory.parent), filename_suffix=".gyro082_exposure") as writer:
            for case, row in rows.items():
                for band, windows in row.items():
                    for window, metrics in windows.items():
                        for key, value in metrics.items():
                            if isinstance(value, list):
                                for quantile, metric in zip((50, 95, 99, 100), value):
                                    writer.add_scalar("Diagnosis/GyroFilter082/" + case + "/" + band + "/" + window + "/gyro_p" + str(quantile) + "_rad_s", metric, 505)
                            else:
                                writer.add_scalar("Diagnosis/GyroFilter082/" + case + "/" + band + "/" + window + "/" + key, value, 505)
    result = dict(schema="quiet082-gyro-release-cpu-prototype-v1", frames=len(obs), nonquiet=int(outside.sum()),
        checks=checks, exposure=exposure, rng_unchanged=True, new_macro_updates=0,
        source_trajectory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_behavior_changed=False, prototype_installed=False,
        limitation="Envelope exposure on previous trajectories only; changed-action balance/HF behavior remains untested")
    write(REPORTS / "gyro_filter082_cpu.json", result)
    print(json.dumps(dict(frames=len(obs), checks=checks)), flush=True)


if __name__ == "__main__":
    main()
