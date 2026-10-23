"""Read-only residual-drift observability in completed native standing trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read, CASES
from wheeled_algo.robust_getup import quiet_gate


def main(run_name):
    torch.set_num_threads(1)
    directory, report, _, data = read(run_name)
    progress = json.loads((directory.parent / "progress.json").read_text())
    assert progress["status"] == "completed"
    # Read the authenticated checkpoint used by this completed native replay.
    process = json.loads((REPORTS / (run_name + "_process.json")).read_text())
    assert process["returncode"] == 0 and process["native_status"] == "completed"
    command = process["command"]
    checkpoint = Path(command[command.index("--checkpoint") + 1])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint_sha256"]
    mean, _ = load_mean(checkpoint)
    rows = []
    for lane in report["lanes"]:
        if lane["name"] not in CASES:
            continue
        i = lane["lane"]
        obs = torch.from_numpy(data["observations"][1:, i].copy())
        with torch.inference_mode():
            estimated, valid = mean.rolling_velocity(obs)
            gate = quiet_gate(obs)
        selected = (data["alive"][1:, i] & ~data["passive"][1:, i]
                    & (data["time"][1:, i] >= 3.))
        assert selected.any() and bool(valid[selected].all())
        # The observation precedes its recorded action; align with prior truth.
        true_velocity = data["velocity"][:-1, i, 0][selected]
        estimate = estimated.numpy().ravel()[selected]
        error = estimate - true_velocity
        active = data["alive"][:, i] & ~data["passive"][:, i]
        xy = data["local_position"][active, i, :2]
        radius = np.linalg.norm(xy - xy[:1], axis=-1)
        tail = xy[data["time"][active, i] >= 3.]
        rows.append(dict(lane=i, case=lane["name"], domain_enabled=lane["domain_enabled"],
            full_excursion_m=float(radius.max()), tail_net_xy_m=(tail[-1] - tail[0]).tolist(),
            true_forward_mean_m_s=float(true_velocity.mean()), estimated_forward_mean_m_s=float(estimate.mean()),
            velocity_mean_error_m_s=float(error.mean()), velocity_mae_m_s=float(np.abs(error).mean()),
            quiet_gate_mean=float(gate.numpy().ravel()[selected].mean()),
            feedback_feature_mean=float(np.tanh(50. * estimate).mean()),
            feedback_feature_abs_p95=float(np.quantile(np.abs(np.tanh(50. * estimate)), .95))))
    summary = {}
    for case in CASES:
        group = [r for r in rows if r["case"] == case]
        worst = max(group, key=lambda r: r["full_excursion_m"])
        summary[case] = dict(worst_lane=worst,
            max_abs_mean_velocity_error_m_s=max(abs(r["velocity_mean_error_m_s"]) for r in group),
            within5cm=sum(r["full_excursion_m"] <= .05 for r in group),
            nominal_within5cm=sum(r["full_excursion_m"] <= .05 for r in group if not r["domain_enabled"]),
            nominal_episodes=sum(not r["domain_enabled"] for r in group))
    result = dict(schema="standing-feedback053-observability-v1", run=run_name,
        checkpoint_sha256=report["checkpoint_sha256"], trajectory_sha256=report["trajectory_sha256"],
        source_report_sha256=hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
        optimizer_updates=0, native_runs=0, summary=summary, lanes=rows,
        scope="Offline truth labels for aligned encoder/IMU velocity error only; no actor truth inputs, no position observer or new feedback gain.")
    with SummaryWriter(str(directory.parent), filename_suffix=".standing_feedback053") as writer:
        for case, row in summary.items():
            for key in ("true_forward_mean_m_s", "estimated_forward_mean_m_s", "velocity_mean_error_m_s",
                        "quiet_gate_mean", "feedback_feature_mean"):
                writer.add_scalar(f"Diagnosis/standing_feedback053/{case}/worst_lane/{key}",
                                  row["worst_lane"][key], progress["completed_updates"])
    target = REPORTS / ("standing_feedback053_" + run_name + ".json")
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    main(parser.parse_args().run)
