"""Read-only frequency ablation on nominal067 inputs, with original-event metrics.

The ideal frequency removal is acausal and diagnostic only. It is never run in
simulation, training or the exported actor. Correlated feedback means individual
feature effects are not additive or a closed-loop causal explanation.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUN = ROOT / "runs/unified_robust_20261022/fresh_nominal067_211131"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from analyze_robust_getup import hf_rms
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import ZeroStandingPosition

GROUPS = dict(gyro=(4, 7), gravity=(7, 10), leg_position=(10, 14), position_reference=(14, 16),
              leg_velocity=(16, 20), wheel_velocity=(20, 22), previous_legs=(22, 26), previous_wheels=(26, 28))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets(action):
    action = np.asarray(action)
    return dict(legs20=hf_rms(action[:, :4] * .25, .005), wheels20=hf_rms(action[:, 4:] * 10., .005),
                legs3=hf_rms(action[:, :4] * .25, .005, 3.), wheels3=hf_rms(action[:, 4:] * 10., .005, 3.))


def main():
    torch.set_num_threads(1)
    mean, _ = load_mean(ROOT / "runs/unified_robust_20261022/quiet_severe_reference065/model_migrated.pt")
    result = dict(schema="quiet080-offline-input-frequency-v1", models={}, new_macro_updates=0,
        native_action_changed=False, acausal_diagnostic_only=True, units="legs rad; wheels rad/s; HF amplitude RMS",
        interpretation="One deterministic nominal condition per height; correlated feature substitutions do not add or prove closed-loop benefit")
    for name, model in (("robust", mean), ("parent", ZeroStandingPosition(mean.parent))):
        directory = RUN / name
        report = json.loads((directory / "report.json").read_text())
        path = directory / "trajectory.npz"
        assert sha(path) == report["trajectory_sha256"]
        data = np.load(path)
        rows = {}
        for case in ("stand_low", "stand", "stand_high"):
            lane = next(row["lane"] for row in report["lanes"] if row["name"] == case)
            selected = data["alive"][:, lane] & (data["time"][:, lane] >= 3.)
            obs = data["observations"][selected, lane].copy()
            x = torch.from_numpy(obs)
            with torch.no_grad():
                action = model(x).numpy()
            applied = data["actions"][selected, lane][:, [0, 1, 3, 4, 2, 5]]
            error = float(np.abs(action - applied).max())
            assert error < 5e-6, (name, case, error)
            base = targets(action)
            substitutions = {}
            for group, (start, end) in GROUPS.items():
                spectrum = np.fft.rfft(obs[:, start:end].astype(np.float64), axis=0)
                frequency = np.fft.rfftfreq(len(obs), .005)
                hf = (frequency >= 20.) & (frequency <= 95.)
                removed = np.fft.irfft(spectrum * hf[:, None], n=len(obs), axis=0)
                alternate = obs.copy()
                alternate[:, start:end] -= removed.astype(np.float32)
                with torch.no_grad():
                    modified = model(torch.from_numpy(alternate)).numpy()
                metrics = targets(modified)
                substitutions[group] = dict(targets=metrics,
                    removed_feature_rms_scaled=float(np.sqrt((removed**2).mean())),
                    wheel20_ratio=metrics["wheels20"] / base["wheels20"],
                    leg20_ratio=metrics["legs20"] / base["legs20"])
            rows[case] = dict(frames=len(obs), cpu_vs_recorded_raw_error=error, baseline_targets=base,
                single_group_fast_feature_removal=substitutions)
        result["models"][name] = dict(cases=rows, trajectory_sha256=report["trajectory_sha256"])
        with SummaryWriter(str(RUN), filename_suffix=".quiet080_input") as writer:
            for case, row in rows.items():
                writer.add_scalar("Diagnosis/QuietHF080/" + name + "/" + case + "/cpu_raw_error", row["cpu_vs_recorded_raw_error"], 505)
                for group, values in row["single_group_fast_feature_removal"].items():
                    for metric in ("removed_feature_rms_scaled", "wheel20_ratio", "leg20_ratio"):
                        writer.add_scalar("Diagnosis/QuietHF080/" + name + "/" + case + "/" + group + "/" + metric, values[metric], 505)
    write(REPORTS / "quiet_hf080_input.json", result)
    print(json.dumps({name: {case: {group: round(row["wheel20_ratio"], 4)
        for group, row in caseinfo["single_group_fast_feature_removal"].items()}
        for case, caseinfo in model["cases"].items()} for name, model in result["models"].items()}), flush=True)


if __name__ == "__main__":
    main()
