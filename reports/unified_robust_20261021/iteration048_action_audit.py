"""Replay diagnostic native actions and record actual internal request corrections."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean
from wheeled_algo.evaluation_command_feedback import StandingCommandFeedbackParent


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "quiet_coupled_state045/model_migrated.pt"
    results = []
    for label, gain in (("negative", -2.), ("positive", 2.)):
        directory = RUNS / f"command_feedback_{label}048" / "robust"
        report = json.loads((directory / "report.json").read_text())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint_sha256"]
        assert hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
        data = np.load(directory / "trajectory.npz")
        active = data["alive"][::60] & ~data["passive"][::60]
        x = torch.from_numpy(data["observations"][::60][active].copy())
        model, info = load_mean(checkpoint)
        model.parent = StandingCommandFeedbackParent(model.parent, gain)
        with torch.inference_mode():
            expected = model(x).numpy()
            correction = model.parent.command_delta(x).numpy()
        actual = data["actions"][::60][active][:, [0, 1, 3, 4, 2, 5]]
        error = float(np.abs(expected - actual).max())
        assert np.isfinite(expected).all() and error < 2e-4
        assert np.array_equal(data["observations"][:, :, 0], np.zeros_like(data["time"]))
        row = dict(label=label, gain=gain, active_recorded_inputs=len(x),
            maximum_raw_action_error=error, trajectory_sha256=report["trajectory_sha256"],
            request_correction_rms_m_s=float(np.square(correction).mean() ** .5),
            request_correction_max_m_s=float(np.abs(correction).max()),
            request_correction_saturation_fraction=float((np.abs(correction) >= .15 - 1e-7).mean()),
            recorded_outer_forward_request_zero=True)
        with SummaryWriter(str(directory.parent), filename_suffix=".command_feedback048") as writer:
            for key in ("maximum_raw_action_error", "request_correction_rms_m_s", "request_correction_saturation_fraction"):
                writer.add_scalar("Diagnosis/standing_command_feedback/" + key, row[key], info["completed_updates"])
        results.append(row)
    output = dict(schema="quiet048-native-action-audit-v1", optimizer_updates=0,
        results=results, scope="CPU authenticated policy plus declared fixed-gain intervention versus recorded CUDA actions; outer requests remain zero")
    (REPORTS / "iteration048_action_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
