"""Confirm the rolling estimator intervened in the actual native policy path."""
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
from wheeled_algo.rolling_velocity import RollingBaseVelocity


def main():
    torch.set_num_threads(1)
    directory = RUNS / "rolling_command_feedback049/robust"
    report = json.loads((directory / "report.json").read_text())
    assert hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
    table = ROOT / "reports/flat_base_continuation_20261019/low_friction_candidate001/height_lookup.json"
    checkpoint = RUNS / "quiet_coupled_state045/model_migrated.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint_sha256"]
    mean, info = load_mean(checkpoint)
    mean.parent = StandingCommandFeedbackParent(mean.parent, 2., RollingBaseVelocity(json.loads(table.read_text())))
    data = np.load(directory / "trajectory.npz")
    active = data["alive"][::60] & ~data["passive"][::60]
    x = torch.from_numpy(data["observations"][::60][active].copy())
    with torch.inference_mode():
        expected = mean(x).numpy()
        delta = mean.parent.command_delta(x).numpy()
    actual = data["actions"][::60][active][:, [0, 1, 3, 4, 2, 5]]
    error = float(np.abs(actual - expected).max())
    assert error < 2e-4
    result = dict(schema="rolling049-action-audit-v1", optimizer_updates=0,
        checkpoint_sha256=report["checkpoint_sha256"], trajectory_sha256=report["trajectory_sha256"],
        actual_native_inputs=len(x), maximum_raw_action_error=error,
        request_correction_rms_m_s=float(np.square(delta).mean()**.5),
        request_correction_saturation_fraction=float((np.abs(delta) >= .15 - 1e-7).mean()))
    with SummaryWriter(str(directory.parent), filename_suffix=".action_audit049") as writer:
        for key in ("maximum_raw_action_error", "request_correction_rms_m_s", "request_correction_saturation_fraction"):
            writer.add_scalar("Diagnosis/rolling_command_feedback/" + key, result[key], info["completed_updates"])
    (REPORTS / "iteration049_action_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
