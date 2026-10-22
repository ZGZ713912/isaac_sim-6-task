"""Check that045/047 recorded native actions match their authenticated PTs."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    results = []
    for name, model in (("coupled_state_static045", "pilot026_coupled_state045/model_final.pt"),
                        ("bias_ablation_static047", "quiet_bias_ablation047/model_ablated.pt")):
        directory, checkpoint = RUNS / name / "robust", RUNS / model
        report = json.loads((directory / "report.json").read_text())
        assert sha(checkpoint) == report["checkpoint_sha256"]
        assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
        data = np.load(directory / "trajectory.npz")
        selected = data["alive"][::60] & ~data["passive"][::60]
        observations = torch.from_numpy(data["observations"][::60][selected].copy())
        actions = data["actions"][::60][selected][:, [0, 1, 3, 4, 2, 5]]
        mean, info = load_mean(checkpoint)
        with torch.inference_mode():
            expected = mean(observations).numpy()
        error = float(np.abs(expected - actions).max())
        assert np.isfinite(expected).all() and error < 2e-4
        with SummaryWriter(log_dir=str(directory.parent), filename_suffix=".action_replay047") as writer:
            writer.add_scalar("Diagnosis/checkpoint_action_replay/max_raw_action_error", error, info["completed_updates"])
        results.append(dict(run=name, checkpoint_sha256=sha(checkpoint), trajectory_sha256=report["trajectory_sha256"],
            active_recorded_inputs=len(observations), maximum_raw_action_error=error,
            tensorboard_step=info["completed_updates"], motor_order_authenticated=True))
    result = dict(schema="quiet047-native-action-replay-v1", results=results,
        scope="CPU PT inference versus recorded native CUDA mean before the unchanged actuator decode; sampled across the full12s and all48lanes",
        new_optimizer_updates=0, hardware_ready=False)
    (REPORTS / "iteration047_native_action_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
