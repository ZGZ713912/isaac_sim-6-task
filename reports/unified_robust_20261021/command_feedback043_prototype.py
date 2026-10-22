"""Read-only local response of the retained RL parent to forward commands.

This records a potential coordinated feedback direction, not its closed-loop
effect. No model, optimizer, native task or deployed file is modified.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import quiet_gate, tilt_gate

torch.set_num_threads(1)
checkpoint = ROOT / "runs/unified_robust_20261022/quiet_wheel_memory040/model_migrated.pt"
mean, _ = load_mean(checkpoint)
directory = ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust"
report = json.loads((directory / "report.json").read_text())
data = np.load(directory / "trajectory.npz")
assert report["trajectory_sha256"] == hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest()
x = torch.from_numpy(data["observations"][[599, 1199, 1799]].reshape(-1, 35).copy())
with torch.inference_mode():
    plus, minus = x.clone(), x.clone()
    plus[:, 0] += .001
    minus[:, 0] -= .001
    derivative = (mean.parent(plus) - mean.parent(minus)) / .002
    # Only the original smoothed parent path is considered. The existing
    # quiet feature head and original quiet/recovery gates keep their inputs.
    derivative *= quiet_gate(x) * mean.quiet_alpha * (1. - tilt_gate(x, True, True))
rows = {}
for case in ("stand_low", "stand", "stand_high"):
    ids = [i for i in range(len(x)) if report["lanes"][i % 48]["name"] == case]
    rows[case] = derivative[ids].mean(0).tolist()
result = dict(schema="quiet043-coordinated-command-prototype-v1", optimizer_updates=0,
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), real_observations=len(x),
    raw_action_derivative_per_m_s_forward_request=rows,
    scope="Local derivative of a frozen policy path only; does not predict DC motion, stability, drift, or vibration",
    proposed_native_test="Bound a standing-only forward request correction from the existing wheel encoder proxy, retaining original quiet/recovery authority and all randomization",
    implementation_status="read_only_prototype_not_in_native_actor_or_delivery")
(REPORTS / "command_feedback043_prototype.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
