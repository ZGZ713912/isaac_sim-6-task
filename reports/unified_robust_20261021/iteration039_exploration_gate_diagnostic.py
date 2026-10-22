"""Read-only sampler sensitivity on recorded baseline states; no causal verdict."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import SensorConditionedMean, quiet_gate, tilt_gate

torch.set_num_threads(1)
checkpoint = ROOT / "runs/unified_robust_20261022/quiet_axis_exploration039/model_migrated.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
mean, info = load_mean(checkpoint)
task = info["settings"]
wrapper = SensorConditionedMean(mean, task["quiet_exploration_fraction"], task["quiet_wheel_exploration_fraction"])
wrapper.load_state_dict({k.removeprefix("mlp."): v for k, v in state["actor_state_dict"].items()}, strict=True)
trajectory = ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust/trajectory.npz"
data = np.load(trajectory)
x = torch.from_numpy(data["observations"].reshape(-1, 35).copy())
t = torch.from_numpy(data["time"].reshape(-1).copy())
fraction = x.new_tensor([task["quiet_exploration_fraction"]] * 4 + [task["quiet_wheel_exploration_fraction"]] * 2)
quiet = quiet_gate(x)
recovery = tilt_gate(x, task["stance_aware_recovery"], task["recovery_shape_aware"])
base_std = wrapper.log_std.detach().clamp(-3., -.22314355).exp()
std = base_std * (.12 + .88 * recovery) * (1. - (1. - fraction) * quiet)
nominal_std = base_std * .12 * fraction
selected = torch.tensor([0, int(std[:, 0].argmax()), int(std[:, 4].argmax()), len(x) - 1])
with torch.inference_mode():
    torch.testing.assert_close(wrapper(x[selected])[:, 1].exp(), std[selected], rtol=1e-6, atol=1e-8)
rows = []
for start, end in ((0., .2), (.2, 3.), (3., 12.001)):
    mask = (t >= start) & (t < end)
    ratio = std[mask] / nominal_std
    rows.append(dict(interval_s=[start, min(end, 12.)], states=int(mask.sum()),
        quiet_gate_below_099_fraction=float((quiet[mask] < .99).float().mean()),
        raw_leg_std_mean=float(std[mask, :4].mean()), raw_leg_std_max=float(std[mask, :4].max()),
        raw_wheel_std_mean=float(std[mask, 4:].mean()), raw_wheel_std_max=float(std[mask, 4:].max()),
        leg_std_peak_ratio_vs_quiet=float(ratio[:, :4].max()),
        wheel_std_peak_ratio_vs_quiet=float(ratio[:, 4:].max())))
result = dict(schema="quiet039-counterfactual-sampler-v1", new_optimizer_updates=0,
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
    actor_source_sha256=hashlib.sha256((ROOT / "src/wheeled_algo/robust_getup.py").read_bytes()).hexdigest(),
    recorded_static_states=len(x), nominal_quiet_raw_std=nominal_std.tolist(), intervals=rows,
    scope="039 sampler evaluated on all115200 recorded baseline states; these are not039 training states. Noise amplification sensitivity is not a unique cause or closed-loop intervention.")
(Path(__file__).parent / "iteration039_exploration_gate_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(rows))
