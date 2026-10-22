"""Read-only original policy clip audit and matched per-lane quiet differences."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src")]
from wheeled_algo.onnx_actor_import import recover_actor

torch.set_num_threads(1)
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
model, _ = recover_actor(source)
directory = ROOT / "runs/unified_robust_20261022/deployed_static041/deployed"
report = json.loads((directory / "report.json").read_text())
data = np.load(directory / "trajectory.npz")
assert report["trajectory_sha256"] == hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest()
clips = {}
with torch.inference_mode():
    for case in ("stand_low", "stand", "stand_high"):
        lanes = [x["lane"] for x in report["lanes"] if x["name"] == case]
        observations = data["observations"][:, lanes]
        alive = data["alive"][:, lanes]
        raw = model(torch.from_numpy(observations[alive].copy())).numpy()
        clips[case] = dict(samples=len(raw), leg_clip_frame_fraction=float((np.abs(raw[:, :4]) > 3.).any(-1).mean()),
            wheel_clip_frame_fraction=float((np.abs(raw[:, 4:]) > 9.).any(-1).mean()),
            raw_leg_peak=float(np.abs(raw[:, :4]).max()), raw_wheel_peak=float(np.abs(raw[:, 4:]).max()),
            terminated_lanes=[dict(lane=x["lane"], seconds=x["termination_time_s"], causes=x["termination_causes"])
                              for x in report["lanes"] if x["lane"] in lanes and x["terminated"]])
result = dict(schema="original041-output-audit-v1", optimizer_updates=0,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), clips=clips,
    scope="Clip incidence on the recorded native original-policy trajectory, not a counterfactual no-clip replay or RMCS hardware validation")
(REPORTS / "original_static041_output_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
