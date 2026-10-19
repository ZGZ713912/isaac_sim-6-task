"""Locate the observed960 low-stand regression without changing the policy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flat_base_20261019'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

target = HERE / 'static896_960_window_audit001.json'
if target.exists():
    raise SystemExit('Preserve the existing static diagnostic')
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = {}
for step, pilot in ((896, 'pilot017'), (960, 'pilot018')):
    checkpoint = RUN / pilot / f'model_{step:06d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    mean = mean_from_checkpoint(state, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx').eval()
    path = RUN / f'feedback{step}_basic001/traces.npz'
    z = np.load(path)
    time = z['episode_ticks'] * .02
    relative_xy = z['position'][:, :, :2] - z['position'][0:1, :, :2]
    distance = np.linalg.norm(relative_xy, axis=-1)
    cases = {}
    for name in ('stand_230mm', 'stand_305mm', 'stand_430mm'):
        ids = np.flatnonzero(z['groups'] == name)
        windows = {}
        for label, lo, hi in (('birth_0_1s', 0., 1.), ('early_1_3s', 1., 3.), ('late_3_10s', 3., 10.)):
            mask = z['active'] & (z['groups'] == name)[None, :] & (time > lo) & (time <= hi)
            obs = torch.from_numpy(z['observations'][mask])
            with torch.inference_mode():
                features = (obs[:, :28] - mean.input_center) * mean.input_scale
                bounded = torch.tanh(mean.feedback(features)).numpy()
            windows[label] = dict(sample_frames=int(mask.sum()),
                root_vx_signed_mean_m_s=float(z['velocity'][:, :, 0][mask].mean()),
                root_vx_mae_m_s=float(np.abs(z['velocity'][:, :, 0][mask]).mean()),
                displacement_from_first_recorded_frame_max_m=float(distance[mask].max()),
                correction_at95percent_bound_fraction_per_axis=(np.abs(bounded) >= .95).mean(0).tolist(),
                correction_bound_usage_max_per_axis=np.abs(bounded).max(0).tolist())
        cases[name] = dict(windows=windows, lanes=[dict(lane=int(i),
            drift_max_m=float(distance[z['active'][:, i], i].max()),
            final_relative_xy_m=relative_xy[z['active'][:, i], i][-1].tolist()) for i in ids])
    evaluation = json.loads((path.parent / 'evaluation.json').read_text())
    rows[str(step)] = dict(checkpoint_sha256=sha(checkpoint), traces_sha256=sha(path), cases=cases,
        settled_stop_speed_m_s=evaluation['result']['cases']['start_stop_05']['settled_stop_speed'])
receipt = dict(schema='flat-static896-960-birth-and-late-drift-audit-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), rows=rows,
    scope='Read-only active development replay frames relative to first recorded post-action position; '
        'authoritative complete-window acceptance remains unchanged.',
    optimizer_updates_performed=0, used_for_optimization=False, behavior_qualified=False)
target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps({step: dict(settled_stop_speed_m_s=row['settled_stop_speed_m_s'],
    low=row['cases']['stand_230mm']) for step, row in rows.items()}, indent=2))
