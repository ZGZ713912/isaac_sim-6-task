"""Read same-input source-action reachability on recorded high-motion frames.

The source action is a diagnostic target, not a required action in a new state.
Its reachability is necessary for exact cancellation only, not for tracking.
"""
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

target = HERE / 'feedback896_allframe_capacity_audit001.json'
if target.exists():
    raise SystemExit('Preserve the existing diagnostic receipt')
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checkpoint = RUN / 'pilot017/model_000896.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
mean = mean_from_checkpoint(state, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx').eval()
rows = {}
for cohort in ('feedback896_envelope001', 'rmcs896_envelope001', 'rmcs_default896_envelope001'):
    path = RUN / cohort / 'traces.npz'
    z = np.load(path)
    cases = {}
    for name in ('forward_3', 'backward_3', 'rotate_4', 'rotate_4_reverse'):
        mask = z['active'] & (z['groups'] == name)[None, :]
        obs = torch.from_numpy(z['observations'][mask])
        times = z['episode_ticks'][mask] * .02
        if not len(obs):
            raise ValueError('Missing high-motion frames: ' + cohort + '/' + name)
        with torch.inference_mode():
            command_offset = mean.command.residual_actions(obs)
            gate = mean.tracking_feedback_gate(obs)
            available = mean.output_scale * gate
            reachable = (command_offset.abs() <= available + 1e-7).all(-1).numpy()
            correction = mean.feedback_actions(obs)
            saturation = (correction.abs() >= available * .95) & (available > 1e-7)
        windows = {}
        for label, lo, hi in (('all_active', 0., 40.), ('birth_0_1s', 0., 1.),
                               ('transient_1_3s', 1., 3.), ('steady_5_9s', 5., 9.)):
            selected = (times > lo) & (times <= hi)
            if not selected.any():
                continue
            windows[label] = dict(sample_frames=int(selected.sum()),
                source_action_reachable_fraction=float(reachable[selected].mean()),
                gate_zero_fraction=float((gate.numpy()[selected, 0] <= 1e-7).mean()),
                gate_mean=float(gate.numpy()[selected].mean()),
                correction_at95percent_available_bound_fraction_per_axis=saturation.numpy()[selected].mean(0).tolist(),
                raw_action_difference_from_source_RMSE=float(np.sqrt(
                    (command_offset.numpy()[selected] + correction.numpy()[selected]).astype(np.float64).__pow__(2).mean())))
        cases[name] = windows
    rows[cohort] = dict(trace_sha256=sha(path), cases=cases)
receipt = dict(schema='flat-feedback896-allframe-source-action-capacity-audit-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), checkpoint_sha256=sha(checkpoint),
    rows=rows, used_for_optimization=False, optimizer_updates_performed=0,
    native_environment_steps=0, behavior_qualified=False,
    limitations='Same-input source action is not necessarily the desired control at these visited states. '
        'A gate-limited cancellation target neither proves nor disproves attainable velocity precision '
        'or identifies a unique cause. This does not change bounds, gates or acceptance.')
target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps({c: {n: v['steady_5_9s'] for n, v in r['cases'].items()} for c, r in rows.items()}, indent=2))
