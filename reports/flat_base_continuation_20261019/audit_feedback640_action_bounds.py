"""Read-only attribution of bounded branches on recorded native observations."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(1)
RUN = ROOT / 'runs/flat_base_20261019'
checkpoint = RUN / 'pilot013/model_000640.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
model = mean_from_checkpoint(state, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx').eval()
results = {}
for cohort in ('rmcs640_envelope001', 'rmcs_default640_envelope001'):
    trace = RUN / cohort / 'traces.npz'
    if not trace.exists():
        continue
    evaluation = json.loads((trace.parent/'evaluation.json').read_text())
    if evaluation['status'] != 'evaluated':
        continue
    data = np.load(trace)
    rows = {}
    for case in ('stand_230mm','stand_305mm','stand_430mm','forward_3','backward_3','curve_low','curve_low_reverse'):
        lanes = data['groups'] == case
        observations = data['observations'][:,lanes]
        times = data['episode_ticks'][:,lanes] * .02
        active = data['active'][:,lanes]
        parts = {}
        for label,start,end in (('first_2s',0,2),('steady_3_to_9s',3,9)):
            mask = active & (times >= start) & (times < end)
            x = torch.from_numpy(observations[mask])
            if not len(x):
                continue
            with torch.inference_mode():
                raw = torch.tanh(model.feedback((x[:,:28]-model.input_center)*model.input_scale))
                gate = model.tracking_feedback_gate(x)
                feedback = model.feedback_actions(x)
                command = model.command.residual_actions(x)
                base = model.command.base(x)
            parts[label] = dict(samples=len(x), gate_zero_fraction=float((gate==0).float().mean()),
                gate_full_fraction=float((gate==1).float().mean()),
                feedback_tanh_abs_gt_point9_fraction_by_axis=(raw.abs()>.9).float().mean(0).tolist(),
                source_action_mean=base.mean(0).tolist(), command_offset_mean=command.mean(0).tolist(),
                feedback_offset_mean=feedback.mean(0).tolist(), feedback_offset_abs_max=feedback.abs().amax(0).tolist())
        rows[case] = parts
    results[cohort] = dict(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), cases=rows)
report = dict(schema='flat-feedback640-action-bound-audit-v1',
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    optimizer_updates_performed=0, counterfactual_closed_loop_performed=False, cohorts=results,
    limitation='Same-input branch values do not establish the sole cause of a closed-loop tracking error')
Path(__file__).with_name('feedback640_action_bound_audit001.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({c:{n:{p:dict(gate_zero=r['gate_zero_fraction'], saturation_max=max(r['feedback_tanh_abs_gt_point9_fraction_by_axis'])) for p,r in parts.items()} for n,parts in row['cases'].items()} for c,row in results.items()}))
