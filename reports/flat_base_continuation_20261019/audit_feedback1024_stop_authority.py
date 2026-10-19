"""Inspect saved stopping frames; no simulation steps or optimization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0,str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

output = HERE / 'feedback1024_stop_authority_audit001.json'
if output.exists():
    raise SystemExit('Preserve the existing read-only diagnostic')
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checkpoint = RUN / 'pilot019/model_001024.pt'
state = torch.load(checkpoint,map_location='cpu',weights_only=True)
actor = mean_from_checkpoint(state,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx').eval()
rows = {}
for name in ('feedback1024_envelope001','rmcs1024_envelope001','rmcs_default1024_envelope001'):
    path = RUN / name / 'traces.npz'
    z = np.load(path)
    time = z['episode_ticks'] * .02
    selected = (z['groups']=='start_stop_3')[None,:] & z['active']
    cases = {}
    for label,start,end in (('first_stop_8_16s',8.,16.),('second_stop_24_32s',24.,32.)):
        windows = {}
        for window,lo,hi in (('first_second',start,start+1),('middle_1_3s',start+1,start+3),('late_3_8s',start+3,end)):
            mask = selected & (time > lo) & (time <= hi)
            if not mask.any():
                raise ValueError('No frames in requested stopping window')
            obs = torch.from_numpy(z['observations'][mask])
            with torch.inference_mode():
                features = (obs[:,:28]-actor.input_center)*actor.input_scale
                low = torch.tanh(actor.feedback(features))
                fast = torch.tanh(actor.fast_feedback(features))
                mix = actor.fast_mix(obs)
                gate = actor.tracking_feedback_gate(obs)
                correction = actor.feedback_actions(obs)
                base = actor.command.base(obs)
                command = actor.command.residual_actions(obs)
                predicted = actor(obs)
            actual = z['actions'][mask]
            difference = np.max(np.abs(predicted.numpy()-actual))
            if not np.allclose(predicted.numpy(),actual,atol=2e-5,rtol=2e-5):
                raise ValueError('Trace actions differ from the recorded1024 mean')
            windows[window] = dict(frames=int(mask.sum()),
                root_vx_mean_m_s=float(z['velocity'][:,:,0][mask].mean()),
                root_planar_speed_mean_m_s=float(np.linalg.norm(z['velocity'][:,:,:2][mask],axis=-1).mean()),
                command_vx_min_max_m_s=[float(z['commands'][:,:,0][mask].min()),float(z['commands'][:,:,0][mask].max())],
                low_bound95_fraction_per_axis=(low.abs()>=.95).float().mean(0).tolist(),
                fast_bound95_fraction_per_axis=(fast.abs()>=.95).float().mean(0).tolist(),
                low_abs_tanh_mean_per_axis=low.abs().mean(0).tolist(),
                fast_mix_mean=float(mix.mean()),gate_mean=float(gate.mean()),
                base_action_mean_per_axis=base.mean(0).tolist(),
                frozen_command_offset_mean_per_axis=command.mean(0).tolist(),
                feedback_offset_mean_per_axis=correction.mean(0).tolist(),
                predicted_trace_action_max_abs_error=float(difference))
        cases[label]=windows
    evaluation = RUN/name/'evaluation.json'
    rows[name] = dict(evaluation_sha256=sha(evaluation),traces_sha256=sha(path),stop_windows=cases)
report = dict(schema='flat-feedback1024-readonly-stopping-authority-audit-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=sha(checkpoint),rows=rows,
    scope='Saved active development replay frames only. Original full-window criteria and all previous grades retained. '
        'Bound usage is a local diagnostic, not proof of causal actuator limitation.',
    native_environment_steps=0,optimizer_updates_performed=0,used_to_fit_weights=False,behavior_qualified=False)
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({n:{k:{w:dict(speed_m_s=x['root_planar_speed_mean_m_s'],
    low_bound95=x['low_bound95_fraction_per_axis'],mix=x['fast_mix_mean']) for w,x in v.items()}
    for k,v in r['stop_windows'].items()} for n,r in rows.items()},indent=2))
