"""Read fixed max-height traces; distinguish command tracking, encoder agreement and output bounds."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
torch.set_num_threads(2)
RUN=ROOT/'runs/flat_base_20261019'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=HERE/'feedback1248_maximum_height_readout001.json'
if out.exists():raise SystemExit('Preserve the completed readout')
checkpoint=RUN/'pilot023/model_001248.pt'
model=mean_from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=True),
                          ROOT/'models/wheel_leg_flat_base_20261018/source.onnx').eval()
checks={};cases={};bound={}
for cohort in ('rmcs_maximum_source_envelope001','rmcs_maximum1248_envelope001','rmcs_default1248_envelope001'):
    path=RUN/cohort/'traces.npz';z=np.load(path);bound[cohort]=sha(path)
    for name in ('backward_05','backward_2','backward_3','forward_3'):
        lanes=np.flatnonzero(z['groups']==name)
        selected=z['active'][:,lanes] & (z['episode_ticks'][:,lanes]>=150)
        obs=torch.from_numpy(z['observations'][:,lanes][selected])
        velocity=z['velocity'][:,lanes][selected,0]
        encoder=.3*(obs[:,20]-obs[:,21]).numpy()
        entry=dict(post3s_frames=len(obs),mean_body_vx_m_s=float(velocity.mean()),
            mean_encoder_vx_m_s=float(encoder.mean()),
            mean_abs_encoder_body_difference_m_s=float(np.abs(encoder-velocity).mean()),
            mean_raw_wheel_actions=z['actions'][:,lanes][selected,4:6].mean(0).tolist(),
            mean_gravity=z['gravity'][:,lanes][selected].mean(0).tolist())
        if cohort!='rmcs_maximum_source_envelope001':
            with torch.inference_mode():
                actual=model(obs);saved=torch.from_numpy(z['actions'][:,lanes][selected])
                base=model.command.base(obs);command=model.command.residual_actions(obs)
                feedback=model.feedback_actions(obs)
                fast=torch.tanh(model.fast_feedback((obs[:,:28]-model.input_center)*model.input_scale))
                gate=model.tracking_feedback_gate(obs)
            error=float((actual-saved).abs().max())
            checks[cohort+'/'+name+'/CPU_matches_native_actions']=error<5e-6
            entry.update(max_CPU_native_action_error=error,
                mean_original_wheel_actions=base[:,4:6].mean(0).tolist(),
                mean_command_residual_wheel_actions=command[:,4:6].mean(0).tolist(),
                mean_feedback_wheel_actions=feedback[:,4:6].mean(0).tolist(),
                mean_tracking_gate=float(gate.mean()),
                fast_MLP_tanh_max_abs_per_axis=fast.abs().amax(0).tolist(),
                fast_MLP_fraction_abs_at_least95percent=float((fast.abs()>=.95).float().mean()))
        cases[cohort+'/'+name]=entry
if not all(checks.values()):raise SystemExit('Readout does not reproduce native actions')
out.write_text(json.dumps(dict(schema='flat-maximum-height-tracking-readout-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    reader_sha256=sha(Path(__file__)),checkpoint_sha256=sha(checkpoint),traces_sha256=bound,cases=cases,
    scope='Read-only local body/encoder velocity and same-input actor decomposition; not a closed-loop causal intervention or capacity impossibility proof',
    new_native_steps=0,new_optimizer_updates=0,behavior_qualified=False),indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'output':str(out)},indent=2))
