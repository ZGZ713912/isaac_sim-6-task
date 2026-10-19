"""Describe existing replay actions; no optimization or native stepping."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(4)
run = ROOT / 'runs/flat_base_20261019'
checkpoint = run / 'pilot021/model_001120.pt'
evaluation = run / 'deployment1120_basic001/evaluation.json'
trace = evaluation.with_name('traces.npz')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data = json.loads(evaluation.read_text())
if data['status'] != 'evaluated' or data['checkpoint_sha256'] != sha(checkpoint):
    raise SystemExit('Read the completed1120 replay only')
actor = mean_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=True),
                             ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx').eval()
with np.load(trace) as values:
    selected = ((values['groups'][None, :] == 'backward_05') & values['active'].astype(bool)
                & (values['episode_ticks'] * .02 > 3.))
    observations = torch.from_numpy(values['observations'][selected].copy())
    recorded = torch.from_numpy(values['actions'][selected].copy())
    velocity = values['velocity'][selected][:, 0]
with torch.inference_mode():
    actions = actor(observations)
    if not len(actions) or not torch.allclose(actions, recorded, atol=2e-5, rtol=2e-5):
        raise SystemExit('CPU actor does not reproduce the recorded native actions')
    gate = actor.tracking_feedback_gate(observations)
    command = actor.command(observations)
    feedback = actor.feedback_actions(observations)
report = dict(schema='flat-feedback1120-backward-steady-action-readout-v1',
    checkpoint_sha256=sha(checkpoint), evaluation_sha256=sha(evaluation), traces_sha256=sha(trace),
    reader_sha256=sha(Path(__file__)), case='backward_05',
    window='active frames with episode_ticks*.02 >3s', sample_frames=len(actions),
    cpu_native_action_max_abs_error=float((actions-recorded).abs().max()),
    root_vx_signed_mean_m_s=float(velocity.mean()), feedback_gate_mean=float(gate.mean()),
    base_and_frozen_command_wheel_raw_mean=command[:, 4:].mean(0).tolist(),
    feedback_wheel_raw_mean=feedback[:, 4:].mean(0).tolist(),
    local_wheel_setpoint_forward_increment_m_s=float((.3*(feedback[:, 4]-feedback[:, 5])).mean()),
    scope='Same-input action contribution only; closed-loop causal effect remains unmeasured',
    native_environment_steps=0, optimizer_updates_performed=0, used_to_fit_weights=False,
    behavior_qualified=False)
output = Path(__file__).with_name('feedback1120_backward_action_readout001.json')
if output.exists():
    raise SystemExit('Preserve the completed action readout')
output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(report))
