"""Probe direct teacher-gradient coupling on synthetic frames, without adoption."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.onnx_actor_import import recover_actor

torch.set_num_threads(1)
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
checkpoint = ROOT / 'runs/flat_base_20261019/pilot015/model_000768.pt'
mean = mean_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=True), source)
teacher, _ = recover_actor(source)
mean.requires_grad_(False)
mean.feedback.requires_grad_(True)
x = torch.zeros(7, 35)
x[:, 3] = torch.tensor([1.15, 1.525, 2.15, 1.525, 1.525, 1.525, 1.525])
x[:, 9] = -1.; x[:, 28] = 1.
x[3:, 0] = torch.tensor([3., -3., 0., 0.])
x[3:, 2] = torch.tensor([0., 0., 4., -4.])
scales = torch.tensor([.025]*4 + [.04]*2)
with torch.no_grad():
    target = teacher(x[3:])
    before = mean(x).clone()
loss = .02 * .5 * ((mean(x[3:]) - target) / scales).square().mean()
mean.zero_grad(set_to_none=True)
loss.backward()
# A virtual SGD sensitivity step isolates the added loss, excluding Adam/other PPO terms.
# This object is never serialized as a policy or used by the native worker.
with torch.no_grad():
    for p in mean.feedback.parameters():
        p.add_(p.grad, alpha=-3e-4)
    after = mean(x)
static_delta = (after[:3] - before[:3]).abs().amax(-1).tolist()
report = dict(schema='flat-teacher-static-coupling-sensitivity-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),
    source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    virtual_step='one synthetic teacher-only SGD sensitivity step; no native Adam or PPO replay',
    static_action_max_abs_delta_by_height=static_delta,
    direct_shared_weight_coupling_present=all(v > 0 for v in static_delta),
    candidate_changes_adopted=False, checkpoint_files_written=0, native_optimizer_updates=0,
    native_environment_steps=0, evaluation_traces_used_for_optimization=False,
    causal_claim='demonstrates possible direct coupling; does not isolate the cause of observed832 drift',
    behavior_qualified=False)
path = Path(__file__).with_name('teacher_static_coupling_cpu001.json')
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
