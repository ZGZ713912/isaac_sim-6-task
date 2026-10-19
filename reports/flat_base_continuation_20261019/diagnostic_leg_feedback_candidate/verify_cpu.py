"""Prove the new output-row interventions on real1120 replay inputs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(4)
file = HERE / 'evaluate.py'
spec = importlib.util.spec_from_file_location('feedback_factorial', file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checkpoint = ROOT / 'runs/flat_base_20261019/pilot021/model_001120.pt'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
actor = mean_from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=True),
    ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx').eval()
trace = ROOT / 'runs/flat_base_20261019/deployment1120_basic001/traces.npz'
with np.load(trace) as data:
    obs = torch.from_numpy(data['observations'][::10].reshape(-1,35).copy())
means = {name:module.diagnostic_ablation(deepcopy(actor), name).eval()
    for name in ('full','leg_feedback_off','all_feedback_off')}
allowed = {f'{branch}.2.{name}' for branch in ('feedback','fast_feedback') for name in ('weight','bias')}
checks = {}
with torch.inference_mode():
    original,command = actor(obs),actor.command(obs)
    checks['full_actor_unchanged'] = torch.equal(original,means['full'](obs))
    for name,mean in means.items():
        if name == 'full':
            continue
        state,expected = mean.state_dict(),actor.state_dict()
        actions,feedback = mean(obs),mean.feedback_actions(obs)
        checks[name+'_other_tensors_unchanged'] = all(torch.equal(state[n],v) for n,v in expected.items() if n not in allowed)
        end = 4 if name == 'leg_feedback_off' else 6
        checks[name+'_selected_rows_zero'] = all(torch.count_nonzero(state[n][:end]).item()==0 for n in allowed)
        checks[name+'_remaining_rows_exact'] = all(torch.equal(state[n][end:],expected[n][end:]) for n in allowed)
        checks[name+'_selected_feedback_zero'] = torch.count_nonzero(feedback[:,:end]).item()==0
        checks[name+'_selected_actions_exact_command'] = torch.equal(actions[:,:end],command[:,:end])
    checks['leg_off_preserves_wheel_actions'] = torch.equal(means['leg_feedback_off'](obs)[:,4:],original[:,4:])
try:
    module.diagnostic_ablation(deepcopy(actor),'wheel_feedback_off')
except ValueError:
    checks['undeclared_intervention_rejected'] = True
else:
    checks['undeclared_intervention_rejected'] = False
report = dict(schema='flat1120-leg-feedback-factorial-cpu-v1',passed=all(checks.values()),checks=checks,
    observations=len(obs),checkpoint_sha256=sha(checkpoint),candidate_sha256=sha(file),
    reader_sha256=sha(Path(__file__)),traces_sha256=sha(trace),native_steps=0,optimizer_updates=0,
    scope='Same-input row isolation only; closed-loop results not inferred')
output = HERE / 'cpu001.json'
if output.exists():
    raise SystemExit('Preserve existing CPU proof')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if not report['passed']:
    raise SystemExit(1)
