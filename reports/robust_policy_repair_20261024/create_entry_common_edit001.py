"""Authenticate a closed-form common-only last-layer NN initialization."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from migrate_gap_entry_preservation import migrate
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
fit_path = R/'entry_common_edit_feasibility001.json'
fit = json.loads(fit_path.read_text())
assert all(fit['checks'].values())
source = B/'final_pair_circular_corrected002/model_migrated.pt'
control = B/'periodic_entry_encoder001/model_probe.pt'
teacher = B/'entry_common_fit_teacher001/model_probe.pt'
split = B/'entry_common_fit_split001/model_migrated.pt'
candidate = B/'entry_common_fit_periodic001/model_probe.pt'
state = torch.load(source, map_location='cpu', weights_only=False)
changed = deepcopy(state)
v = torch.tensor(fit['edit_vector'], dtype=torch.float32)
direction = torch.tensor([1., 1., -1., -1., 0., 0.])
weight_key, bias_key = 'mlp.mean.recovery.4.weight', 'mlp.mean.recovery.4.bias'
changed['actor_state_dict'][weight_key] += direction[:,None]*v[None,:-1]
changed['actor_state_dict'][bias_key] += direction*v[-1]
changed['infos']['recovery_common_weight_edit'] = dict(schema='closed-form-entry-common-NN-edit-v1',
    source_checkpoint=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    fit=str(fit_path), fit_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
    tensors=[weight_key,bias_key], method='One predeclared constrained ridge solve; no gradient optimizer',
    independent_common_parameters=129, new_PPO=0,
    scope='Offline reference head for explicit entry-only copying; shared teacher never runs in native simulation')
teacher.parent.mkdir(parents=True, exist_ok=False)
torch.save(changed, teacher)
load_mean(teacher)
migrate(source, teacher, split, 2603)
subprocess.run([sys.executable,'scripts/migrate_periodic_entry_encoder.py','--source',str(split),
                '--output',str(candidate)],check=True)
old, oi = load_mean(control)
new, ni = load_mean(candidate)
allowed = {'entry_recovery.4.weight','entry_recovery.4.bias'}
exact_others = all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items() if k not in allowed)
assert exact_others
for suffix in ('weight','bias'):
    assert torch.equal(old.entry_recovery[-1].state_dict()[suffix][4:], new.entry_recovery[-1].state_dict()[suffix][4:])
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation',
             'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation'):
    path = B/name/'trajectory.npz'
    z = np.load(path)
    frames = z['observations'].reshape(-1,35)
    invariant = exact = 0
    finite = wheel_exact = True
    max_delta = knee_logit_delta = 0.
    for start in range(0,len(frames),2048):
        obs = torch.tensor(frames[start:start+2048])
        outside = handoff_entry_gate(obs)[:,0] == 0.
        frame = torch.cat((obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]),-1)
        with torch.inference_mode():
            a,b = old(obs),new(obs)
            ra = periodic_canonical_entry_logits(old.entry_recovery,frame,old.feature_scale)
            rb = periodic_canonical_entry_logits(new.entry_recovery,frame,new.feature_scale)
        invariant += int(outside.sum())
        exact += int((a[outside]==b[outside]).all(-1).sum())
        wheel_exact &= torch.equal(a[:,4:],b[:,4:])
        finite &= bool(torch.isfinite(b).all())
        d = (a[:,:4]-b[:,:4])*.25
        max_delta = max(max_delta,float(torch.atan2(d.sin(),d.cos()).abs().max()))
        knee_logit_delta = max(knee_logit_delta,float(((rb[:,[1,3]]-rb[:,[0,2]])-(ra[:,[1,3]]-ra[:,[0,2]])).abs().max()))
    rows.append(dict(trace=str(path),frames=len(frames),entry0_frames=invariant,entry0_frames_bit_exact=exact,
        all_wheel_outputs_bit_exact=wheel_exact,finite=finite,maximum_physical_target_delta_rad=max_delta,
        max_float32_knee_logit_change=knee_logit_delta))
checks = dict(only_entry_last_layer_changed=exact_others,
    wheel_last_layer_rows_exact=True,clocks_equal=oi['completed_updates']==ni['completed_updates']==1501,
    all_entry0_bit_exact=all(x['entry0_frames']==x['entry0_frames_bit_exact'] for x in rows),
    all_wheel_outputs_exact=all(x['all_wheel_outputs_bit_exact'] for x in rows),
    knee_logit_roundoff_under_1e_5=all(x['max_float32_knee_logit_change']<1e-5 for x in rows),
    finite=all(x['finite'] for x in rows))
out = dict(schema='entry-common-fit-checkpoint-preflight-v1',checkpoint=str(candidate),
    sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),reference=str(control),
    reference_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),checks=checks,replays=rows,
    fit=str(fit_path),fit_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
    method='Closed-form existing last-layer common-only edit, original frozen fallen/quiet/motion/std and knee/wheel mapping retained.',
    native_hypothesis='State-dependent drive can leave the observed wheels-up equilibrium while retaining successful entry actions.',
    new_PPO=0,gradient_optimizer_updates=0,actual_experiment_consumed=2603,
    qualification=None,hardware_started=False,created_unix=time.time())
(R/'entry_common_fit_preflight001.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='replays'},indent=2))
assert all(checks.values())
