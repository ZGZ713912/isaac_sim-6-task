"""Verify unchanged-v16 full resume through the actual1217 midpoint only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/flat_base_20261019'
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.separated_feedback import KIND

parser=argparse.ArgumentParser(description=__doc__)
args=parser.parse_args()
torch.set_num_threads(1)
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=load(RUN/'pilot022/model_001184.pt')
trial=RUN/'pilot023'
initial=load(trial/'initial_state.pt')
info=initial['infos']
contract=json.loads((HERE/'continuation1248_contract001.json').read_text())


def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


checks={f'initial_{key}_exact_parent':exact(initial[key],parent[key])
    for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict','torch_rng_state','cuda_rng_state')}
# The saved environment RNG is consumed by the declared fresh physical reset;
# its exact state before that reset is checked inside the native trainer.
checks.update(native_six_restore_checks=all(info['full_restore_verification'].values()),
    no_rate_change=info['optimizer_after_declared_rate_exact'] and not info['optimizer_rate_contract']['declared_actor_rate_change'],
    same_plan_source_parent=info['plan_sha256']==contract['plan_sha256']==parent['infos']['plan_sha256']
        and info['source_sha256']==contract['source_sha256']==parent['infos']['source_sha256']
        and info['parent_checkpoint_sha256']==contract['parent_checkpoint_sha256'],
    same_kind_no_identity_fork=info['actor_adaptation']==KIND and info['restore_verification_reference']=='parent'
        and not info.get('separated_feedback_identity_extension'),
    no_critic_warmup=info['critic_warmup_remaining']==0,
    parent_counters=(info['successful_updates_total'],info['actor_updates_total'],info['active_branch_actor_updates'],
        info['feedback_branch_actor_updates'],info['post_split_actor_updates'],info['inherited_shared_feedback_updates'],
        info['consumed_optimizer_updates_total'])==(1184,1004,44,626,322,304,1185),
    actual_selective_teacher_unchanged=info['teacher_anchor_runtime']==parent['infos']['teacher_anchor_runtime'])
timing=info['startup']['runtime_preflight']['observed']
checks['actual_native_clock_unchanged1000Hz']=all(timing[k]==v for k,v in dict(physics_dt=.001,policy_dt=.02,pc_control_dt=.001,decimation=20).items())
checks['exact_parent_control_and_command_reference']=info['control_math_sha256']==parent['infos']['control_math_sha256']=='cc30b2b0572999224c8feb79ffa7afb3df1b8a28800bca4a153663b9ea8d4bf8' and info['training_command_reference']==parent['infos']['training_command_reference']
mean=mean_from_checkpoint(initial,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx')
names=['distribution.std_param']+['mlp.'+n for n,_ in mean.named_parameters()]
ids=dict(zip(names,initial['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
prefixes=('mlp.feedback.','mlp.fast_feedback.')
learned=[n for n in names if n.startswith(prefixes)]
frozen=[n for n in names if not n.startswith(prefixes)]
slots0=initial['optimizer_state_dict']['state']
required=(1185,1217)
milestones={}
for step in required:
    path=trial/f'model_{step:06d}.pt'
    if not path.exists():continue
    state=load(path);row=state['infos'];slots=state['optimizer_state_dict']['state'];delta=step-1184
    proof=dict(frozen_actor_std_buffers_exact=all(torch.equal(v,state['actor_state_dict'][n])
        for n,v in initial['actor_state_dict'].items() if not n.startswith(prefixes)),
        frozen_Adam_slots_exact=all(exact(slots0.get(ids[n]),slots.get(ids[n])) for n in frozen),
        eight_feedback_tensors_learned=len(learned)==8 and all(not torch.equal(initial['actor_state_dict'][n],state['actor_state_dict'][n]) for n in learned),
        exactly2252_trainable_feedback_parameters=row['actor_trainability']['trainable_parameters']==2252
            and set(row['actor_trainability']['trainable_names'])==set(learned),
        feedback_critic_Adam_steps_exact=all(float(slots[i]['step'])==float(slots0[i]['step'])+12*delta
            for i in [ids[n] for n in learned]+initial['optimizer_state_dict']['param_groups'][1]['params']),
        critic_learned=any(not torch.equal(v,state['critic_state_dict'][n]) for n,v in initial['critic_state_dict'].items()),
        all_counters_without_refund=(row['successful_updates_total'],row['actor_updates_total'],row['active_branch_actor_updates'],
            row['feedback_branch_actor_updates'],row['post_split_actor_updates'],row['inherited_shared_feedback_updates'],
            row['consumed_optimizer_updates_total'],row['critic_warmup_remaining'])==(step,1004+delta,44,626+delta,322+delta,304,step+1,0),
        exact_group_rates=[g['lr'] for g in state['optimizer_state_dict']['param_groups']]==[3e-4,5e-4],
        actual_rate_logs=math.isclose(row['last_losses']['actor_learning_rate'],3e-4,rel_tol=1e-12)
            and math.isclose(row['last_losses']['critic_learning_rate'],5e-4,rel_tol=1e-12),
        all_losses_finite=all(np.isfinite(v) for v in row['last_losses'].values()))
    checks.update({str(step)+'_'+k:v for k,v in proof.items()})
    milestones[str(step)]=dict(checkpoint_sha256=sha(path),checks=proof,last_losses=row['last_losses'])
receipt=dict(schema='flat-feedback1248-native-full-resume-mid1217-v1',passed=all(checks.values()),checks=checks,
    all_required_milestones_present=len(milestones)==len(required),milestones=milestones,
    first_only=False,verification_scope='Midpoint only; final1248 remains unassessed',optimizer_updates_performed_by_verifier=0,behavior_qualified=False)
path=HERE/'feedback1248_native1217.json'
if path.exists():raise SystemExit('Preserve the completed midpoint verification')
path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=receipt['passed'],checks=len(checks),milestones=list(milestones),
    all_required_milestones_present=receipt['all_required_milestones_present'],failures=[k for k,v in checks.items() if not v])))
if not receipt['passed'] or not receipt['all_required_milestones_present']:raise SystemExit(1)
