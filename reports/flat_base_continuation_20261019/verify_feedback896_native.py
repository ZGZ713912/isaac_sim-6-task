"""Verify the real identity fork, cloned Adam, frozen state and GPU learning."""
import argparse
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.separated_feedback import KIND
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--first-only',action='store_true');args=parser.parse_args()
torch.set_num_threads(4)
RUN=ROOT/'runs/flat_base_20261019';TRIAL=RUN/'pilot017';HERE=Path(__file__).resolve().parent
load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=load(RUN/'pilot016/model_000832.pt');initial=load(TRIAL/'initial_state.pt');info=initial['infos']
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
old_mean=mean_from_checkpoint(parent,source);new_mean=mean_from_checkpoint(initial,source)
old_names=['distribution.std_param']+['mlp.'+n for n,_ in old_mean.named_parameters()]
new_names=['distribution.std_param']+['mlp.'+n for n,_ in new_mean.named_parameters()]
old_opt=parent['optimizer_state_dict'];expected_opt=deepcopy(old_opt)
expected_opt['param_groups'][0]['params']=list(range(len(new_names)))
critic_count=len(old_opt['param_groups'][1]['params'])
expected_opt['param_groups'][1]['params']=list(range(len(new_names),len(new_names)+critic_count))
old_ids=dict(zip(old_names,old_opt['param_groups'][0]['params'],strict=True))
new_ids=dict(zip(new_names,expected_opt['param_groups'][0]['params'],strict=True))
expected_opt['state']={}
for name,new_id in new_ids.items():
 old_id=old_ids[name.replace('mlp.fast_feedback.','mlp.feedback.',1)]
 if old_id in old_opt['state']:expected_opt['state'][new_id]=deepcopy(old_opt['state'][old_id])
for a,b in zip(old_opt['param_groups'][1]['params'],expected_opt['param_groups'][1]['params'],strict=True):
 if a in old_opt['state']:expected_opt['state'][b]=deepcopy(old_opt['state'][a])
expected_actor=deepcopy(parent['actor_state_dict'])
for n,v in parent['actor_state_dict'].items():
 if n.startswith('mlp.feedback.'):expected_actor[n.replace('mlp.feedback.','mlp.fast_feedback.',1)]=v.clone()
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
revision=json.loads((HERE/'training_plan_revision015.json').read_text())
checks=dict(initial_actor_exact_parent_plus_four_clones=exact(initial['actor_state_dict'],expected_actor),
 initial_critic_exact_parent=exact(initial['critic_state_dict'],parent['critic_state_dict']),
 initial_Adam_exact_all_named_parent_slots_and_fast_clones=exact(initial['optimizer_state_dict'],expected_opt),
 native_six_model_Adam_RNG_restore_checks=all(info['full_restore_verification'].values()),
 native_post_rate_exact=info['optimizer_after_declared_rate_exact'],
 no_rate_change=info['optimizer_rate_contract']['before']==dict(actor=3e-4,critic=5e-4) and info['optimizer_rate_contract']['after']==dict(actor=3e-4,critic=5e-4) and not info['optimizer_rate_contract']['declared_actor_rate_change'],
 parent_counters_retained=(info['successful_updates_total'],info['actor_updates_total'],info['active_branch_actor_updates'],info['feedback_branch_actor_updates'],info['consumed_optimizer_updates_total'])==(832,682,44,304,833),
 split_clock_starts_zero_without_erasing_inheritance=info['post_split_actor_updates']==0 and info['inherited_shared_feedback_updates']==304,
 no_additional_critic_warmup=info['critic_warmup_remaining']==0,
 exact_identity_extension_kind=info['actor_adaptation']==KIND and info['restore_verification_reference']=='declared_identity_extended_parent_with_feedback_and_Adam_clones',
 exact_declared_sources=info['source_sha256']==revision['new_source_sha256'],
 exact_declared_plan=info['plan_sha256']==revision['new_plan_sha256'],
 exact_declared_parent=info['parent_checkpoint_sha256']==revision['parent_checkpoint_sha256'],
 actual_selective_teacher_runtime=info['teacher_anchor_runtime']==dict(coefficient=.02,action_scales=torch.tensor([.025]*4+[.04]*2).tolist(),min_vx=2.5,min_yaw=3.,min_visible_wheel_error_m_s=.5))
base_names=[n for n in initial['actor_state_dict'] if n.startswith('mlp.command.base.')]
checks['original_eight_50758_parameters_present']=len(base_names)==8 and sum(initial['actor_state_dict'][n].numel() for n in base_names)==50758
frames=torch.from_numpy(np.load(RUN/'rmcs832_envelope001/traces.npz')['observations'][::5].reshape(-1,35))
with torch.no_grad():checks['initial_mean_bit_exact_parent_on_recorded_frames']=torch.equal(old_mean(frames),new_mean(frames))
frozen_ids=[i for n,i in new_ids.items() if not n.startswith(('mlp.feedback.','mlp.fast_feedback.'))]
feedback_ids=[i for n,i in new_ids.items() if n.startswith(('mlp.feedback.','mlp.fast_feedback.'))]
initial_slots=initial['optimizer_state_dict']['state'];milestones={};required=(833,) if args.first_only else (833,865,896)
for step in required:
 path=TRIAL/f'model_{step:06d}.pt'
 if not path.exists():continue
 state=load(path);row_info=state['infos'];slots=state['optimizer_state_dict']['state'];delta=step-832
 row=dict(checkpoint_sha256=sha(path),successful_updates_total=row_info['successful_updates_total'],actor_updates_total=row_info['actor_updates_total'],feedback_branch_actor_updates=row_info['feedback_branch_actor_updates'],post_split_actor_updates=row_info['post_split_actor_updates'])
 row['checks']=dict(frozen_actor_std_buffers_exact=all(torch.equal(v,initial['actor_state_dict'][n]) for n,v in state['actor_state_dict'].items() if not n.startswith(('mlp.feedback.','mlp.fast_feedback.'))),
  frozen_Adam_slots_exact=all(exact(slots.get(i),initial_slots.get(i)) for i in frozen_ids),
  all_eight_feedback_tensors_learned=sum(n.startswith(('mlp.feedback.','mlp.fast_feedback.')) for n in state['actor_state_dict'])==8 and all(not torch.equal(v,initial['actor_state_dict'][n]) for n,v in state['actor_state_dict'].items() if n.startswith(('mlp.feedback.','mlp.fast_feedback.'))),
  exactly2252_feedback_parameters_trainable=row_info['actor_trainability']['trainable_parameters']==2252 and all(n.startswith(('mlp.feedback.','mlp.fast_feedback.')) for n in row_info['actor_trainability']['trainable_names']),
  all_updates_actor_and_split_clock_exact=row_info['actor_updates_total']==682+delta and row_info['feedback_branch_actor_updates']==304+delta and row_info['post_split_actor_updates']==delta and row_info['critic_warmup_remaining']==0,
  command_and_inheritance_clocks_retained=row_info['active_branch_actor_updates']==44 and row_info['inherited_shared_feedback_updates']==304,
  consumed_without_refund=row_info['consumed_optimizer_updates_total']==step+1,
  both_feedback_and_critic_Adam_steps_retained=all(float(slots[i]['step'])==float(initial_slots[i]['step'])+12*delta for i in feedback_ids+expected_opt['param_groups'][1]['params']),
  actual_rates_unchanged=[g['lr'] for g in state['optimizer_state_dict']['param_groups']]==[3e-4,5e-4],
  actual_rate_logs=math.isclose(row_info['last_losses']['actor_learning_rate'],3e-4,rel_tol=1e-12) and math.isclose(row_info['last_losses']['critic_learning_rate'],5e-4,rel_tol=1e-12),
  all_losses_finite=all(np.isfinite(v) for v in row_info['last_losses'].values()),
  original_eight_tensors_exact=all(torch.equal(initial['actor_state_dict'][n],state['actor_state_dict'][n]) for n in base_names))
 row['last_losses']=row_info['last_losses'];milestones[str(step)]=row;checks.update({str(step)+'_'+k:v for k,v in row['checks'].items()})
receipt=dict(schema='flat-separated896-native-milestone-verification-v1',passed=all(checks.values()),checks=checks,milestones=milestones,all_required_milestones_present=len(milestones)==len(required),optimizer_updates_performed_by_verifier=0,qualification_eligible=False,first_only=args.first_only)
p=HERE/('feedback896_native833.json' if args.first_only else 'feedback896_native_milestones.json');p.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=receipt['passed'],checks=len(checks),milestones=list(milestones),all_required_milestones_present=receipt['all_required_milestones_present'],failures=[k for k,v in checks.items() if not v])))
if not receipt['passed'] or not receipt['all_required_milestones_present']:raise SystemExit(1)
