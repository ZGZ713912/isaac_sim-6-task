"""Check a masked teacher objective on synthetic inputs without optimizer steps."""
from copy import deepcopy
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import configure_guarded_feedback_training
from wheeled_algo.onnx_actor_import import recover_actor

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan',type=Path,default=HERE/'retention_candidate_v12.json')
parser.add_argument('--ppo-source',type=Path)
parser.add_argument('--visible-gate-cases',action='store_true')
parser.add_argument('--output',type=Path,default=HERE/'retention_candidate_v12_cpu001.json')
args=parser.parse_args()
if args.ppo_source:
    spec=importlib.util.spec_from_file_location('visible_retention_candidate',args.ppo_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    TeacherAnchoredPPO=module.TeacherAnchoredPPO
candidate=args.plan
plan=json.loads(candidate.read_text())
checkpoint=ROOT/'runs/flat_base_20261019/pilot015/model_000768.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
mean=mean_from_checkpoint(state,source)
original,_=recover_actor(source)
frame=torch.zeros(8,35)
frame[:,3]=1.525;frame[:,9]=-1.;frame[:,28]=1.
frame[:,0]=torch.tensor([0.,.5,2.,3.,-3.,0.,0.,1.])
frame[:,2]=torch.tensor([0.,0.,0.,0.,0.,4.,-4.,.6])
expected_selected=[False,False,False,True,True,True,True,False]
if args.visible_gate_cases:
    matched=frame[3:7].clone()
    vx,yaw=matched[:,0],matched[:,2]
    matched[:,20:22]=torch.stack((vx-.4373*yaw/2,-vx-.4373*yaw/2),-1)/.6
    near=matched[:1].expand(2,-1).clone()
    near[:,20]+=torch.tensor([.49/.6,.51/.6])
    frame=torch.cat((frame,matched,near))
    expected_selected+=[False]*4+[False,True]
count=len(frame)
obs=TensorDict(dict(policy=frame,critic=torch.zeros(count,83)),batch_size=[count])
groups=dict(actor=['policy'],critic=['critic'])
actor=MLPModel(obs,groups,'actor',6,hidden_dims=[256,128,64],activation='elu',obs_normalization=False,
    distribution_cfg=dict(class_name='GaussianDistribution',init_std=.04,std_type='scalar'))
actor.mlp=mean;actor.load_state_dict(state['actor_state_dict'],strict=True)
critic=MLPModel(obs,groups,'critic',1,hidden_dims=[256,128,64],activation='elu',obs_normalization=False)
critic.load_state_dict(state['critic_state_dict'],strict=True)
alg=TeacherAnchoredPPO(actor,critic,RolloutStorage('rl',count,4,obs,[6]),schedule='fixed',learning_rate=3e-4)
settings=plan['teacher_anchor']
extra={k:settings[k] for k in ('min_visible_wheel_error_m_s',) if k in settings}
alg.configure_teacher(original,coefficient=settings['coefficient'],action_scales=settings['action_scales'],
    min_vx=settings['min_vx'],min_yaw=settings['min_yaw'],**extra)
configure_guarded_feedback_training(actor,True,train_command=False)
before=deepcopy(actor.state_dict())
losses=[];selected=[]
for i in range(len(frame)):
    loss,_,fraction=alg.teacher_loss(TensorDict(dict(policy=frame[i:i+1]),batch_size=[1]))
    losses.append(float(loss.detach()));selected.append(bool(fraction))
loss,_,fraction=alg.teacher_loss(obs)
weighted=settings['coefficient']*loss
weighted.backward()
gradients={n:p.grad for n,p in actor.named_parameters() if p.grad is not None}
mean.verify_original(original)
checks=dict(high_command_mask_exact=selected==expected_selected,
    low_static_and_matched_teacher_exact_zero=all(losses[i]==0. for i,v in enumerate(expected_selected) if not v),
    selected_high_command_loss_positive=all(losses[i]>0. for i,v in enumerate(expected_selected) if v),
    only_four_feedback_tensors_receive_gradient=len(gradients)==4 and all(n.startswith('mlp.feedback.') for n in gradients),
    gradients_finite_and_nonzero=all(torch.isfinite(v).all() for v in gradients.values())
        and any(torch.count_nonzero(v) for v in gradients.values()),
    teacher_and_distribution_frozen=all(p.grad is None for p in alg.teacher.parameters())
        and all(p.grad is None for p in actor.distribution.parameters()),
    all_actor_parameters_unchanged=all(torch.equal(v,actor.state_dict()[n]) for n,v in before.items()),
    critic_unchanged=all(torch.equal(v,critic.state_dict()[n]) for n,v in state['critic_state_dict'].items()),
    weighted_loss_finite=torch.isfinite(weighted).item(),original_source_verified=True)
if args.visible_gate_cases:
    for i,bad in enumerate((0.,-.5,float('nan'),float('inf'),True)):
        rejected=False
        try:
            alg.configure_teacher(original,coefficient=settings['coefficient'],action_scales=settings['action_scales'],
                min_vx=settings['min_vx'],min_yaw=settings['min_yaw'],min_visible_wheel_error_m_s=bad)
        except ValueError:
            rejected=True
        checks['invalid_threshold_rejected_before_mutation_'+str(i)]=rejected and alg.teacher_min_visible_wheel_error_m_s==.5
checks={k:bool(v) for k,v in checks.items()}
receipt=dict(schema='flat-high-command-teacher-candidate-cpu-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    checks=checks,passed=all(checks.values()),selected=selected,loss_per_synthetic_frame=losses,
    raw_teacher_loss=float(loss.detach()),weighted_loss=float(weighted.detach()),selected_fraction=float(fraction),
    feedback_gradient_norm=float(torch.sqrt(sum(v.square().sum() for v in gradients.values()))),
    input_source=str(count)+' synthetic flat35 sensor frames; no evaluation traces or training rollouts',
    ppo_source_sha256=hashlib.sha256(args.ppo_source.read_bytes()).hexdigest() if args.ppo_source else None,
    adopted=False,optimizer_updates_performed=0,native_environment_steps=0,behavior_qualified=False)
args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=receipt['passed'],checks=len(checks),selected=selected,
    raw_teacher_loss=receipt['raw_teacher_loss'],weighted_loss=receipt['weighted_loss'],feedback_gradient_norm=receipt['feedback_gradient_norm'])))
if not receipt['passed']:
    raise SystemExit(1)
