"""Load the real896 actor, value network and full Adam with no architecture change."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import torch
import numpy as np
from tensordict import TensorDict
from rsl_rl.models import MLPModel
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.separated_feedback import KIND,configure_separated_feedback_training

torch.set_num_threads(1)
p=ROOT/'runs/flat_base_20261019/pilot017/model_000896.pt'
parent=torch.load(p,map_location='cpu',weights_only=True);original=deepcopy(parent)
generator=torch.Generator().manual_seed(191029)
x=torch.randn(64,35,generator=generator);x[:,0]*=3;x[:,2]*=4;x[:,3]=1.525
x[:16,:3]=0.;x[16:32,2]=0.;x[16:32,0]=.5
obs=TensorDict(dict(policy=x,critic=torch.randn(64,83,generator=generator)),batch_size=[64]);groups=dict(actor=['policy'],critic=['critic'])
mean=mean_from_checkpoint(parent,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx')
actor=MLPModel(obs,groups,'actor',6,hidden_dims=[256,128,64],activation='elu',obs_normalization=False,
 distribution_cfg=dict(class_name='GaussianDistribution',init_std=.04,std_type='scalar'))
actor.mlp=mean
critic=MLPModel(obs,groups,'critic',1,hidden_dims=[256,128,64],activation='elu',obs_normalization=False)
opt=torch.optim.Adam([dict(params=actor.parameters(),lr=3e-4,name='actor'),dict(params=critic.parameters(),lr=5e-4,name='critic')])

def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b

rng=torch.get_rng_state().clone()
actor.load_state_dict(parent['actor_state_dict'],strict=True)
critic.load_state_dict(parent['critic_state_dict'],strict=True)
opt.load_state_dict(parent['optimizer_state_dict'])
scope=configure_separated_feedback_training(actor,True)
trace=ROOT/'runs/flat_base_20261019/feedback896_basic001/traces.npz'
z=np.load(trace);mask=z['active'][::10]
frames=z['observations'][::10][mask];recorded=z['actions'][::10][mask]
with torch.no_grad():actual=actor.mlp(torch.from_numpy(frames)).numpy()
error=float(np.abs(actual-recorded).max())
checks=dict(real896_separated_kind=parent['infos']['actor_adaptation']==KIND,
 parent_object_unchanged=exact(parent,original),actor_all_tensors_buffers_std_exact=exact(actor.state_dict(),parent['actor_state_dict']),
 critic_all_tensors_exact=exact(critic.state_dict(),parent['critic_state_dict']),all_Adam_groups_slots_steps_exact=exact(opt.state_dict(),parent['optimizer_state_dict']),
 load_preserves_current_cpu_rng=torch.equal(rng,torch.get_rng_state()),recorded_native_actions_match=bool(np.isclose(actual,recorded,atol=2e-5,rtol=2e-5).all()),
 both_branches_only2252_trainable=scope['trainable_parameters']==2252 and len(scope['trainable_names'])==8 and all(n.startswith(('mlp.feedback.','mlp.fast_feedback.')) for n in scope['trainable_names']),
 frozen_gradients_none=all(v.grad is None for v in actor.parameters()),all_counters_unchanged=(parent['infos']['successful_updates_total'],parent['infos']['actor_updates_total'],parent['infos']['feedback_branch_actor_updates'],parent['infos']['post_split_actor_updates'],parent['infos']['consumed_optimizer_updates_total'])==(896,746,368,64,897),
 rates_unchanged=[g['lr'] for g in opt.param_groups]==[3e-4,5e-4])
receipt=dict(schema='flat-same-v13-real896-cpu-load-v1',recorded_at=datetime.now(timezone.utc).isoformat(),checks=checks,passed=all(checks.values()),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),synthetic_sample_frames=64,recorded_native_action_samples=len(frames),native_recorded_actions_max_abs_error=error,trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),used_for_optimization=False,checkpoint_files_written=0,cpu_optimizer_updates=0,native_environment_steps=0,behavior_qualified=False)
Path(__file__).with_name('same_v13_real896_resume_cpu002.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert receipt['passed']
