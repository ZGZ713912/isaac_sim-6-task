"""Zero-PPO sign intervention of four low-stance wheel-position weights."""
from copy import deepcopy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import numpy as np
from export_robust_getup import load_mean
from train_robust_getup import write_json,digest
from wheeled_algo.robust_getup import quiet_gate
source=ROOT/'runs/robust_policy_repair_20261024/high_common0125_probe001/model_probe.pt'
output=ROOT/'runs/robust_policy_repair_20261024/low_wheel_feedback_reverse_probe001/model_probe.pt'
if output.exists():raise ValueError('Never overwrite a probe')
old,info=load_mean(source)
original=torch.load(source,map_location='cpu',weights_only=False)
state=deepcopy(original)
key='mlp.mean.quiet_velocity.low.weight'
assert state['actor_state_dict'][key].shape==(4,2)
state['actor_state_dict'][key][2:4].neg_()
state['infos']['low_wheel_position_gain_probe']=dict(
 source=str(source),sha256=digest(source),factor=-1.,changed_parameter_count=4,
 channels=['left_wheel','right_wheel'],changed_tensor=key,
 hypothesis='Test restoring rather than reinforcing each signed rolling displacement. Current sign is a hypothesis, not a proven unique cause.',
 optimizer='Unchanged moments; this is a manual weight intervention, not an equivalent coordinate transformation',new_PPO=0)
state['infos']['hardware_ready']=False
output.parent.mkdir(parents=True,exist_ok=False)
torch.save(state,output)
new,_=load_mean(output)
frames=np.load(ROOT/'reports/robust_policy_repair_20261024/knee_step_source001/v7_filter005_replay.npz')
obs=torch.tensor(frames['observations'])
with torch.no_grad():
 a=old(obs);b=new(obs)
 outside=(obs[:,3] > 1.525) | (quiet_gate(obs).squeeze(-1)==0)
 checks=dict(all_leg_targets_exact=torch.equal(a[:,:4],b[:,:4]),
             outside_low_expert_exact=torch.equal(a[outside],b[outside]),
             unrelated_actor_tensors_exact=all(torch.equal(v,state['actor_state_dict'][k]) for k,v in original['actor_state_dict'].items() if k!=key),
             common_leg_position_weights_exact=torch.equal(original['actor_state_dict'][key][:2],state['actor_state_dict'][key][:2]))
assert all(checks.values()),checks
write_json(output.with_suffix('.json'),dict(schema='low-wheel-feedback-sign-probe-v1',
 source_sha256=digest(source),checkpoint_sha256=digest(output),checks=checks,
 maximum_same_input_wheel_target_delta_rad_s=float((a[:,4:]-b[:,4:]).abs().max()*10.),
 learning_clock=1501,new_PPO=0,experiment_consumed=2555,
 critic_optimizer_rng_unchanged=True,qualification=None,hardware_ready=False))
print(output)
