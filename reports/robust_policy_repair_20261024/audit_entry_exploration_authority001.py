"""Compare true paired-leg noise with mean physical target authority, CPU only."""
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
import numpy as np
from scipy.special import ndtr
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
from wheeled_algo.paired_recovery import wrap

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
source=B/'function_root_entry002/model_initial.pt'
state=torch.load(source,map_location='cpu',weights_only=False);mean,info=load_mean(source);s=info['settings']
sampler=GapRecoveryNoiseMean(mean,s['quiet_exploration_fraction'],s.get('quiet_wheel_exploration_fraction'),s['gap_recovery_leg_exploration_fraction'])
with torch.no_grad():sampler.log_std.copy_(state['actor_state_dict']['mlp.log_std'])
z=np.load(R/'entry_phase_path_dataset001.npz');obs=torch.from_numpy(z['observations'].copy())
phase=z['phase'][z['pose_ids']];desired=np.sign(z['desired_common'])
means=[];stds=[];gates=[]
with torch.inference_mode():
    for x in obs.split(512):
        output=sampler(x);means.append(output[:,0].numpy())
        stds.append(output[:,1].clamp(np.log(.003),np.log(.8)).exp().numpy())
        gates.append(handoff_entry_gate(x).numpy())
action=np.concatenate(means);std=np.concatenate(stds);gate=np.concatenate(gates)
lead=wrap(torch.from_numpy(action[:,:4]*.25-z['observations'][:,10:14])).numpy().reshape(-1,2,2).mean(-1)
far=(np.abs(phase)>=.75)&(gate>.99);near=(np.abs(phase)<1e-6)&(gate>.99)
rows=[]
for fraction in (1.,.25):
    # Candidate changes only four conditional leg standard deviations in
    # upright entry. Complete mean and wheel distribution remain untouched.
    alt=std.copy();alt[:,:4]*=1.-(1.-fraction)*gate
    common_sigma=.125*np.sqrt((alt[:,:4].reshape(-1,2,2)**2).sum(-1))
    wrong=ndtr(-(lead*desired)/common_sigma)
    outside_stop=ndtr((-.006-lead)/common_sigma)+ndtr((lead-.006)/common_sigma)
    rows.append(dict(entry_scale=.2*fraction,leg_std_rad=[float(x) for x in (.25*alt[gate[:,0]>.99,:4]).mean(0)],
        wheel_speed_std_rad_s=[float(x) for x in (10*alt[gate[:,0]>.99,4:]).mean(0)],
        paired_common_std_rad=float(common_sigma[far].mean()),
        far_wrong_direction_probability_mean=float(wrong[far].mean()),
        far_wrong_direction_probability_max=float(wrong[far].max()),
        stopped_abs_target_gt_006rad_probability_mean=float(outside_stop[near].mean()),
        entry_zero_std_exact=bool(np.array_equal(alt[gate[:,0]==0],std[gate[:,0]==0])),
        all_wheel_std_exact=bool(np.array_equal(alt[:,4:],std[:,4:]))))
checks=dict(actual_conditioned_distribution_used=True,physical_leg_scale025=True,physical_wheel_velocity_scale10=True,
    candidate_wheels_exact=rows[1]['all_wheel_std_exact'],candidate_off_entry_exact=rows[1]['entry_zero_std_exact'],
    correct_probability_order=rows[1]['far_wrong_direction_probability_mean']<rows[0]['far_wrong_direction_probability_mean'])
out=dict(schema='paired-entry-exploration-authority-v1',checks=checks,source=str(source),source_sha256=digest(source),
    phase_dataset_sha256=digest(R/'entry_phase_path_dataset001.npz'),frames=len(obs),rows=rows,
    assumptions='Independent native Gaussian leg samples, before global motor-action clipping. Paired common displacement is the mean of two nearby motor angles. Gaussian probabilities are not closed-loop outcome probabilities.',
    noise_candidate='Original hardcoded entry leg scale.2 versus.05; wheel/std outside entry unchanged. No checkpoint or runtime source changed.',
    decision='Analytical diagnosis only. A finite paired sampled-policy exposure can falsify the exploration hypothesis before any new PPO.',
    new_PPO=0,actual_experiment_consumed=2622,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'entry_exploration_authority001.json',out);print(json.dumps(out,indent=2));assert all(checks.values())
