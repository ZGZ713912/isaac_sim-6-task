"""Counterfactual: retain only the learned final common-row increment.

The two real PPO updates remain charged and the branch clock remains1503.
This is a documented weight composition, not an optimizer-equivalent migration.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from wheeled_algo.entry_checkpoint import entry_digest
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import exact_state_equal,periodic_canonical_entry_logits

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
reference=B/'motion_damping080002/model_probe.pt'
source=B/'pilot009_contract002/model_migrated.pt'
root,trained=[torch.load(p,map_location='cpu',weights_only=False) for p in (reference,source)]
assert root['infos']['completed_updates']==1501 and trained['infos']['completed_updates']==1503
result=deepcopy(trained)
result['actor_state_dict']=deepcopy(root['actor_state_dict'])
prefix='mlp.mean.entry_recovery.'
for suffix in ('weight','bias'):
    key=prefix+'4.'+suffix
    difference=trained['actor_state_dict'][key]-root['actor_state_dict'][key]
    for first in (0,2):
        common=.5*(difference[first]+difference[first+1])
        result['actor_state_dict'][key][first]+=common
        result['actor_state_dict'][key][first+1]+=common
result['infos']['function_root_learning']['entry_state_sha256']=entry_digest(result['actor_state_dict'])
result['infos']['learned_entry_common_projection']=dict(schema='post-PPO-common-row-composition-v1',
    learned_source=str(source.resolve()),learned_source_sha256=digest(source),
    fixed_basis_reference=str(reference.resolve()),fixed_basis_sha256=digest(reference),
    method='Restore the original feature basis and retain only the mean of each learned final left/right leg-row increment; retain original knee differences and wheels.',
    optimizer='Retain the sealed2-update Adam history for provenance; not a coordinate-equivalent continuation and no further PPO admitted.',
    actual_learning_clock=1503,new_PPO=0)
target=B/'pilot009_common_projection001/model_probe.pt'
target.parent.mkdir(parents=True,exist_ok=False);torch.save(result,target)
before,_=load_mean(reference);after,_=load_mean(target)
z=np.load(R/'entry_phase_path_dataset001.npz');frames=z['observations'];desired=z['desired_common'];phase=z['phase'][z['pose_ids']]
outs=[[],[]];raws=[[],[]];gates=[]
with torch.inference_mode():
    for values in np.array_split(frames,(len(frames)+511)//512):
        obs=torch.from_numpy(values.copy())
        frame=torch.cat((obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]),-1)
        gates.append(handoff_entry_gate(obs).numpy())
        for i,m in enumerate((before,after)):
            outs[i].append(m(obs).numpy());raws[i].append(periodic_canonical_entry_logits(m.entry_recovery,frame,m.feature_scale).numpy())
outs=[np.concatenate(v) for v in outs];raws=[np.concatenate(v) for v in raws];gate=np.concatenate(gates)
far=(np.abs(phase)>=.75)&(gate>.99);near=(np.abs(phase)<1e-6)&(gate>.99)
common=raws[1][:,:4].reshape(-1,2,2).mean(-1)
lead=wrap(torch.from_numpy(outs[1][:,:4]*.25-frames[:,10:14])).numpy().reshape(-1,2,2).mean(-1)
stats=dict(far_actual_signed_lead_min_rad=float((lead*np.sign(desired))[far].min()),
    far_actual_correct_fraction=float(((lead*np.sign(desired))[far]>.003).mean()),
    stopped_actual_target_peak_rad=float(np.abs(lead[near]).max()),
    knee_logit_change_peak=float(np.abs((raws[1][:,[1,3]]-raws[1][:,[0,2]])-(raws[0][:,[1,3]]-raws[0][:,[0,2]])).max()),
    physical_leg_change_peak_rad=float(wrap(torch.from_numpy((outs[1][:,:4]-outs[0][:,:4])*.25)).abs().max()))
outside=wheel_exact=True;count=0
with torch.inference_mode():
    for name in ('diagnosis012_final_arithmetic_fixed/evaluation','diagnosis012_final_arithmetic_rmcs/evaluation'):
        obs=np.load(B/name/'trajectory.npz')['observations'].reshape(-1,35)
        for values in np.array_split(obs,(len(obs)+511)//512):
            x=torch.from_numpy(values.copy());a,b=before(x),after(x);zero=handoff_entry_gate(x).ravel()==0.
            outside &= torch.equal(a[zero],b[zero]);wheel_exact &= torch.equal(a[:,4:],b[:,4:]);count+=len(x)
checks=dict(strict_existing_V3_decoder_accepts=True,
    old_critic_Adam_RNG_clock_exact=all(exact_state_equal(trained[k],result[k]) for k in trained if k not in ('actor_state_dict','infos')),
    only_last_two_entry_tensors_changed=all(torch.equal(v,result['actor_state_dict'][k]) for k,v in root['actor_state_dict'].items() if k not in (prefix+'4.weight',prefix+'4.bias')),
    original_far_direction_margin=stats['far_actual_signed_lead_min_rad']>=.014,
    original_stop_constraint=stats['stopped_actual_target_peak_rad']<=.006,
    original_knee_roundoff_bound=stats['knee_logit_change_peak']<=2e-5,
    wheels_exact=wheel_exact,outside_entry_exact=outside,
    bounded_change=stats['physical_leg_change_peak_rad']<=.003)
out=dict(schema='partial-PPO-common-projection-preflight-v1',checkpoint=str(target),sha256=digest(target),
    source=str(source),source_sha256=digest(source),reference=str(reference),reference_sha256=digest(reference),
    checks=checks,phase_frames=len(frames),recorded_frames=count,statistics=stats,
    hypothesis='Test whether the known stopped-phase constraint loss came from allowing the feature basis/knee/wheel outputs to change. Only the learned final common-row increment is retained.',
    native_gate=dict(standing_at_least=19,motion_MAE_at_most=.135,entry_recovery_at_least=8,
        entry_followup_at_least=8,fallen_recovery_at_least=4,fallen_followup_at_least=2,
        entry_leg_peak_at_most_rad_s=8.911178255081177,fallen_leg_peak_at_most_rad_s=24.11666153503418,
        impact_terminations_at_most=1,closure_at_most_m=.003,passive_motor_peak_nm=0.,
        improvement_required='entry_recovery>=9 OR fallen_recovery>=5 OR entry_followup>=9 OR fallen_followup>=3'),
    admission='native_only_if_all_checks_pass; no PPO',new_PPO=0,actual_experiment_consumed=2605,
    actual_learning_clock=1503,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'partial_PPO_common_projection_preflight001.json',out)
print(json.dumps(out,indent=2));assert all(checks.values()),checks
