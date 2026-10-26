"""Explicit zero-PPO output-bias interventions; preserve all source evidence."""
import sys,json,hashlib,copy,time
from pathlib import Path
sys.path[:0]=['src','scripts']
import torch,numpy as np
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import tilt_gate
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1)
base=Path('runs/robust_policy_repair_20261024');source=base/'final_pair_circular_corrected002/model_migrated.pt';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();model,info=load_mean(source);state=torch.load(source,map_location='cpu',weights_only=False);key='mlp.mean.recovery.4.bias'
trace=base/'diagnosis012_final_circular_fixed/evaluation/trajectory.npz';frames=np.load(trace)['observations'].reshape(-1,35)
for value,label in [(.15,'015'),(.30,'030')]:
 out=base/f'recovery_common_bias{label}_probe001';out.mkdir(exist_ok=False)
 child=copy.deepcopy(state);old=child['actor_state_dict'][key].clone();change=torch.tensor([-value,-value,value,value,0.,0.]);child['actor_state_dict'][key]+=change
 child['infos']['recovery_common_bias_probe']=dict(schema='recovery-common-bias-weight-probe-v1',source=str(source.resolve()),source_sha256=sha(source),bias_delta=change.tolist(),tensor=key,new_PPO=0,source_clock=info['completed_updates'],optimizer='Preserved source slots; intervention is not an equivalent coordinate transformation',hypothesis='Test whether insufficient learned common-angle drive accounts for upright body with both legs above body. Same original sensor-conditioned NN, no runtime trajectory, phase or FSM; existing error, target, torque and mechanical limits unchanged')
 cp=out/'model_probe.pt';torch.save(child,cp);probe,_=load_mean(cp)
 checks=dict(other_weights_exact=all(torch.equal(v,child['actor_state_dict'][k]) for k,v in state['actor_state_dict'].items() if k!=key),six_bias_values_delta=torch.allclose(child['actor_state_dict'][key]-old,change,rtol=0,atol=1e-8))
 changed=0;outside=0;wheels_exact=True;max_phys=0.
 with torch.inference_mode():
  for o in torch.tensor(frames).split(2048):
   a=model(o);b=probe(o);gate=torch.maximum(tilt_gate(o,model.stance_aware_recovery,model.recovery_shape_aware),handoff_entry_gate(o)).ravel();outside+=int(((a[gate==0]!=b[gate==0]).any(-1)).sum());changed+=int((a!=b).any(-1).sum());wheels_exact &= torch.equal(a[:,4:],b[:,4:]);diff=(a[:,:4]-b[:,:4])*.25;physical=torch.atan2(diff.sin(),diff.cos()).abs().max().item();max_phys=max(max_phys,physical)
 checks.update(all_wheel_outputs_exact=wheels_exact,outside_recovery_entry_gates_exact=outside==0)
 if not all(checks.values()):raise ValueError(checks)
 receipt=dict(schema='zero-PPO-recovery-common-bias-preflight-v1',checkpoint=str(cp),sha256=sha(cp),source_sha256=sha(source),bias_magnitude=value,checks=checks,frames=len(frames),changed_frames=changed,max_physical_leg_target_difference_rad=max_phys,trace_sha256=sha(trace),new_PPO=0,actual_experiment_consumed=2587,gate=dict(fixed_standing_at_least=19,motion_MAE_at_most=.135,fixed_fallen_recovery_at_least=3,fixed_fallen_followup_at_least=1,entry_leg_speed_at_most_rad_s=8.486836433410645*1.05,fallen_leg_speed_at_most_rad_s=24.11665153503418,fixed_impact_at_most=2,RMCS_recovery_at_least=55,RMCS_followup_at_least=36,RMCS_impact_at_most=2,RMCS_leg_speed_at_most_rad_s=18.907798767089844,no_automatic_PPO=True),qualification=None,hardware_started=False,created_unix=time.time())
 (out/'model_probe.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))
