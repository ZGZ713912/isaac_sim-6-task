"""Compare a declared policy-method diagnostic without erasing its override."""
import json,time,subprocess,sys
from pathlib import Path
sys.path[:0]=['src','scripts']
import numpy as np
from train_robust_getup import digest,write_json
from run_gap_long_training import behavior_metrics
from compare_gap_robustness import read,aggregate
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'diagnosis_plan035.json').read_text());queue=json.loads((R/'diagnosis_queue035.json').read_text())
assert queue['status']=='completed'
protocol=json.loads((R/'quiet_stance_protocol001.json').read_text());gate=protocol['gate']
run=Path(plan['jobs'][1]['output']);baseline=Path(protocol['baseline'])
assert all(digest(Path(n))==h for n,h in plan['source_sha256'].items())
a,ra,sa,ta=read(baseline);b,rb,sb,tb=read(run)
override=b['evaluation_overrides'];assert set(override)=={'quiet_gyro_stance_authority_probe'} and a['evaluation_overrides']=={}
m=override['quiet_gyro_stance_authority_probe']
assert m['schema']=='policy-quiet-gyro-stance-authority-probe-v2' and m['retained_quiet_authority']==.5
assert not m['observer_changed'] and not m['sampling_std_changed'] and m['gaussian_mean_changed']
assert m['eligible_for_checkpoint_mean_qualification'] is False
assert m['wrapper_sha256']==digest(R/'evaluate_quiet_stance_authority_probe001.py') and m['probe_sha256']==digest(R/'quiet_gyro_stance_authority_probe001.py')
checks=dict(declared_policy_override_exact=True,frozen_inputs=True,
    same_checkpoint=a['checkpoint_sha256']==b['checkpoint_sha256']==protocol['source_sha256'],
    native_eval_recipe=all(a.get(k)==b.get(k) for k in ['seed','seconds','severity','actuator_model','actuator_timing_profile','nominal_dynamics']),
    case_names=[x['name'] for x in sa['rows']]==[x['name'] for x in sb['rows']],
    sensor_actuator_material_exact=all(x['first_reset_domain']==y['first_reset_domain'] and x['material_coefficients']==y['material_coefficients'] for x,y in zip(ra['lanes'],rb['lanes'],strict=True)),
    initial_actor_inputs_exact=np.array_equal(ta['observations'][0],tb['observations'][0]),
    passive_mask_exact=np.array_equal(ta['passive'][0],tb['passive'][0]))
passive=ta['passive'][0]&tb['passive'][0]
for key in ('motor_position','motor_velocity'):checks['passive_poststep_'+key]=np.array_equal(ta[key][0,passive],tb[key][0,passive])
prea,preb=[np.load(x/'evaluation_precontrol_state.npz') for x in (baseline,run)]
for key in prea.files:checks['precontrol_'+key]=np.array_equal(prea[key],preb[key])
startup=[json.loads((x/'startup.json').read_text()) for x in (baseline,run)]
checks['rigid_parameters_exact']=startup[0]['dynamics_randomization']['sampled_properties_sha256']==startup[1]['dynamics_randomization']['sampled_properties_sha256']
ca,cb=[json.loads((x/'task_contract.json').read_text())['settings'] for x in (baseline,run)]
task_delta={k:dict(before=ca.get(k,'MISSING'),after=cb.get(k,'MISSING')) for k in ca.keys()|cb.keys() if ca.get(k)!=cb.get(k)}
expected_policy_fields={'gap_entry_encoder_version','motion_leg_damping_version','motion_leg_damping_alpha'}
checks['same_loaded_checkpoint_actor_settings']=a['actor_settings']==b['actor_settings']
checks['all_task_settings_exact']=task_delta=={}
checks['current_control_exactly_reproduces_history']=all(json.loads((R/'current_control035_replay001.json').read_text())['checks'].values())
checks['recovery_scope_explicit']=m['recovery_zero_authority_threshold']==.05
standing_path=R/'quiet_stance050_standing001.json'
with (R/'quiet_stance050_standing_analysis001.log').open('x') as stream:
 subprocess.run([sys.executable,'scripts/analyze_gap_standing.py','--run',str(run),'--output',str(standing_path)],check=True,stdout=stream,stderr=subprocess.STDOUT)
standing=json.loads(standing_path.read_text());metrics=behavior_metrics(run,standing_path)
impacts=sum(any('impact' in c for c in row['termination_causes']) for row in rb['lanes'])
capture=[]
for name in gate['capture_names']:
 lane=next(x['lane'] for x in rb['lanes'] if x['name']==name)
 mask=tb['alive'][:,lane]&(tb['time'][:,lane]>=6.)
 capture.append(dict(name=name,angular_RMS_rad_s=float(np.sqrt((tb['omega'][mask,lane,:2]**2).sum(-1).mean())) if mask.any() else None,survived=rb['lanes'][lane]['survived']))
mean=sum(x['angular_RMS_rad_s'] for x in capture)/len(capture) if all(x['angular_RMS_rad_s'] is not None for x in capture) else None
ratio=mean/gate['capture_angular_RMS_mean_baseline'] if mean is not None else None
checks.update(standing=standing['pose_and_position_pass']>=gate['standing_at_least'],
 motion=metrics['motion']['vx_mae_m_s']<=gate['motion_MAE_at_most'],
 entry_recovery=metrics['entry']['recovered']>=gate['entry_recovery_at_least'],entry_followup=metrics['entry']['followup_success']>=gate['entry_followup_at_least'],
 fallen_recovery=metrics['fallen']['recovered']>=gate['fallen_recovery_at_least'],fallen_followup=metrics['fallen']['followup_success']>=gate['fallen_followup_at_least'],
 entry_speed=metrics['entry']['max_leg_speed_rad_s']<=gate['entry_peak_at_most_rad_s'],fallen_speed=metrics['fallen']['max_leg_speed_rad_s']<=gate['fallen_peak_at_most_rad_s'],
 impacts=impacts<=gate['impacts_at_most'],closure=rb['max_gap_m']<=gate['closure_gap_at_most_m'],passive=rb['passive_motor_peak_nm']==0.,
 capture_oscillation_reduction=ratio is not None and ratio<=gate['capture_angular_RMS_ratio_at_most'],
 entry_followup_improvement=metrics['entry']['followup_success']>=gate['entry_followup_improvement_at_least'])
checks={k:bool(v) for k,v in checks.items()}
decision='eligible_for_separate_policy_implementation_review_only' if all(checks.values()) else 'reject_probe_adoption_and_PPO_admission'
out=dict(schema='policy-only-quiet-authority-diagnostic-decision-v1',checks=checks,metrics=metrics,standing_pass=standing['pose_and_position_pass'],impacts=impacts,
 capture=capture,capture_angular_RMS_mean_rad_s=mean,capture_angular_RMS_ratio=ratio,override=override,task_metadata_delta=task_delta,
 fresh_current_source_control_completed=True,
 interpretation='Explicit policy-method counterfactual; stored checkpoint mean unchanged. No actuator/sensor/observer intervention and no override normalization in original artifacts.',
 checkpoint=str(protocol['source']),checkpoint_sha256=protocol['source_sha256'],frozen_inputs=len(plan['source_sha256']),
 decision=decision,new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,
 qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'quiet_stance050_fixed_decision001.json',out)
p=json.loads((R/'progress.json').read_text());p.update(status='quiet_stance_zero_PPO_reviewed',active_diagnosis_queue=None,active_native_queue=None,
 active_job=None,latest_completed_evaluation=str(run),latest_research_decision=decision,new_PPO_admitted=0,updated_unix=time.time())
write_json(R/'progress.json',p);print(json.dumps(out,indent=2))
