"""Apply predeclared fixed behavior gates; no training or adoption side effects."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());folder=Path(plan['supervisor_output']);progress=json.loads((folder/'progress.json').read_text());assert progress['status']=='review_ready';paired=json.loads((folder/'paired.json').read_text());standing=json.loads((folder/'standing.json').read_text());groups=paired['groups'];reports={key:json.loads((Path(plan[path])/'evaluation/report.json').read_text()) for key,path in [('before','baseline_evaluation'),('after','evaluation_output')]}
def counts(rep):
 out={}
 for group,test in [('entry',lambda c:c.startswith('real_start')),('fallen',lambda c:c.startswith(('left_symmetric','right_symmetric','inverted_symmetric')))]:
  lanes=[r for r in rep['lanes'] if test(r['behavior_case'])]
  out[group]=dict(episodes=len(lanes),survived=sum(r['survived'] for r in lanes),recovered=sum(r['recovered'] for r in lanes),followup=sum(r['followup_success'] for r in lanes))
 out['impacts']=sum(any('impact' in c for c in r['termination_causes']) for r in rep['lanes'])
 out['passive_motor_peak_nm']=rep['passive_motor_peak_nm'];out['max_gap_m']=rep['max_gap_m']
 return out
before,after=[counts(reports[k]) for k in ('before','after')];gate=plan['review_gate']
checks=dict(standing_at_least19=standing['pose_and_position_pass']>=19,motion_MAE=groups['motion']['after']['vx_mae_m_s']['mean']<=gate['motion_MAE_at_most'],entry_peak=groups['entry']['after']['max_leg_speed_rad_s']['max']<=gate['entry_peak_at_most_rad_s'],fallen_peak=groups['fallen']['after']['max_leg_speed_rad_s']['max']<=gate['fallen_peak_at_most_rad_s'],fixed_impact_at_most2=after['impacts']<=2,
 improvement_in_recovery_or_followup=any(after[g][k]>before[g][k] for g in ('entry','fallen') for k in ('recovered','followup')),closure_3mm=after['max_gap_m']<=.003,zero_passive_effort=after['passive_motor_peak_nm']==0.)
receipt=dict(schema='entry-pilot-predeclared-fixed-gate-v1',plan=str(a.plan),checks=checks,before=before,after=after,standing_pass=standing['pose_and_position_pass'],motion_MAE=groups['motion']['after']['vx_mae_m_s']['mean'],entry_peak_rad_s=groups['entry']['after']['max_leg_speed_rad_s']['max'],fallen_peak_rad_s=groups['fallen']['after']['max_leg_speed_rad_s']['max'],decision='eligible_for_additional_zero_PPO_behavior_validation' if all(checks.values()) else 'reject_adoption_and_extension',paired_initial_exact=paired['initial_exact'],native_pairing_exact=paired['sensor_actuator_material_exact'] and paired['rigid_parameters_exact'],new_PPO=progress['new_optimizer_updates'],actual_experiment_consumed=progress['actual_experiment_consumed'],qualification=None,hardware_started=False,created_unix=time.time())
a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
