"""Apply unchanged fixed physical gates after the eight-update projected trial."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
from run_gap_long_training import behavior_metrics
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'plan013_projected_entry001.json').read_text())
queue=json.loads((Path(plan['supervisor_output'])/'progress.json').read_text());assert queue['status']=='review_ready'
endpoint=json.loads((R/'projected_round013_endpoint001.json').read_text())
assert all(endpoint['checks'].values()) and endpoint['all_development_phase_checks_pass']
assert endpoint['real_PPO_completed']==8
run=Path(plan['evaluation_output']);sup=Path(plan['supervisor_output'])
assert json.loads((run/'progress.json').read_text())['status']=='completed'
standing=json.loads((sup/'standing.json').read_text());paired=json.loads((sup/'paired.json').read_text())
metrics=behavior_metrics(run,sup/'standing.json');gate=plan['review_gate']
report=json.loads((run/'evaluation/report.json').read_text())
impacts=sum(any('impact' in x for x in row['termination_causes']) for row in report['lanes'])
checks=dict(endpoint_audit=True,development_phase=True,
    frozen_inputs=all(digest(Path(n))==h for n,h in plan['source_sha256'].items()),
    standing=standing['pose_and_position_pass']>=gate['standing_at_least'],
    motion=metrics['motion']['vx_mae_m_s']<=gate['motion_MAE_at_most'],
    entry_recovery=metrics['entry']['recovered']>=gate['entry_recovery_at_least'],
    entry_followup=metrics['entry']['followup_success']>=gate['entry_followup_at_least'],
    fallen_recovery=metrics['fallen']['recovered']>=gate['fallen_recovery_at_least'],
    fallen_followup=metrics['fallen']['followup_success']>=gate['fallen_followup_at_least'],
    entry_speed=metrics['entry']['max_leg_speed_rad_s']<=gate['entry_peak_at_most_rad_s'],
    fallen_speed=metrics['fallen']['max_leg_speed_rad_s']<=gate['fallen_peak_at_most_rad_s'],
    impacts=impacts<=gate['impacts_at_most'],closure=report['max_gap_m']<=.003,
    passive=report['passive_motor_peak_nm']==0.,
    paired_initial=all(paired['initial_exact'].values()) and paired['sensor_actuator_material_exact'] and paired['rigid_parameters_exact'],
    measured_improvement=metrics['entry']['followup_success']>=9 or metrics['fallen']['recovered']>=5 or metrics['fallen']['followup_success']>=3)
checks={k:bool(v) for k,v in checks.items()};decision='admit_zero_PPO_expanded_review' if all(checks.values()) else 'reject_extension_and_adoption'
out=dict(schema='bounded-projected-Adam-fixed-review-v1',checks=checks,metrics=metrics,
    standing_pass=standing['pose_and_position_pass'],impacts=impacts,
    checkpoint=endpoint['checkpoint'],checkpoint_sha256=endpoint['checkpoint_sha256'],
    frozen_inputs=len(plan['source_sha256']),decision=decision,
    source_clock=1501,candidate_clock=1509,new_PPO=8,
    actual_experiment_consumed=2630,repair_optimizer_updates=250,
    qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'projected_round013_fixed_decision001.json',out)
p=json.loads((R/'progress.json').read_text());p.update(status='projected_common_entry_fixed_reviewed',
    new_PPO_attempted=250,new_PPO_completed=250,repair_optimizer_updates=250,actual_experiment_consumed=2630,
    active_job=None,active_training=None,active_supervisor=None,active_evaluation=None,active_learning_clock=None,
    new_PPO_admitted=0,training_admission=None,latest_completed_evaluation=str(run),
    latest_research_decision=decision,
    lineage='Best candidate clock1501 remains unchanged. Independent013 finished1509 after8 updates; cumulative repair250 and experiment2630. No refunds.',
    planned_PPO='No new PPO admitted. Apply expanded zero-PPO review if fixed gates pass; otherwise diagnose the failed behavior.',updated_unix=time.time())
if not any(x.get('name')=='pilot013_projected_entry' for x in p['experiments']):
    p['experiments'].append(dict(name='pilot013_projected_entry',updates=8,clock=1509,status=decision,checkpoint=endpoint['checkpoint'],checkpoint_sha256=endpoint['checkpoint_sha256']))
write_json(R/'progress.json',p)
print(json.dumps(out,indent=2))
