"""Apply unchanged fixed nonregression gates before zero-PPO moving-fall assessment."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json
from run_gap_long_training import behavior_metrics

R = Path('reports/robust_policy_repair_20261024')
plan = json.loads((R / 'plan017_quiet_output001.json').read_text())
sup = Path(plan['supervisor_output'])
queue = json.loads((sup / 'progress.json').read_text())
assert queue['status'] == 'review_ready'
endpoint = json.loads((R / 'quiet_round017_endpoint001.json').read_text())
assert all(endpoint['checks'].values()) and endpoint['real_PPO_completed'] == 8
run = Path(plan['evaluation_output'])
assert json.loads((run / 'progress.json').read_text())['status'] == 'completed'
standing = json.loads((sup / 'standing.json').read_text())
paired = json.loads((sup / 'paired.json').read_text())
metrics = behavior_metrics(run, sup / 'standing.json')
report = json.loads((run / 'evaluation/report.json').read_text())
gate = plan['review_gate']
impacts = sum(any('impact' in x for x in row['termination_causes']) for row in report['lanes'])
checks = dict(endpoint_audit=True,
    frozen_inputs=all(digest(Path(n)) == h for n, h in plan['source_sha256'].items()),
    standing=standing['pose_and_position_pass'] >= gate['standing_at_least'],
    motion=metrics['motion']['vx_mae_m_s'] <= gate['motion_MAE_at_most'],
    entry_recovery=metrics['entry']['recovered'] >= gate['entry_recovery_at_least'],
    entry_followup=metrics['entry']['followup_success'] >= gate['entry_followup_at_least'],
    fallen_recovery=metrics['fallen']['recovered'] >= gate['fallen_recovery_at_least'],
    fallen_followup=metrics['fallen']['followup_success'] >= gate['fallen_followup_at_least'],
    entry_speed=metrics['entry']['max_leg_speed_rad_s'] <= gate['entry_peak_at_most_rad_s'],
    fallen_speed=metrics['fallen']['max_leg_speed_rad_s'] <= gate['fallen_peak_at_most_rad_s'],
    impacts=impacts <= gate['impacts_at_most'], closure=report['max_gap_m'] <= gate['max_gap_m'],
    passive=report['passive_motor_peak_nm'] == gate['passive_motor_peak_nm'],
    paired_initial=all(paired['initial_exact'].values()) and paired['sensor_actuator_material_exact']
        and paired['rigid_parameters_exact'])
decision = 'admit_zero_PPO_expanded_target_review' if all(checks.values()) else 'reject_extension_and_adoption'
out = dict(schema='bounded-quiet-output-fixed-review-v1', checks={k: bool(v) for k, v in checks.items()},
    metrics=metrics, standing_pass=standing['pose_and_position_pass'], impacts=impacts,
    checkpoint=endpoint['checkpoint'], checkpoint_sha256=endpoint['checkpoint_sha256'],
    frozen_inputs=len(plan['source_sha256']), decision=decision,
    target_behavior_still_required='Development standing must improve above17/21, then full30s and original moving-fall/RMCS/heldout qualification. Neither stable loss nor frozen outside-scope actions proves this.',
    source_clock=1501, candidate_clock=1509, new_PPO=8,
    actual_experiment_consumed=2662, repair_optimizer_updates=282,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_round017_fixed_decision001.json', out)
p = json.loads((R / 'progress.json').read_text())
p.update(status='quiet_output_fixed_reviewed', new_PPO_attempted=282, new_PPO_completed=282,
    repair_optimizer_updates=282, actual_experiment_consumed=2662,
    active_job=None, active_training=None, active_supervisor=None, active_evaluation=None,
    active_learning_clock=None, new_PPO_admitted=0, training_admission=None,
    latest_completed_evaluation=str(run), latest_research_decision=decision,
    planned_PPO='No new PPO admitted; zero-PPO expanded target assessment only if fixed gates pass.',
    lineage='Best9a clock1501 unchanged; independent017 completed8 updates1501to1509. Repair282, experiment2662; no refunds.',
    updated_unix=time.time())
if not any(x.get('name') == 'pilot017_quiet_output' for x in p['experiments']):
    p['experiments'].append(dict(name='pilot017_quiet_output', updates=8, clock=1509,
        status=decision, checkpoint=endpoint['checkpoint'], checkpoint_sha256=endpoint['checkpoint_sha256']))
write_json(R / 'progress.json', p)
print(json.dumps(out, indent=2))
