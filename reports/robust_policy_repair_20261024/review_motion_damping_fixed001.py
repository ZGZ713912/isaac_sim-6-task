"""Apply the frozen periodic-entry protocol to the completed native fixed run."""
from pathlib import Path
import json
import time

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
protocol = json.loads((R/'motion_damping_protocol001.json').read_text())
gate = protocol['gate']
paired = json.loads((R/'motion_damping_fixed_comparison001.json').read_text())
standing = json.loads((R/'motion_damping_standing001.json').read_text())
run = B/'diagnosis025_motion_damping080_fixed'
assert json.loads((run/'progress.json').read_text())['status'] == 'completed'
rep = json.loads((run/'evaluation/report.json').read_text())
counts = {}
for group, select in [('entry', lambda s: s.startswith('real_start')),
                      ('fallen', lambda s: s in ('left_symmetric','right_symmetric','inverted_symmetric'))]:
    rows = [row for row in rep['lanes'] if select(row['behavior_case'])]
    counts[group] = dict(episodes=len(rows), survived=sum(x['survived'] for x in rows),
                         recovered=sum(x['recovered'] for x in rows), followup=sum(x['followup_success'] for x in rows))
groups = paired['groups']
motion = groups['motion']['after']['vx_mae_m_s']['mean']
entry_peak = groups['entry']['after']['max_leg_speed_rad_s']['max']
fallen_peak = groups['fallen']['after']['max_leg_speed_rad_s']['max']
impacts = sum(any('impact' in c for c in row['termination_causes']) for row in rep['lanes'])
checks = dict(standing=standing['pose_and_position_pass'] >= gate['fixed_standing_at_least'],
    motion_mae=motion <= gate['fixed_motion_mae_at_most'],
    fallen_recovery=counts['fallen']['recovered'] >= gate['fixed_fallen_recovery_at_least'],
    fallen_followup=counts['fallen']['followup'] >= gate['fixed_fallen_followup_at_least'],
    entry_recovery=counts['entry']['recovered'] >= gate['fixed_entry_recovery_at_least'],
    entry_followup=counts['entry']['followup'] >= gate['fixed_entry_followup_at_least'],
    entry_peak=entry_peak <= gate['fixed_entry_leg_peak_at_most_rad_s'],
    fallen_peak=fallen_peak <= gate['fixed_fallen_leg_peak_at_most_rad_s'],
    impacts=impacts <= gate['fixed_impacts_at_most'], closure=rep['max_gap_m'] <= .003,
    passive=rep['passive_motor_peak_nm'] == 0.,
    pairing=all(paired['initial_exact'].values()) and paired['sensor_actuator_material_exact'] and paired['rigid_parameters_exact'])
out = dict(schema='motion-damping-predeclared-fixed-decision-v1', protocol=str(R/'motion_damping_protocol001.json'),
    checks=checks, counts=counts, standing_pass=standing['pose_and_position_pass'], motion_mae=motion,
    entry_peak_rad_s=entry_peak, fallen_peak_rad_s=fallen_peak, impacts=impacts,
    decision='admit_additional_zero_PPO_native_validation' if all(checks.values()) else 'reject_expansion_training_and_adoption',
    new_PPO=0, actual_experiment_consumed=2603, qualification=None, hardware_started=False, created_unix=time.time())
(R/'motion_damping_fixed_decision001.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
