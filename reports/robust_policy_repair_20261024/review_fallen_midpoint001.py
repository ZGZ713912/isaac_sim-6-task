"""Apply the unchanged fixed behavior gates to the already consumed4-update midpoint."""
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = ['src', 'scripts', 'reports/robust_policy_repair_20261024']
from train_robust_getup import digest, write_json
from run_gap_long_training import behavior_metrics

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R/'diagnosis_plan043.json').read_text())
assert json.loads((R/'diagnosis_queue043.json').read_text())['status']=='completed'
assert all(digest(Path(n))==h for n,h in plan['source_sha256'].items())
protocol = json.loads((R/'fallen_midpoint_protocol001.json').read_text())
gate = protocol['gate']
run = Path(plan['jobs'][0]['output'])
standing_path = R/'fallen_midpoint_standing001.json'
if not standing_path.exists():
    subprocess.run([sys.executable, 'scripts/analyze_gap_standing.py', '--run', str(run),
        '--output', str(standing_path)], check=True, stdout=subprocess.DEVNULL)
standing = json.loads(standing_path.read_text())
assert standing['trajectory_sha256']==digest(run/'evaluation/trajectory.npz')
paired_path = R/'fallen_midpoint_pairing001.json'
if not paired_path.exists():
    subprocess.run([sys.executable, 'scripts/compare_gap_robustness.py',
        '--before', str(B/'diagnosis025_motion_damping080_fixed'), '--after', str(run),
        '--output', str(paired_path)], check=True, stdout=subprocess.DEVNULL)
comparison = json.loads(paired_path.read_text())
progress = json.loads((run/'progress.json').read_text())
report = json.loads((run/'evaluation/report.json').read_text())
metrics = behavior_metrics(run, standing_path)
impacts = sum(any('impact' in x for x in row['termination_causes']) for row in report['lanes'])
checks = dict(frozen_sources=True, paired_initial=all(comparison['initial_exact'].values()) and comparison['rigid_parameters_exact'] and comparison['sensor_actuator_material_exact'],
    source_archived=all(digest(run/'source'/n)==h for n,h in progress['source_sha256'].items()),
    standing=standing['pose_and_position_pass']>=gate['standing_at_least'],
    motion=metrics['motion']['vx_mae_m_s']<=gate['motion_MAE_at_most'],
    entry_recovery=metrics['entry']['recovered']>=gate['entry_recovery_at_least'],
    entry_followup=metrics['entry']['followup_success']>=gate['entry_followup_at_least'],
    fallen_recovery=metrics['fallen']['recovered']>=gate['fallen_recovery_at_least'],
    fallen_followup=metrics['fallen']['followup_success']>=gate['fallen_followup_at_least'],
    entry_speed=metrics['entry']['max_leg_speed_rad_s']<=gate['entry_peak_at_most_rad_s'],
    fallen_speed=metrics['fallen']['max_leg_speed_rad_s']<=gate['fallen_peak_at_most_rad_s'],
    impacts=impacts<=gate['impacts_at_most'], closure=report['max_gap_m']<=.003,
    passive=report['passive_motor_peak_nm']==0.,
    measured_gain=metrics['fallen']['recovered']>=5 or metrics['entry']['followup_success']>=9
        or metrics['fallen']['followup_success']>=3)
decision = 'admit_zero_PPO_expanded_function_diagnosis' if all(checks.values()) else 'reject_adoption_and_PPO_extension'
out = dict(schema='fallen-output-midpoint-native-review-v1', checks={k:bool(v) for k,v in checks.items()},
    standing_pass=standing['pose_and_position_pass'], metrics=metrics, impacts=impacts, decision=decision,
    frozen_inputs=len(plan['source_sha256']), native_checkpoint_mean_unmodified=True,
    checkpoint_clock=1505, already_consumed_updates=4, new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R/'fallen_midpoint_decision001.json', out)
write_json(R/'diagnosis043_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1',
    queue='043', source_files=len(plan['source_sha256']), all_frozen_sources_verified=True,
    decision_sha256=digest(R/'fallen_midpoint_decision001.json'), new_PPO=0,
    qualification=None, created_unix=time.time()))
p = json.loads((R/'progress.json').read_text())
p.update(status='fallen_midpoint_diagnosed', active_job=None, active_supervisor=None,
    active_evaluation=None, active_native_queue=None, active_diagnosis_queue=None,
    new_PPO_admitted=0, latest_research_decision=decision, updated_unix=time.time())
write_json(R/'progress.json', p)
print(json.dumps(out, indent=2))
