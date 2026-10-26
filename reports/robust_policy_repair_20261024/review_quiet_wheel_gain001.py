"""Review the declared wheel-only probe without treating it as a saved policy."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts', str(Path(__file__).resolve().parent)]
from pair_declared_mean_probe001 import pair
from run_gap_long_training import behavior_metrics
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
plan = json.loads((R / 'diagnosis_plan048.json').read_text())
assert json.loads((R / 'diagnosis_queue048.json').read_text())['status'] == 'completed'
assert all(digest(p) == sha for p, sha in plan['source_sha256'].items())
protocol = json.loads((R / 'quiet_wheel_gain_protocol001.json').read_text())
run = Path(plan['jobs'][0]['output'])
assert json.loads((run / 'progress.json').read_text())['status'] == 'completed'
standing = json.loads((R / 'quiet_wheel_gain_development_standing001.json').read_text())
before = json.loads((R / 'quiet_stance050_development_standing001.json').read_text())
paired = pair(protocol['control'], run, override_key='quiet_wheel_gain_probe',
              schema='quiet-wheel-displacement-gain3-diagnostic-v1')
write_json(R / 'quiet_wheel_gain_development_pairing001.json', paired)
old = {r['name']: r for r in before['rows']}
lost = [r['name'] for r in standing['rows'] if old[r['name']]['pose_and_position_pass'] and not r['pose_and_position_pass']]
gained = [r['name'] for r in standing['rows'] if not old[r['name']]['pose_and_position_pass'] and r['pose_and_position_pass']]
report = json.loads((run / 'evaluation/report.json').read_text())
metrics = behavior_metrics(run, R / 'quiet_wheel_gain_development_standing001.json')
gate = protocol['gates']
checks = dict(frozen_inputs=True, physical_pairing=all(paired['checks'].values()),
    standing_survived=sum(r['survived'] for r in standing['rows']) == gate['standing_survived'],
    standing_pose_position=standing['pose_and_position_pass'] >= gate['standing_pose_position_at_least'],
    no_previous_pass_lost=len(lost) <= gate['lost_previous_development_passes_at_most'],
    closure=report['max_gap_m'] <= gate['max_gap_m'],
    passive=report['passive_motor_peak_nm'] == gate['passive_motor_peak_nm'])
out = dict(schema='quiet-wheel-gain-development-review-v1', checks={k: bool(v) for k, v in checks.items()},
    standing_pass=standing['pose_and_position_pass'], previous_standing_pass=before['pose_and_position_pass'],
    lost_passes=lost, gained_passes=gained, metrics=metrics,
    failed_standing=[r for r in standing['rows'] if not r['pose_and_position_pass']],
    decision='permit_further_zero_PPO_assessment_only' if all(checks.values()) else 'reject_intervention',
    standard_policy_only_comparison_accepted=False, eligible_for_checkpoint_mean_qualification=False,
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_wheel_gain_development_decision001.json', out)
write_json(R / 'diagnosis048_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
    status='completed', all_frozen_sources_verified=True, frozen_inputs=len(plan['source_sha256']),
    source_map_sha256=digest(R / 'diagnosis_plan048.json'), review_sha256=digest(__file__),
    new_PPO=0, actual_experiment_consumed=2646, qualification=None, created_unix=time.time()))
print(json.dumps({k: v for k, v in out.items() if k not in ('failed_standing', 'metrics')}, indent=2))
