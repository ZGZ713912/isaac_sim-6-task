"""Keep the frozen yaw-mechanism gate separate from standing qualification."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts', str(Path(__file__).resolve().parent)]
import numpy as np
from pair_declared_mean_probe001 import pair
from run_gap_long_training import behavior_metrics
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
plan = json.loads((R / 'diagnosis_plan052.json').read_text())
assert json.loads((R / 'diagnosis_queue052.json').read_text())['status'] == 'completed'
assert all(digest(p) == s for p, s in plan['source_sha256'].items())
protocol = json.loads((R / 'quiet_common_reverse_protocol001.json').read_text())
run = Path(plan['jobs'][0]['output'])
standing = json.loads((R / 'quiet_common_reverse_development_standing001.json').read_text())
before = json.loads((R / 'quiet_stance050_development_standing001.json').read_text())
old = {x['name']: x for x in before['rows']}
lost = [x['name'] for x in standing['rows'] if old[x['name']]['pose_and_position_pass'] and not x['pose_and_position_pass']]
gained = [x['name'] for x in standing['rows'] if not old[x['name']]['pose_and_position_pass'] and x['pose_and_position_pass']]
paired = pair(protocol['control'], run, override_key='quiet_wheel_common_reverse_probe',
              schema='quiet-wheel-common-sign-reversal-after-yaw-v1')
write_json(R / 'quiet_common_reverse_development_pairing001.json', paired)
rows = {}
for label, path in [('before', Path(protocol['control'])), ('after', run)]:
    report = json.loads((path / 'evaluation/report.json').read_text())
    with np.load(path / 'evaluation/trajectory.npz') as trace:
        for lane, rec in enumerate(report['lanes']):
            if rec['name'] in protocol['gates']['yaw_cases']:
                mask = (trace['time'][:, lane] > 6.) & trace['alive'][:, lane]
                assert mask.sum() >= 399
                value = float(np.abs(trace['omega'][mask, lane, 2]).mean())
                rows.setdefault(rec['name'], {})[label] = value
ratio = protocol['gates']['yaw_at_most_fraction_of_original']
checks = dict(frozen_inputs=True, native_pairing=all(paired['checks'].values()),
    standing_survived=sum(x['survived'] for x in standing['rows']) == 21,
    no_previous_pass_lost=not lost, closure=report['max_gap_m'] <= .003,
    passive=report['passive_motor_peak_nm'] == 0.)
for name, row in rows.items():
    row['after_before_ratio'] = row['after'] / row['before']
    checks['yaw_halved_' + name] = row['after'] <= ratio * row['before']
mechanism = all(checks.values())
position = standing['pose_and_position_pass'] >= 19
out = dict(schema='quiet-common-sign-native-review-v1', checks={k: bool(v) for k, v in checks.items()},
    native_tail_absolute_yaw_rad_s=rows, mechanism_gate_pass=mechanism,
    standing_gate_pass=position, standing_pass=standing['pose_and_position_pass'],
    gained_passes=gained, lost_passes=lost,
    failed_standing=[x for x in standing['rows'] if not x['pose_and_position_pass']],
    metrics=behavior_metrics(run, R / 'quiet_common_reverse_development_standing001.json'),
    decision='permit_further_zero_PPO_confirmation_only' if mechanism and position else 'reject_common_response_hypothesis',
    eligible_for_checkpoint_mean_qualification=False, new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_common_reverse_development_decision001.json', out)
write_json(R / 'diagnosis052_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
    status='completed', all_frozen_sources_verified=True, frozen_inputs=len(plan['source_sha256']),
    source_map_sha256=digest(R / 'diagnosis_plan052.json'), review_sha256=digest(__file__),
    new_PPO=0, actual_experiment_consumed=2654, qualification=None, created_unix=time.time()))
print(json.dumps({k: v for k, v in out.items() if k not in ('failed_standing', 'metrics')}, indent=2))
