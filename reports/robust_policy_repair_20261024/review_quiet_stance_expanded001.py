"""Review the complete, predeclared motion-damping native comparisons."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from analyze_robust_getup import analyze

R = ROOT / 'reports/robust_policy_repair_20261024'
B = ROOT / 'runs/robust_policy_repair_20261024'


def read(path):
    return json.loads(path.read_text())


def save(name, data):
    (R / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


queue = read(R / 'diagnosis_queue037.json')
assert queue['status'] == 'completed'
plan = read(R / 'diagnosis_plan037.json')
changed = [name for name, sha in plan['source_sha256'].items()
           if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha]
assert not changed, changed
save('quiet_stance_expanded_completion001.json', dict(
    schema='frozen-native-queue-completion-v1', inputs=len(plan['source_sha256']),
    changed_inputs=changed, jobs=queue['jobs'], new_PPO=0, created_unix=time.time()))
gate = read(R / 'quiet_stance_expanded_protocol001.json')['gate']
run = B / 'diagnosis037_quiet_stance050_rmcs/evaluation'
report, summary = read(run / 'report.json'), read(run / 'gap_summary.json')
comparison = read(R / 'quiet_stance_RMCS_comparison001.json')
rmcs = dict(episodes=len(report['lanes']), survived=sum(x['survived'] for x in report['lanes']),
    recovered=sum(x['recovered'] for x in report['lanes']),
    followup=sum(x['followup_success'] for x in report['lanes']),
    tail90=sum(x['tail_stable_fraction'] >= .9 for x in summary['rows']),
    leg_peak=max(x['max_leg_speed_rad_s'] for x in summary['rows']),
    impacts=sum(any('impact' in c for c in x['termination_causes']) for x in report['lanes']),
    max_gap_m=report['max_gap_m'], passive_motor_peak_nm=report['passive_motor_peak_nm'])
checks = dict(RMCS_survival=rmcs['survived'] >= gate['RMCS_survival_at_least'], RMCS_recovery=rmcs['recovered'] >= gate['RMCS_recovery_at_least'],
    RMCS_followup=rmcs['followup'] >= gate['RMCS_followup_at_least'],
    RMCS_tail90=rmcs['tail90'] >= gate['RMCS_tail90_at_least'],
    RMCS_leg_peak=rmcs['leg_peak'] <= gate['RMCS_leg_peak_at_most_rad_s'],
    RMCS_impacts=rmcs['impacts'] <= gate['RMCS_impacts_at_most'],
    RMCS_closure=rmcs['max_gap_m'] <= .003, RMCS_passive=rmcs['passive_motor_peak_nm'] == 0.,
    RMCS_pairing=all(comparison['initial_exact'].values())
        and comparison['sensor_actuator_material_exact'] and comparison['rigid_parameters_exact'])

nominal = B / 'diagnosis037_quiet_stance050_nominal_pair'
assert read(nominal / 'progress.json')['status'] == 'completed'
reports = {name: read(nominal / name / 'report.json') for name in ('robust', 'reference')}
analyses = {name: analyze(nominal / name) for name in reports}
for name, data in analyses.items():
    save('quiet_stance_nominal_' + name + '_analysis001.json', data)
precontrol = dict(np.load(nominal / 'all_precontrol_state.npz'))
lane_count = len(reports['robust']['lanes'])
assert lane_count == 70
pairing = {key: bool(np.array_equal(value[:lane_count], value[lane_count:]))
           for key, value in precontrol.items()}
traces = {name: np.load(nominal / name / 'trajectory.npz') for name in reports}
pairing['initial_observations'] = bool(np.array_equal(
    traces['robust']['observations'][0], traces['reference']['observations'][0]))
case_keys = ('name', 'command', 'recovery', 'sequence', 'push', 'push_sign',
             'pose_index', 'world_origin_m')
pairing['cases'] = all(all(a.get(key) == b.get(key) for key in case_keys)
    for a, b in zip(reports['robust']['lanes'], reports['reference']['lanes'], strict=True))
# In nominal domains only disabled wobble frequency/phase may differ. They do
# not affect the sampled sensors, which are separately compared above.
def effective_domain(domain):
    result = dict(domain)
    if result.get('imu_wobble_amplitude_rad') == [0., 0.]:
        for key in ('imu_wobble_frequency_hz', 'imu_wobble_phase_rad'):
            result.pop(key, None)
    return result
pairing['effective_domains'] = all(effective_domain(a['first_reset_domain'])
    == effective_domain(b['first_reset_domain']) for a, b in zip(
        reports['robust']['lanes'], reports['reference']['lanes'], strict=True))
pairing['domains_disabled'] = all(report['robust_domains']['enabled_environments'] == 0
                                  for report in reports.values())
families = analyses['robust']['actual_settled_attitudes']
for family in ('compound', 'inverted', 'right', 'left'):
    checks['nominal_' + family] = families.get(family, {}).get(
        'practical_recovery_and_motion_success', 0) >= gate['nominal_' + family + '_followup_at_least']
checks.update(nominal_motion=analyses['robust']['motion_survived'] >= gate['nominal_motion_survival_at_least'],
    nominal_pairing=all(pairing.values()),
    nominal_closure=analyses['robust']['max_gap_m'] <= .003,
    nominal_passive=analyses['robust']['passive_motor_peak_nm'] == 0.)
compact = {name: {key: row[key] for key in ('actual_settled_attitudes', 'motion_episodes',
    'motion_survived', 'practical_moving_fall_recoveries', 'max_gap_m', 'passive_motor_peak_nm')}
    for name, row in analyses.items()}
out = dict(schema='quiet-stance-expanded-decision-v1', checks=checks,
    RMCS=rmcs, nominal=compact, pairing=pairing,
    decision='promote_development_candidate_only' if all(checks.values()) else 'reject_candidate_adoption_and_PPO',
    new_PPO=0, actual_experiment_consumed=2630, qualification=None,
    hardware_started=False, created_unix=time.time())
save('quiet_stance_expanded_decision001.json', out)
print(json.dumps(out, indent=2))

if all(checks.values()):
    fixed = read(R / 'quiet_stance050_fixed_decision001.json')
    cp = plan['jobs'][0]['checkpoint']
    progress = read(R / 'progress.json')
    progress.update(status='quiet_stance_development_candidate_promoted',current_candidate=cp,current_candidate_clock=1501,current_learning_clock=1501,
        standing_pass=fixed['standing_pass'],active_native_queue=None,active_diagnosis_queue=None,active_supervisor=None,active_job=None,
        qualification=None,qualified=False,selected_deployment_model=None,new_PPO_admitted=0,
        source_promotion=str(R/'quiet_stance_expanded_decision001.json'),updated_unix=time.time())
    save('progress.json',progress)
