"""A quiet component must pass the declared physical and behavior gates."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
from run_gap_long_training import behavior_metrics

parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--endpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();plan=json.loads(args.plan.read_text())
run=Path(plan['supervisor_output']);evaluation=Path(plan['evaluation_output'])
state=json.loads((run/'progress.json').read_text());assert state['status']=='review_ready'
bad=[n for n,h in plan['source_sha256'].items() if digest(Path(n))!=h];assert not bad,bad
endpoint=json.loads(args.endpoint.read_text());assert all(endpoint['checks'].values())
standing=json.loads((run/'standing.json').read_text());comparison=json.loads((run/'paired.json').read_text())
baseline=json.loads(Path('reports/robust_policy_repair_20261024/low_diagonal_fixed_standing001.json').read_text())
report=json.loads((evaluation/'evaluation/report.json').read_text())
metrics=behavior_metrics(evaluation,run/'standing.json');gate=plan['review_gate']
oldpass={x['name'] for x in baseline['rows'] if x['pose_and_position_pass']}
newpass={x['name'] for x in standing['rows'] if x['pose_and_position_pass']}
lost=sorted(oldpass-newpass)
oldfailed={x['name']:x['max_excursion_m'] for x in baseline['rows'] if not x['pose_and_position_pass']}
newrows={x['name']:x for x in standing['rows']}
improvement=standing['pose_and_position_pass']==21 or any(newrows[n]['max_excursion_m']<=.8*v for n,v in oldfailed.items())
impacts=sum(any('impact' in x for x in lane['termination_causes']) for lane in report['lanes'])
checks=dict(endpoint=all(endpoint['checks'].values()),frozen_inputs=not bad,
    no_lost_standing_pass=not lost,standing=standing['pose_and_position_pass']>=gate['fixed_standing_at_least'],
    standing_all_survive=metrics['standing']['survived']==21,
    motion_mae=metrics['motion']['vx_mae_m_s']<=gate['motion_MAE_at_most'],
    entry_recovery=metrics['entry']['recovered']>=gate['entry_recovery_at_least'],
    entry_followup=metrics['entry']['followup_success']>=gate['entry_followup_at_least'],
    fallen_recovery=metrics['fallen']['recovered']>=gate['fallen_recovery_at_least'],
    fallen_followup=metrics['fallen']['followup_success']>=gate['fallen_followup_at_least'],
    entry_leg_peak=metrics['entry']['max_leg_speed_rad_s']<=gate['entry_leg_peak_at_most_rad_s'],
    fallen_leg_peak=metrics['fallen']['max_leg_speed_rad_s']<=gate['fallen_leg_peak_at_most_rad_s'],
    standing_leg_peak=metrics['standing']['max_leg_speed_rad_s']<=gate['standing_leg_peak_at_most_rad_s'],
    impacts=impacts<=gate['impact_terminations_at_most'],closure=report['max_gap_m']<=.003,
    passive=report['passive_motor_peak_nm']==0.,
    paired_initial=all(comparison['initial_exact'].values()) and comparison['sensor_actuator_material_exact'],
    measured_improvement=improvement)
out=dict(schema='bounded-quiet-fixed-review-v1',role=plan['role'],checks=checks,metrics=metrics,
    lost_standing_passes=lost,failing_case_comparison={n:dict(before_m=v,after_m=newrows[n]['max_excursion_m']) for n,v in oldfailed.items()},
    impacts=impacts,checkpoint=state['checkpoint'],checkpoint_sha256=state['checkpoint_sha256'],
    all_frozen_inputs_exact=len(plan['source_sha256']),
    decision='admit_zero_PPO_development_validation' if all(checks.values()) else 'reject_extension_and_adoption',
    new_PPO_completed=8,actual_experiment_consumed=state['actual_experiment_consumed'],
    source_clock=1501,branch_clock=1509,qualification=None,hardware_started=False,created_unix=time.time())
write_json(args.output,out);print(json.dumps(out,indent=2))
