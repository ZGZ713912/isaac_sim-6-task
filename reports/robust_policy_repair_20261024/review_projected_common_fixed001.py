"""Apply the declared fixed gates to the one projected common-only candidate."""
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
from run_gap_long_training import behavior_metrics

R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'diagnosis_plan031.json').read_text())
queue=json.loads((R/'diagnosis_queue031.json').read_text());assert queue['status']=='completed'
run=Path(plan['jobs'][0]['output']);assert json.loads((run/'progress.json').read_text())['status']=='completed'
assert all(digest(Path(n))==h for n,h in plan['source_sha256'].items())
standing_path=R/'native_common_projected_standing001.json';paired_path=R/'native_common_projected_comparison001.json'
with (R/'native_common_projected_analysis001.log').open('x') as log:
    subprocess.run([sys.executable,'scripts/analyze_gap_standing.py','--run',str(run),'--output',str(standing_path)],check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run([sys.executable,'scripts/compare_gap_robustness.py','--before',plan['baseline_evaluation'],'--after',str(run),'--output',str(paired_path)],check=True,stdout=log,stderr=subprocess.STDOUT)
metrics=behavior_metrics(run,standing_path);gate=plan['review_gate']
report=json.loads((run/'evaluation/report.json').read_text());pair=json.loads(paired_path.read_text())
standing=json.loads(standing_path.read_text())
impacts=sum(any('impact' in cause for cause in row['termination_causes']) for row in report['lanes'])
improvement=(metrics['entry']['followup_success']>=9 or metrics['fallen']['recovered']>=5 or metrics['fallen']['followup_success']>=3)
checks=dict(frozen_inputs=True,
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
    paired_initial=all(pair['initial_exact'].values()) and pair['sensor_actuator_material_exact'] and pair['rigid_parameters_exact'],
    measured_improvement=improvement)
out=dict(schema='projected-common-fixed-decision-v1',checks=checks,metrics=metrics,
    standing_pass=standing['pose_and_position_pass'],impacts=impacts,
    checkpoint=plan['jobs'][0]['checkpoint'],checkpoint_sha256=digest(Path(plan['jobs'][0]['checkpoint'])),
    frozen_inputs=len(plan['source_sha256']),
    decision='admit_zero_PPO_expanded_review' if all(checks.values()) else 'reject_expansion_training_and_adoption',
    new_PPO=0,charged_branch_PPO=1,source_clock=1501,candidate_clock=1502,
    actual_experiment_consumed=2622,repair_optimizer_updates=242,
    qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'native_common_projected_fixed_decision001.json',out)
p=json.loads((R/'progress.json').read_text());p.update(status='native_common_projected_fixed_reviewed',
    active_job=None,active_evaluation=None,active_native_queue=None,
    latest_completed_evaluation=str(run),latest_research_decision=out['decision'],updated_unix=time.time())
write_json(R/'progress.json',p);print(json.dumps(out,indent=2))
