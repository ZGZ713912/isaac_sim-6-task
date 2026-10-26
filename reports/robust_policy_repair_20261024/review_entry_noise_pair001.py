"""Review the declared sampled-policy pair; no deterministic-mean acceptance."""
import json
from pathlib import Path
import sys
import time
sys.path[:0]=['src','scripts']
import numpy as np
from train_robust_getup import digest,write_json

R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
plan=json.loads((R/'diagnosis_plan033.json').read_text());queue=json.loads((R/'diagnosis_queue033.json').read_text())
assert queue['status']=='completed'
assert all(digest(Path(n))==h for n,h in plan['source_sha256'].items())
reports=[];traces=[];rows=[];paths=[]
for job in plan['jobs']:
    run=Path(job['output']);paths.append(run)
    report=json.loads((run/'evaluation/report.json').read_text());reports.append(report)
    summary=json.loads((run/'evaluation/gap_summary.json').read_text())
    assert digest(run/'evaluation/trajectory.npz')==summary['trajectory_sha256']
    z=np.load(run/'evaluation/trajectory.npz');traces.append(z)
    rows.append(dict(name=job['name'],episodes=len(report['lanes']),
        **{k:sum(x[k] for x in report['lanes']) for k in ('survived','recovered','followup_success')},
        impacts=sum(any('impact' in cause for cause in x['termination_causes']) for x in report['lanes']),
        max_leg_speed_rad_s=max(x['max_leg_speed_rad_s'] for x in summary['rows']),
        closure_peak_m=report['max_gap_m'],passive_motor_peak_nm=report['passive_motor_peak_nm'],
        trace_sha256=summary['trajectory_sha256']))
initial=[np.load(p/'evaluation_precontrol_state.npz') for p in paths]
pair_checks={k:bool(np.array_equal(initial[0][k],initial[1][k])) for k in initial[0].files}
pair_checks['initial_policy_input']=bool(np.array_equal(traces[0]['observations'][0],traces[1]['observations'][0]))
pair_checks['same_cases']=[x['name'] for x in reports[0]['lanes']]==[x['name'] for x in reports[1]['lanes']]
pair_checks['domains_and_materials']=all(x['first_reset_domain']==y['first_reset_domain'] and x['material_coefficients']==y['material_coefficients'] for x,y in zip(reports[0]['lanes'],reports[1]['lanes']))
startup=[json.loads((p/'startup.json').read_text()) for p in paths]
pair_checks['rigid_properties']=startup[0]['dynamics_randomization']['sampled_properties_sha256']==startup[1]['dynamics_randomization']['sampled_properties_sha256']
sampling=[x['sampled_policy_exposure_diagnostic'] for x in reports]
pair_checks['same_private_gaussian_draws']=all(sampling[0][k]==sampling[1][k] for k in ('seed','calls','samples_per_call','generator_state_sha256'))
a,b=rows;gate=plan['review_gate']
checks=dict(frozen_inputs=True,paired=all(pair_checks.values()),
    survivors=b['survived']>=a['survived'],impacts=b['impacts']<=a['impacts'],
    recovered_gain=b['recovered']>=a['recovered']+gate['recovered_gain_at_least'],
    followup_count=b['followup_success']>=gate['followup_at_least'],
    followup_no_loss=b['followup_success']>=a['followup_success'],
    leg_peak=b['max_leg_speed_rad_s']<=a['max_leg_speed_rad_s'],
    closure=max(x['closure_peak_m'] for x in rows)<=.003,
    passive=max(x['passive_motor_peak_nm'] for x in rows)==0.)
out=dict(schema='entry-leg-noise-sampled-pair-decision-v1',checks=checks,pairing=pair_checks,rows=rows,
    source_checkpoint=plan['jobs'][0]['checkpoint'],source_sha256=digest(Path(plan['jobs'][0]['checkpoint'])),
    frozen_inputs=len(plan['source_sha256']),
    decision='eligible_to_design_a_bounded_noise_adjusted_PPO_test' if all(checks.values()) else 'reject_PPO_admission_from_this_noise_pair',
    qualification_scope='Private sampled exposure only; never qualifies the deterministic mean or hardware.',
    new_PPO=0,actual_experiment_consumed=2622,repair_optimizer_updates=242,
    qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'entry_noise_pair_decision001.json',out);print(json.dumps(out,indent=2))
p=json.loads((R/'progress.json').read_text());p.update(status='entry_noise_pair_reviewed',active_job=None,
    active_evaluation=None,active_native_queue=None,latest_research_decision=out['decision'],updated_unix=time.time())
write_json(R/'progress.json',p)
