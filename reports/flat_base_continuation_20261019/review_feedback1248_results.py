"""Review the three development screens after the bounded clock pilot exits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_flat_base import compare

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
output = HERE / 'feedback1248_behavior_review001.json'
if output.exists():
    raise SystemExit('Preserve the completed screening review')
queue = read(RUN / 'feedback1248_queue001.json')
if queue['status'] != 'complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for all native children and the queue to finish and exit')
checkpoint = RUN / 'pilot023/model_001248.pt'
checkpoint_sha = sha(checkpoint)
rows = {}
specs = (
    ('deployment1248_basic001', [('source','deployment_source001'),('parent1184','deployment1184_basic001')]),
    ('rmcs_default1248_envelope001', [('source','rmcs_default_source_envelope001'),('parent1184','rmcs_default1184_envelope001')]),
    ('feedback1248_basic001', [('source','baseline_native002'),('parent1184','feedback1184_basic001')]))
for name, references in specs:
    path = RUN / name / 'evaluation.json'
    data = read(path)
    if data['status'] != 'evaluated' or data['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Incomplete or mismatched checkpoint replay: '+name)
    cases, comparisons = data['result']['cases'], {}
    for label, reference in references:
        refpath = RUN / reference / 'evaluation.json'
        baseline = read(refpath)
        paired = compare(baseline,data)
        relative = []
        for case_name, case in cases.items():
            vx,yaw,_ = case['command']
            if label == 'source' and abs(vx) < .75 and abs(yaw) < 1.5:
                continue
            tolerances = dict(vx_mae_m_s=.02,yaw_mae_rad_s=.05)
            if label != 'source' and case_name.startswith('stand_'):
                tolerances['stand_drift_max_m'] = .03
            previous = baseline['result']['cases'][case_name]
            for key, tolerance in tolerances.items():
                if case[key] > previous[key] + tolerance:
                    relative.append(dict(case=case_name,metric=key,before=previous[key],after=case[key],tolerance=tolerance))
        comparisons[label] = dict(reference_sha256=sha(refpath),no_lost_checks=paired['no_lost_baseline_checks'],
            lost_checks=paired['lost_baseline_checks'],no_relative_regressions=not relative,relative_regressions=relative)
    rows[name] = dict(evaluation_sha256=sha(path),passed_cases=sum(c['passed'] for c in cases.values()),
        requested_cases=len(cases),all_targets_passed=data['result']['passed'],comparisons=comparisons,
        requested_episodes=sum(c['requested_episodes'] for c in cases.values()),
        accounted_episodes=sum(c['episodes'] for c in cases.values()),
        full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
        all_horizons_completed=all(c['full_horizon_episodes']==c['requested_episodes'] for c in cases.values()),
        termination_totals={k:sum(c['termination_reasons'][k] for c in cases.values())
            for k in next(iter(cases.values()))['termination_reasons']},
        metrics={n:{key:c[key] for key in ('vx_mae_m_s','yaw_mae_rad_s','height_mae_m',
            'stand_drift_max_m','full_horizon_episodes','termination_reasons')} for n,c in cases.items()})
progress = read(RUN / 'pilot023/progress.json')
counts = {key:progress[key] for key in ('successful_updates_total','actor_updates_total','active_branch_actor_updates',
    'feedback_branch_actor_updates','post_split_actor_updates','inherited_shared_feedback_updates','consumed_optimizer_updates_total')}
report = dict(schema='flat-feedback1248-unchanged-v16-pilot-screening-review-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    checkpoint_sha256=checkpoint_sha,training_status=progress['status'],**counts,
    source_training_updates='unknown',cohorts=rows,all_three_cohorts_complete=True,
    screening_scope='Raw1000Hz basic, RMCS-default1000Hz full envelope, raw200Hz basic only. '
        'Nominal1000Hz full envelope, heldout and remaining release cohorts unassessed.',
    long_training_started=False,next_PPO_block_queued=False,behavior_qualified=False,hardware_motion_started=False)
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({n:dict(passed_cases=r['passed_cases'],requested_cases=r['requested_cases'],
    full_horizon_episodes=r['full_horizon_episodes'],requested_episodes=r['requested_episodes'],
    relative_regressions={k:len(v['relative_regressions']) for k,v in r['comparisons'].items()},
    lost_check_cases={k:len(v['lost_checks']) for k,v in r['comparisons'].items()}) for n,r in rows.items()}))
