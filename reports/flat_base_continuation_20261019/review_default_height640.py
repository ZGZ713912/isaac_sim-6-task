"""Review the completed default-height pair without changing native evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_flat_base import compare
from flat_default_height_profile import read_default_height_contract
from flat_rmcs_command_reference import read_reference_contract

RUN = ROOT / 'runs/flat_base_20261019'
REPORT = Path(__file__).parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = [RUN / n / 'evaluation.json' for n in ('rmcs_default_source_envelope001','rmcs_default640_envelope001')]
a,b = [json.loads(p.read_text()) for p in paths]
if any(x['status'] != 'evaluated' for x in (a,b)):
    raise SystemExit('The default-height pair has not completed; no review emitted')
rmcs = Path("/home/noir/Documents/workspace/RMCS")
reference = read_reference_contract(rmcs)
default = read_default_height_contract(rmcs, reference)
queue = json.loads((RUN/'default_height640_queue001.json').read_text())
checks = dict(candidate_exact_640_checkpoint=b['checkpoint_sha256']==sha(RUN/'pilot013/model_000640.pt'),
    source_actor_exact=all(x['source_onnx_sha256']==sha(ROOT/'models/wheel_leg_flat_base_20261018/source.onnx') for x in (a,b)),
    clocks=all((x['policy_hz'],x['physics_hz'])==(50,1000) for x in (a,b)),
    seed=all(x['seed']==191029 for x in (a,b)),
    all_88_episodes_accounted=all(len(x['result']['cases'])==22 and all(c['episodes']==4 for c in x['result']['cases'].values()) for x in (a,b)),
    default_height_contract=all(x['packaged_default_height']==default for x in (a,b)),
    rmcs_reference_contract=all(x['rmcs_command_reference']==reference for x in (a,b)),
    original_frozen_source_unchanged=all(sha(ROOT/p)==s for p,s in queue['source_sha256'].items()))
paired = compare(a,b)
rows = {}
regressions = []
for name,old in a['result']['cases'].items():
    new = b['result']['cases'][name]
    keys=('passed','full_horizon_episodes','vx_mae_m_s','yaw_mae_rad_s','height_mae_m','stand_drift_max_m','termination_reasons')
    rows[name] = dict(command=new['command'], source={k:old[k] for k in keys},candidate640={k:new[k] for k in keys})
    for key,tolerance in (('vx_mae_m_s',.02),('yaw_mae_rad_s',.05)):
        if new[key]>old[key]+tolerance:
            regressions.append(dict(case=name,metric=key,before=old[key],after=new[key],tolerance=tolerance))
summary=lambda d:dict(passed_cases=sum(c['passed'] for c in d['result']['cases'].values()),
    full_horizon_episodes=sum(c['full_horizon_episodes'] for c in d['result']['cases'].values()),
    requested_episodes=88,elapsed_s=d['elapsed_s'],
    termination_totals={k:sum(c['termination_reasons'][k] for c in d['result']['cases'].values()) for k in next(iter(d['result']['cases'].values()))['termination_reasons']})
report=dict(schema='flat-feedback640-default-height-paired-review-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    pair_complete=True,all_source_and_checkpoint_bindings_valid=all(checks.values()),checks=checks,
    evaluation_sha256={p.parent.name:sha(p) for p in paths},source=summary(a),candidate640=summary(b),
    packaged_default_height=default,paired_comparison=paired,no_relative_regressions=not regressions,
    relative_regressions=regressions,cases=rows,optimizer_updates_performed=0,
    behavior_qualified=False,heldout_confirmation_completed=False,hardware_motion_started=False)
p=REPORT/'feedback640_default_height_review001.json'
p.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(dict(pair_complete=True,bindings_valid=all(checks.values()),failed_checks=[k for k,v in checks.items() if not v],source=report['source'],candidate640=report['candidate640'],relative_regressions=regressions),ensure_ascii=False))
if not all(checks.values()):
    raise SystemExit(1)
