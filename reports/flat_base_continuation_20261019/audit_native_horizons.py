"""Separate first-episode accounting from full-duration survival in saved results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
rows=[]
for path in sorted((ROOT/'runs/flat_base_20261019').glob('*/evaluation.json')):
    data=json.loads(path.read_text())
    if data.get('status')!='evaluated' or not data.get('result',{}).get('cases'):
        continue
    cases=data['result']['cases']
    if not all('full_horizon_episodes' in c for c in cases.values()):
        continue
    requested=sum(c['requested_episodes'] for c in cases.values())
    accounted=sum(c['episodes'] for c in cases.values())
    full=sum(c['full_horizon_episodes'] for c in cases.values())
    if any(not 0<=c['full_horizon_episodes']<=c['episodes']<=c['requested_episodes'] for c in cases.values()):
        raise ValueError('Invalid horizon arithmetic: '+str(path))
    rows.append(dict(run=path.parent.name,evaluation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=data.get('checkpoint_sha256'),protocol_id=data['settings']['protocol_id'],
        policy_hz=data['policy_hz'],physics_hz=data['physics_hz'],diagnostic_only=bool(data.get('diagnostic_counterfactual')),
        requested=requested,accounted=accounted,full_horizon=full,
        passed_cases=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
        early_cases={n:dict(requested=c['requested_episodes'],full_horizon=c['full_horizon_episodes'],
            termination_reasons=c['termination_reasons']) for n,c in cases.items()
            if c['full_horizon_episodes']<c['requested_episodes']}))
report=dict(schema='flat-native-horizon-accounting-audit-v2',recorded_at=datetime.now(timezone.utc).isoformat(),
    scope='Read-only saved evaluations; protocols and durations remain separate; no behavior qualification',
    invariant='accounted first episodes do not imply full-duration survival',evaluations=rows)
Path(__file__).with_name('native_horizon_accounting_audit002.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({r['run']:dict(requested=r['requested'],accounted=r['accounted'],full_horizon=r['full_horizon'])
    for r in rows if any(s in r['run'] for s in ('704','768'))}))
