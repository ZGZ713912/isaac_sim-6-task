"""Read the sealed basic and two height cohorts without inferring qualification."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = HERE/'low_gate1280_behavior_review001.json'
if out.exists():
    raise SystemExit('Preserve the completed behavior review')
basic_path = HERE/'low_gate1280_completion001.json'
height_path = HERE/'low_gate1280_height_completion001.json'
basic, height = read(basic_path), read(height_path)
if not basic['passed'] or not height['passed'] or basic['checkpoint_sha256'] != height['checkpoint_sha256']:
    raise SystemExit('Wait for both same1280 completion seals')
cohorts = {}
for name, expected in [('deployment1280_basic001', basic['evaluation_sha256'])] + [
        (name,row['evaluation_sha256']) for name,row in height['cohorts'].items()]:
    path = RUN/name/'evaluation.json'
    if sha(path) != expected:
        raise ValueError('Sealed evaluation changed')
    data = read(path)
    cases = {case_name:{k:v for k,v in case.items() if k in (
        'passed','command','vx_mae_m_s','yaw_mae_rad_s','height_mae_m','stand_drift_max_m',
        'planar_speed_mean_m_s','full_horizon_episodes','requested_episodes','termination_reasons','checks')
        or 'settled' in k} for case_name,case in data['result']['cases'].items()}
    cohorts[name] = dict(evaluation_sha256=expected, cases=cases,
        passed_cases=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
        full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
        requested_episodes=sum(c['requested_episodes'] for c in cases.values()))
result = dict(schema='flat-low-gate1280-fixed-behavior-review-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=basic['checkpoint_sha256'],
    basic_completion_sha256=sha(basic_path),height_completion_sha256=sha(height_path),cohorts=cohorts,
    qualified=False,long_training_eligible=False,heldout_seed_used=False,hardware_motion_started=False,
    scope='Same1280 checkpoint and existing fixed development thresholds. Basic7/8 and any full-range failures remain failures; no long training or release is granted.')
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({n:{k:v for k,v in x.items() if k!='cases'} for n,x in cohorts.items()},indent=2))
