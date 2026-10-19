"""Read the full original horizons; retain static failures in the gate-only trial."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
OUT = HERE / 'full_gate1264_basic_review001.json'
if OUT.exists():
    raise SystemExit('Preserve the completed full-horizon gate review')
original_path = RUN / 'deployment1264_basic001/evaluation.json'
candidate_path = RUN / 'full_gate1264_basic001/evaluation.json'
old, new = read(original_path), read(candidate_path)
if new['status'] != 'evaluated':
    raise SystemExit('Wait for the complete original case horizons')
z = {label: np.load(RUN / name / 'traces.npz') for label, name in
     (('old', 'deployment1264_basic001'), ('new', 'full_gate1264_basic001'))}
before, after = z['old'], z['new']
common = before['active'].astype(bool) & after['active'].astype(bool)
static_lanes = np.char.startswith(before['groups'], 'stand_')
static_active = common & static_lanes[None, :]
checks = dict(same_parent_checkpoint=old['checkpoint_sha256'] == new['checkpoint_sha256'],
    same_original_case_horizons_and_thresholds=old['settings'] == new['settings'],
    same50Hz1000Hz=new['policy_hz'] == old['policy_hz'] == 50 and old['physics_hz'] == new['physics_hz'] == 1000,
    same_seed=new['seed'] == old['seed'] == 191029,
    same_group_layout=np.array_equal(before['groups'], after['groups']),
    same_initial_observations=np.array_equal(before['observations'][0], after['observations'][0]),
    same_command_inputs_on_common_active_frames=np.array_equal(before['observations'][common][:,:4], after['observations'][common][:,:4]),
    static_actions_and_velocity_bit_exact=np.array_equal(before['actions'][static_active], after['actions'][static_active])
        and np.array_equal(before['velocity'][static_active], after['velocity'][static_active]),
    no_native_failure_or_truncation=all(not any(c['termination_reasons'].values())
        and c['full_horizon_episodes'] == c['requested_episodes'] for c in new['result']['cases'].values()),
    still_explicitly_ineligible=new['result']['diagnostic_only'] and not new['result']['qualification_eligible']
        and not new['result']['passed'] and not new['diagnostic_counterfactual']['qualification_eligible'],
    zero_optimizer_or_checkpoint_writes=new['new_optimizer_updates'] == 0
        and not new['diagnostic_counterfactual']['checkpoint_saved'])
checks = {k:bool(v) for k,v in checks.items()}
metrics = {}
for name, case in new['result']['cases'].items():
    metrics[name] = {label:{key:row[key] for key in ('passed','vx_mae_m_s','yaw_mae_rad_s',
        'stand_drift_max_m','settled_stop_speed','full_horizon_episodes','checks')}
        for label, row in (('old', old['result']['cases'][name]), ('gate_floor', case))}
for values in z.values():
    values.close()
result = dict(schema='flat-full-horizon-low-gate1264-review-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=all(checks.values()), checks=checks,
    checkpoint_sha256=new['checkpoint_sha256'], candidate_evaluation_sha256=sha(candidate_path),
    original_evaluation_sha256=sha(original_path),
    trace_sha256={name:sha(RUN/name/'traces.npz') for name in ('deployment1264_basic001','full_gate1264_basic001')},
    metrics=metrics, native_steps_performed_by_reader=0, optimizer_updates=0, behavior_qualified=False,
    scope='Original10s static/motion and16s start-stop horizons; gate-only fixed-weight comparison. All static drift failures and deployment envelope/heldout requirements remain.')
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=result['passed'], checks=checks, metrics=metrics),indent=2))
if not result['passed']:
    raise SystemExit('Full-horizon pairing proof failed; preserve the negative review')
