"""Read the completed native counterfactuals; never construct a policy or optimizer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
DIRECTORY = BASE/REL/'fresh_height_start_native001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt = read(DIRECTORY/'receipt.json')
result = read(DIRECTORY/'result.json')
run = Path(receipt['run'])
event, = DIRECTORY.glob('events.out.tfevents.*')
event_bytes = event.read_bytes()
with tempfile.TemporaryDirectory() as temporary:
    frozen = Path(temporary)/event.name
    frozen.write_bytes(event_bytes)
    events = EventAccumulator(str(frozen), size_guidance={'scalars': 0}).Reload()
    required = [f'HeightStartDiagnostic/{arm}/h{height}/{metric}'
        for arm in receipt['counterfactuals'] for height in (230, 305, 420)
        for metric in ('tilt_mean_deg', 'height_mae_m')]
    logging_checks = {tag: [row.step for row in events.Scalars(tag)] == list(range(200, 1601, 200))
        and all(np.isfinite(row.value) for row in events.Scalars(tag)) for tag in required}
with np.load(DIRECTORY/'native_trace.npz', allow_pickle=False) as trace:
    gravity, valid, times = (trace[name] for name in ('gravity', 'valid', 'time'))
    tilt = np.degrees(np.arccos(np.clip(-gravity[:, :, 2], -1., 1.)))
    groups = {}
    for arm in receipt['counterfactuals']:
        groups[arm] = {}
        for height in (.23, .305, .42):
            lanes = [i for i, row in enumerate(result['rows'])
                if row['case']['arm'] == arm and row['case']['command'][2] == height]
            first_crossings = {}
            for lane in lanes:
                above = np.flatnonzero(valid[:, lane] & (tilt[:, lane] >= 10.))
                first_crossings[str(result['rows'][lane]['case']['pitch_deg'])] = (
                    float(times[above[0], lane]) if above.size else None)
            rows = [result['rows'][lane] for lane in lanes]
            groups[arm][str(height)] = dict(
                terminal_tilt_mean_deg=float(np.mean([row['terminal_tilt_deg'] for row in rows])),
                terminal_height_mean_m=float(np.mean([row['terminal_height_m'] for row in rows])),
                true_leg_speed_peak_rad_s=max(row['leg_speed_physical_peak_rad_s'] for row in rows),
                native_guard_cases=sum(row['native_guard'] for row in rows),
                first_10deg_time_s_by_start_pitch_deg=first_crossings)
checks = dict(
    diagnostic_process_exited=not Path(f"/proc/{receipt['pid']}").exists(),
    completed_all168_cases=receipt['status'] == 'completed_unqualified'
        and receipt['case_count'] == len(result['rows']) == 168,
    real1600_policy8000_physics=receipt['native_policy_steps'] == 1600 and receipt['physical_steps'] == 8000,
    zero_PPO_no_optimizer=receipt['new_PPO'] == 0 and receipt['optimizer_constructed'] is False,
    parameters_unchanged=receipt['actor_parameters_unchanged'] is True,
    contemporaneous_sources_verified=receipt['all_current_sources_unchanged'] is True,
    frozen272_sources_match= len(receipt['source_sha256']) == 272 and all(
        sha(run/'source_snapshot'/name) == expected for name, expected in receipt['source_sha256'].items()),
    exact_own_diagnostic_checkpoints=all(sha(run/name) == expected for name, expected in receipt['checkpoints'].items()),
    trace_and_result_integrity=sha(DIRECTORY/'native_trace.npz') == receipt['trace_sha256']
        and sha(DIRECTORY/'result.json') == receipt['result_sha256'],
    diagnostic_source_unchanged=sha(ROOT/'scripts/diagnose_fresh_height_start.py') == receipt['diagnostic_source_sha256'],
    actual_nominal_population=not any(receipt['actual_domain_samples']['samples']['enabled']),
    actual_physics=receipt['physics_runtime']['physical_dt_s'] == .001
        and receipt['physics_runtime']['policy_dt_s'] == .005
        and receipt['physics_runtime']['articulation_iterations']['physxArticulation:solverPositionIterationCount'] == 64
        and receipt['physics_runtime']['articulation_iterations']['physxArticulation:solverVelocityIterationCount'] == 32,
    all48_RSL_tags_complete=all(logging_checks.values()) and len(logging_checks) == 48,
)
assert all(checks.values()), checks
output = dict(schema='fresh-height-start-native-verification-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(), checks=checks, all_checks_pass=True,
    receipt=str(DIRECTORY/'receipt.json'), receipt_sha256=sha(DIRECTORY/'receipt.json'),
    trace_sha256=receipt['trace_sha256'], result_sha256=receipt['result_sha256'],
    final_normalizer_counts=receipt['final_normalizer_counts'], groups=groups,
    RSL_event=dict(path=str(event), bytes=len(event_bytes), sha256=hashlib.sha256(event_bytes).hexdigest(),
        required_tag_count=len(required), step_unit='actual policy interval; not a PPO update'),
    inference='Neither updating/fixing statistics nor removing the initial MLP residual solves the failed foundation. '
        'No isolated normalization root cause is established. Static-load targets do not balance an inverted pendulum.',
    next_single_change='Restore failed-standing termination only for pure-standing upright training births; '
        'leave initial/moving recovery and fixed-evaluation horizons intact. Start a new actor/critic/empty Adam.',
    new_PPO=0, qualified=False, hardware_ready=False, producer_sha256=sha(Path(__file__)))
for directory in (ROOT/REL, BASE/REL):
    (directory/'fresh_height_start_native001_verified.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
(BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(all_checks_pass=True, checks=len(checks), cases=168, native_policy_steps=1600, new_PPO=0)))
