"""Locate first-episode failures in the completed1328 new-material RMCS replay."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

directory = ROOT / 'runs/flat_base_20261019/ground025_parent1328_default_envelope001'
report_path = directory / 'evaluation.json'
trace_path = directory / 'traces.npz'
target = HERE / 'static_grip_candidate002/parent_default_dynamics001.json'
if target.exists():
    raise SystemExit('Preserve the existing parent dynamics audit')
report = json.loads(report_path.read_text())
assert report['status'] == 'evaluated' and report['new_optimizer_updates'] == 0
material = report['startup']['contact_material']
assert material['ground_static_range'] == [.2, .5] and material['wheel_static_friction'] == .1
cases = report['result']['cases']
lane_rows = []
with np.load(trace_path) as trace:
    groups, active, done = trace['groups'], trace['active'].astype(bool), trace['done'].astype(bool)
    for j, group in enumerate(groups):
        indices = np.flatnonzero(active[:, j])
        if not len(indices):
            raise ValueError('Missing first-episode trace')
        ends = np.flatnonzero(active[:, j] & done[:, j])
        i = int(ends[0]) if len(ends) else int(indices[-1])
        terminal_seconds = float(trace['episode_ticks'][i, j]) / report['policy_hz']
        settings = next(row for row in report['settings']['cases'] if row['name'] == str(group))
        early = terminal_seconds < settings['episode_seconds'] - 1 / report['policy_hz']
        if not early:
            continue
        window = np.arange(max(0, i - 10), i + 1)
        obs = trace['observations'][window, j]
        body_vx = trace['velocity'][window, j, 0]
        encoder_forward = (obs[:, 20] - obs[:, 21]) * .3
        gravity = trace['gravity'][i, j]
        ground = material['ground_static_friction_per_environment'][j]
        lane_rows.append(dict(case=str(group), lane=j, terminal_seconds=terminal_seconds,
            ground_material_mu=ground, wheel_ground_average_mu=(ground + .1) / 2,
            body_vx_terminal_m_s=float(body_vx[-1]), encoder_forward_terminal_m_s=float(encoder_forward[-1]),
            encoder_body_difference_peak_previous200ms_m_s=float(np.max(np.abs(encoder_forward - body_vx))),
            command_terminal=trace['commands'][i, j].tolist(),
            tilt_terminal_deg=float(np.degrees(np.arccos(np.clip(-gravity[2], -1., 1.)))),
            case_aggregate_termination_reasons={key:value for key,value in cases[str(group)]['termination_reasons'].items() if value}))
reasons = {}
for row in cases.values():
    for key, value in row['termination_reasons'].items():
        reasons[key] = reasons.get(key, 0) + value
assert len(lane_rows) == 88 - sum(row['full_horizon_episodes'] for row in cases.values())
target.write_bytes(encode(dict(schema='flat-ground025-parent-dynamics-audit-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), evaluation_sha256=digest(report_path),
    traces_sha256=digest(trace_path), checkpoint_sha256=report['checkpoint_sha256'],
    passed_cases=sum(row['passed'] for row in cases.values()), requested_cases=len(cases),
    full_horizon_episodes=sum(row['full_horizon_episodes'] for row in cases.values()), requested_episodes=88,
    termination_reasons=reasons, early_episode_lanes=lane_rows,
    inference='Several ordinary direct-command trajectories terminate before braking. A braking-only explanation is insufficient.',
    encoder_difference_scope='Mean forward wheel-encoder rim speed versus body-frame root velocity; not a measured contact slip velocity. Wheel lift, leg motion, and body rotation are not removed.',
    candidate1338_comparison_available=False, new_optimizer_updates=0, qualified=False,
    reader_sha256=digest(Path(__file__)),
    scope='Completed parent checkpoint at the new material domain; cannot attribute any failure to the subsequent static-only update')))
print(json.dumps(dict(early_episodes=len(lane_rows), termination_reasons=reasons,
    first_second_failures=[row['case'] for row in lane_rows if row['terminal_seconds'] <= 1.])))
