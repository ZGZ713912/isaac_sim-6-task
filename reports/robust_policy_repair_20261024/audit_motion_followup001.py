"""Explain fallen-output fixed failures using the unchanged native predicates."""
import json
import math
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R / 'plan015_motion_output001.json').read_text())
assert all(digest(Path(n)) == h for n, h in plan['source_sha256'].items())
assert json.loads((B / 'round015/progress.json').read_text())['status'] == 'review_ready'
write_json(R / 'round015_followup_source_confirmation001.json', dict(schema='closed-bounded-round-source-seal-v1',
    round='015', source_files=len(plan['source_sha256']), all_frozen_sources_verified=True,
    decision_sha256=digest(R / 'motion_round015_fixed_decision001.json'),
    attempted=8, completed=8, actual_experiment_consumed=2646, qualification=None, created_unix=time.time()))

def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float((ends - starts).max() * .005) if len(starts) else 0.

models = []
for label, directory, clock in [('best9a', 'diagnosis025_motion_damping080_fixed', 1501),
                                ('motion015', 'pilot015_motion_fixed', 1509)]:
    path = B / directory / 'evaluation'
    report = json.loads((path / 'report.json').read_text())
    trace = path / 'trajectory.npz'
    assert digest(trace) == report['trajectory_sha256']
    z = np.load(trace)
    writer = SummaryWriter(str(path / 'metrics'), filename_suffix='.fallen-followup-diagnosis')
    rows = []
    for lane in report['lanes']:
        if not lane['recovery']:
            continue
        i = lane['lane']
        t = z['time'][:, i]
        active = z['alive'][:, i] & ~z['passive'][:, i]
        ever = z['recovered'][:, i]
        ids = np.flatnonzero(ever & active)
        recovered_s = float(t[ids[0]]) if len(ids) else None
        conditions = dict(tilt=z['gravity'][:, i, 2] < -math.cos(math.radians(15)),
            angular=np.linalg.norm(z['omega'][:, i, :2], axis=-1) < .7,
            height=z['height'][:, i] > .19, contact=z['wheel_contact'][:, i].all(-1),
            leg_speed=np.abs(z['motor_velocity'][:, i][:, [0, 1, 3, 4]]).max(-1) < 3.,
            vx=np.abs(z['velocity'][:, i, 0] - z['commands'][:, i, 0]) < .25,
            yaw=np.abs(z['omega'][:, i, 2] - z['commands'][:, i, 1]) < .4)
        reconstructed = np.logical_and.reduce(list(conditions.values())) & ~z['passive'][:, i]
        assert np.array_equal(reconstructed, z['stable'][:, i] & conditions['vx'] & conditions['yaw'])
        released = ~z['reward_recovery_active'][:, i].astype(bool)
        follow = active & ever & released & reconstructed
        post = active & (t >= max(4., (recovered_s or 8.) + .5))
        contact_break = post & ~conditions['contact']
        peak_values = np.where(active, z['leg_speed_peak_rad_s'][:, i], -1.)
        peak = int(peak_values.argmax())
        row = dict(name=lane['name'], recovered=lane['recovered'], followup=lane['followup_success'],
            survived=lane['survived'], recovered_s=recovered_s, longest_follow_s=longest(follow),
            post_frames=int(post.sum()), post_failure_frames={k: int((post & ~v).sum()) for k, v in conditions.items()},
            contact_break_clearance_peak_m=float(z['wheel_clearance_peak'][contact_break, i].max()) if contact_break.any() else None,
            contact_break_air_run_peak_s=float(z['wheel_air_run_peak'][contact_break, i].max()) if contact_break.any() else None,
            native_leg_speed_peak_rad_s=float(peak_values[peak]), native_leg_peak_time_s=float(t[peak]),
            same_step_endpoint_leg_speed_peak_rad_s=float(np.abs(z['motor_velocity'][peak, i, [0, 1, 3, 4]]).max()))
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar('FollowupDiagnosis/' + row['name'] + '/' + key, value, clock)
        for key, value in row['post_failure_frames'].items():
            writer.add_scalar('FollowupDiagnosis/' + row['name'] + '/failure_frames/' + key, value, clock)
    writer.close()
    models.append(dict(label=label, trace=str(trace), sha256=digest(trace), rows=rows))
changes = []
for a, b in zip(models[0]['rows'], models[1]['rows'], strict=True):
    assert a['name'] == b['name']
    if (a['recovered'], a['followup']) != (b['recovered'], b['followup']):
        changes.append(dict(name=a['name'], before=a, after=b))
out = dict(schema='motion-output-fixed-followup-audit-v1', models=models, changed_rows=changes,
    scope='Exact recorded predicates, released recovery state and physical substep speed maxima; no force/contact/duration or other gate changed. No unique causal claim.',
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'motion_round015_followup_audit001.json', out)
print(json.dumps(dict(changed_rows=changes), indent=2))
