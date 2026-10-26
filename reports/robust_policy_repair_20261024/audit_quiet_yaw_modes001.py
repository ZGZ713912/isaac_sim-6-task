"""Recorded mode decomposition; sensor/trajectory association is not causality."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
cp = B / 'quiet_stance050003/model_probe.pt'
mean, _ = load_mean(cp)
weights = {}
for name in ('low', 'mid', 'high'):
    w = mean.quiet_velocity[name].weight.detach().numpy()[2:]
    weights[name] = dict(wheels=w.tolist(), common_column_response=w.sum(-1).tolist(),
        differential_column_response=(w[:, 0] - w[:, 1]).tolist())
rows = []
for label, run in [('original', B / 'diagnosis038_quiet_stance050_development'),
                   ('gain3', B / 'diagnosis048_quiet_wheel_gain3_development')]:
    path = run / 'evaluation/trajectory.npz'
    report = json.loads((run / 'evaluation/report.json').read_text())
    with np.load(path) as z:
        for lane, record in enumerate(report['lanes']):
            if record['name'] not in ('stand_low__roll_plus2', 'stand__roll_plus2',
                                      'stand_high__pitch_minus2', 'stand_high__combined'):
                continue
            a, t = z['observations'][:, lane, 14:16] / 20., z['time'][:, lane]
            selected = np.flatnonzero((t > .2) & (np.abs(a).max(-1) < .245))
            assert len(selected) > 200 and z['commands'][:, lane, 1].max() == 0.
            angular = z['omega'][:, lane, 2]
            integrated = np.r_[0., np.cumsum(.5 * (angular[1:] + angular[:-1]) * np.diff(t))]
            differential = a[:, 0] - a[:, 1]
            first, last = selected[0], selected[-1]
            rows.append(dict(label=label, case=record['name'], trajectory_sha256=digest(path),
                unsaturated_frames=len(selected), integrated_body_z_angular_velocity_rad=float(integrated[last]-integrated[first]),
                wheel_coordinate_difference_change_m=float(differential[last]-differential[first]),
                differential_integrated_omega_correlation=float(np.corrcoef(differential[selected], integrated[selected])[0, 1]),
                coordinate_common_change_m=float(a[last].mean()-a[first].mean()),
                physical_XY_final_m=z['local_position'][last, lane, :2].tolist()))
    with SummaryWriter(log_dir=str(run), filename_suffix='.quiet_modes') as writer:
        for row in (x for x in rows if x['label'] == label):
            for key in ('integrated_body_z_angular_velocity_rad', 'wheel_coordinate_difference_change_m',
                        'differential_integrated_omega_correlation', 'coordinate_common_change_m'):
                writer.add_scalar('QuietMode/' + row['case'] + '/' + key, row[key], 1501)
out = dict(schema='quiet-wheel-mode-recorded-audit-v1', weights=weights, rows=rows,
    inference='Large differential coordinate accumulation accompanies actual unwanted yaw in zero-yaw commands. Existing wheel feedback strongly couples columns: the right-wheel differential response is almost cancelled and has the same sign as its common response. Test a common-row-sum-preserving diagonal allocation, instead of amplifying both modes together.',
    limits='Integrated body-z omega is not an exact global heading for a tilted robot. Correlation and actor weights do not establish a unique physical cause or closed-loop improvement. Counterfactual native test is still required.',
    proposed_test='Keep the sum of each wheel row and every leg/other weight. Set left wheel feedback to its own coordinate and right wheel feedback to its own coordinate. Same.5 authority and all gates; full51 development cases.',
    checkpoint=str(cp), checkpoint_sha256=digest(cp), new_PPO=0, qualification=None, created_unix=time.time())
write_json(R / 'quiet_yaw_modes_audit001.json', out)
print(json.dumps(dict(weights=weights, rows=rows), indent=2))
