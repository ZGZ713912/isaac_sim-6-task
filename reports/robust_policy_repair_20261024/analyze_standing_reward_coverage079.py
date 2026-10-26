"""Offline coverage of existing velocity costs in authenticated physical traces.

This measures which recorded failures trigger the existing costs. It cannot
identify the causal controller/plant mechanism or prove a new reward will help.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(1)
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
B = ROOT / 'runs/robust_policy_repair_20261024'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


results = []
for name, folder in (
    ('019', 'diagnosis065_trained_quiet_full019'),
    ('021final32', 'diagnosis071_quiet021_final32'),
    ('075nominal', 'diagnosis075_nominal_dynamics'),
):
    run = B / folder
    report = json.loads((run / 'evaluation/report.json').read_text())
    meta = json.loads((run / 'evaluation/native_wheel_contact_1ms.json').read_text())
    cfg = json.loads((run / 'task_contract.json').read_text())['settings']
    assert sha(run / 'evaluation/trajectory.npz') == report['trajectory_sha256']
    assert sha(run / 'evaluation/native_wheel_contact_1ms.npz') == meta['sha256']
    trace = np.load(run / 'evaluation/trajectory.npz')
    data = np.load(run / 'evaluation/native_wheel_contact_1ms.npz')
    present = (np.linalg.norm(data['force'], axis=-1) > 2.).all(-1)
    missing = (~present).sum(1)
    leg_peak = trace['leg_speed_peak_rad_s']
    speed_limit = cfg['gap_leg_quality']['speed_soft_limit_rad_s']
    peak_density = cfg['gap_leg_quality']['speed_weight'] * np.maximum(leg_peak - speed_limit, 0.) ** 2
    quiet = ((np.abs(trace['commands'][..., :2]) < [.1, .2]).all(-1)
        & (trace['gravity'][..., 2] < -.94) & ~trace['passive'] & ~trace['reward_recovery_active'])
    terminal_density = cfg['quiet_motor_speed_cost'] * (trace['motor_velocity'][..., [0, 1, 3, 4]] ** 2).mean(-1) * quiet
    checkpoint = Path(json.loads((run / 'progress.json').read_text())['checkpoint'])
    clock = torch.load(checkpoint, map_location='cpu', weights_only=False)['infos']['completed_updates']
    writer = SummaryWriter(str(run / 'evaluation/metrics'), filename_suffix='.standing-reward-coverage-v1')
    rows = []
    for lane in report['lanes']:
        i = lane['lane']
        lost = int(missing[:, i].sum())
        row = dict(name=lane['name'], lost_physical_ms=lost,
            peak_speed_rad_s=float(leg_peak[:, i].max()),
            physical_speed_soft_limit_rad_s=float(speed_limit),
            missing_ms_with_zero_peak_velocity_cost=int((missing[:, i] * (peak_density[:, i] == 0.)).sum()),
            peak_velocity_cost_density_mean_per_s=float(peak_density[:, i].mean()),
            terminal_velocity_cost_density_mean_per_s=float(terminal_density[:, i].mean()),
            terminal_velocity_cost_density_max_per_s=float(terminal_density[:, i].max()),
            frames_above_6rad_s=int((leg_peak[:, i] > speed_limit).sum()))
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar('StandingRewardCoverage/' + lane['name'] + '/' + key, value, clock)
    writer.close()
    lost = int(missing.sum())
    results.append(dict(name=name, run=str(run.relative_to(ROOT)), clock=clock, rows=rows,
        lost_physical_ms=lost,
        lost_ms_with_zero_peak_speed_cost=int((missing * (peak_density == 0.)).sum()),
        peak_speed_rad_s=float(leg_peak.max()),
        peak_velocity_cost_density_max_per_s=float(peak_density.max()),
        terminal_velocity_cost_density_mean_per_s=float(terminal_density.mean()),
        trajectory_sha256=report['trajectory_sha256'], support_data_sha256=meta['sha256'],
        task_contract_sha256=sha(run / 'task_contract.json')))
out = dict(schema='standing-velocity-reward-coverage-v1', results=results,
    interpretation='Native1ms peak cost uses the existing6rad/s soft threshold; below-threshold motion still pays the existing end-of-policy-step quadratic velocity cost, plus other costs. This is coverage, not causal attribution or permission to ignore support loss.',
    no_policy_or_reward_changed=True, new_PPO=0, qualification=None, hardware_ready=False,
    script_sha256=sha(__file__), created_unix=time.time())
(R / 'standing_reward_coverage079.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps([dict(name=x['name'], lost_ms=x['lost_physical_ms'], zero_peak_cost_lost_ms=x['lost_ms_with_zero_peak_speed_cost'],
    peak_speed=x['peak_speed_rad_s'], terminal_speed_cost_density_mean=x['terminal_velocity_cost_density_mean_per_s']) for x in results], indent=2))
