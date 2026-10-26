"""Offline reconstruction of one recorded physical reward component only."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
output = BASE / 'sampled_cost_audit001.json'
if output.exists():
    raise ValueError('Retain the first sealed audit')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = dict(schema='recorded-gap-physical-cost-audit-v1',
    scope='Exact physical leg-quality density component reconstructed from recorded 1 ms aggregates. This is not total reward or a unique causal explanation.',
    rows=[], new_PPO=0, qualification=None)
for label in ('mean', 'sample_original', 'sample_noise015'):
    run = ROOT / ('runs/robust_policy_repair_20261024/diagnosis002_v7_' + label)
    progress = json.loads((run / 'progress.json').read_text())
    if progress['status'] != 'completed':
        raise ValueError('Require complete exposure: ' + label)
    report = json.loads((run / 'evaluation/report.json').read_text())
    path = run / 'evaluation/trajectory.npz'
    if sha(path) != report['trajectory_sha256']:
        raise ValueError('Native trajectory changed')
    contract = json.loads((run / 'task_contract.json').read_text())
    cfg = contract['settings']['gap_leg_quality'] if 'settings' in contract else contract['gap_leg_quality']
    trace = np.load(path)
    active = trace['alive'] & ~trace['passive'] & ~trace['enable_event']
    speed = cfg['speed_weight'] * np.maximum(trace['leg_speed_peak_rad_s'] - cfg['speed_soft_limit_rad_s'], 0.) ** 2
    saturation = cfg['saturation_weight'] * trace['leg_saturation_fraction']
    slew = cfg['slew_weight'] * (trace['leg_effort_slew_rms_nm'] / 10.) ** 2
    components = dict(speed=speed, saturation=saturation, slew=slew)
    density = speed + saturation + slew
    if not np.isfinite(density).all():
        raise ValueError('Nonfinite physical cost')
    dt = .005
    writer = SummaryWriter(str(run / 'evaluation/metrics'), filename_suffix='.physical-cost-audit')
    row = dict(label=label, trajectory_sha256=sha(path), dt_s=dt,
        active_policy_intervals=int(active.sum()), active_policy_seconds=float(active.sum() * dt),
        components={})
    # Learning clock 1501 is shared by these zero-PPO exposures.
    for name, values in components.items():
        row['components'][name] = dict(mean_density=float(values[active].mean()),
            integrated_paid_cost=float((values * active).sum() * dt))
        writer.add_scalar('ExposurePhysicalCost/' + name + '_mean_density', row['components'][name]['mean_density'], 1501)
    row['all_leg_quality_mean_density'] = float(density[active].mean())
    writer.add_scalar('ExposurePhysicalCost/all_mean_density', row['all_leg_quality_mean_density'], 1501)
    for index, lane in enumerate(report['lanes']):
        selected = active[:, index]
        for name, values in components.items():
            if selected.any():
                writer.add_scalar('ExposurePhysicalCost/' + lane['name'] + '/' + name + '_mean_density', float(values[selected, index].mean()), 1501)
    writer.close()
    result['rows'].append(row)
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
