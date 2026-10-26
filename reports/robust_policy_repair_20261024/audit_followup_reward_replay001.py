"""Quantify a training reward hypothesis on existing traces, never qualify it."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from train_robust_getup import digest, write_json
from wheeled_tasks.chassis.followup_continuity_reward import bonus_from_hold, contract

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
models = []
for label, directory, clock in [('best9a', 'diagnosis025_motion_damping080_fixed', 1501),
                                ('motion015', 'pilot015_motion_fixed', 1509)]:
    path = B / directory / 'evaluation'
    report = json.loads((path / 'report.json').read_text())
    trace = path / 'trajectory.npz'
    assert digest(trace) == report['trajectory_sha256']
    with np.load(trace) as archive:
        z = {key: archive[key] for key in ('time', 'alive', 'passive', 'gravity', 'height',
            'recovered', 'reward_recovery_active', 'stable', 'velocity', 'commands', 'omega',
            'followup_success', 'native_reward')}
    n = len(report['lanes'])
    hold = np.zeros(n, dtype=np.float32)
    latched = np.zeros(n, dtype=bool)
    holds, bonuses = [], []
    all_latches = True
    for j in range(len(z['time'])):
        active = z['alive'][j] & ~z['passive'][j]
        fallen = active & ((z['gravity'][j, :, 2] > -.7) | (z['height'][j] < .13))
        follow = (active & z['recovered'][j] & ~z['reward_recovery_active'][j].astype(bool)
            & z['stable'][j] & (np.abs(z['velocity'][j, :, 0] - z['commands'][j, :, 0]) < .25)
            & (np.abs(z['omega'][j, :, 2] - z['commands'][j, :, 1]) < .4))
        # The saved first row can be an initial observation, not an elapsed step.
        elapsed_step = z['time'][j] > 0.
        hold = np.where(follow & elapsed_step, hold + np.float32(.005), np.float32(0.))
        latched[fallen] = False
        latched |= hold >= 2.
        all_latches &= np.array_equal(latched[active], z['followup_success'][j, active])
        bonus = bonus_from_hold(torch.from_numpy(hold), torch.from_numpy(active), weight=5.).numpy()
        holds.append(hold.copy())
        bonuses.append(bonus)
    holds, bonuses = np.array(holds), np.array(bonuses)
    rows = []
    writer = SummaryWriter(str(path / 'metrics'), filename_suffix='.followup-reward-hypothesis')
    for lane in report['lanes']:
        i = lane['lane']
        active = z['alive'][:, i] & ~z['passive'][:, i]
        row = dict(name=lane['name'], recovery=lane['recovery'], recovered=lane['recovered'],
            followup=lane['followup_success'], survived=lane['survived'],
            max_native_timer_reconstructed_seconds=float(holds[:, i].max()),
            original_native_reward_sum=float(z['native_reward'][active, i].sum()),
            hypothetical_continuity_bonus_sum=float(bonuses[:, i].sum()),
            bonus_mean_per_active_second=float(bonuses[:, i].sum() / max(.005, active.sum() * .005)),
            bonus_step_peak=float(bonuses[:, i].max()))
        rows.append(row)
        for key in ('original_native_reward_sum', 'hypothetical_continuity_bonus_sum',
                    'max_native_timer_reconstructed_seconds'):
            writer.add_scalar('FollowupRewardReplay/' + row['name'] + '/' + key, row[key], clock)
    writer.close()
    checks = dict(native_latched_success_exact=all_latches, step_bonus_bounded=float(bonuses.max()) <= .025000001,
        ordinary_unrecovered_bonus_zero=all(x['hypothetical_continuity_bonus_sum'] == 0.
            for x in rows if not x['recovery'] and not x['recovered']))
    models.append(dict(label=label, trace=str(trace), trace_sha256=digest(trace), checks=checks, rows=rows))
changes = []
for a, b in zip(models[0]['rows'], models[1]['rows'], strict=True):
    if a['followup'] != b['followup']:
        changes.append(dict(name=a['name'], original_followup=a['followup'], motion015_followup=b['followup'],
            original_hypothetical_bonus=a['hypothetical_continuity_bonus_sum'],
            motion015_hypothetical_bonus=b['hypothetical_continuity_bonus_sum']))
result = dict(schema='continuous-followup-training-reward-replay-v1', models=models, changed_followup_rows=changes,
    contract=contract(5.), checks={f'{m["label"]}/{k}': bool(v) for m in models for k, v in m['checks'].items()},
    hypothesis='Native squared hold density rewards continuous recovery-followup; it must earn actual target gains and unchanged nonregression gates in a bounded native trial.',
    limits='Counterfactual reward calculation on recorded states only. It does not prove PPO credit, causal benefit, moving-fall recovery or closed-loop qualification.',
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'followup_reward_replay001.json', result)
print(json.dumps(dict(checks=result['checks'], changed_followup_rows=changes), indent=2))
assert all(result['checks'].values()), result['checks']
