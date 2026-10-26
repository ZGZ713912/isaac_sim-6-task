"""Two-weight, zero-PPO test of the high-stance displacement feedback sign."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest
from wheeled_algo.robust_getup import quiet_gate
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'final_pair_circular_corrected002/model_migrated.pt'
original = torch.load(source, map_location='cpu', weights_only=False)
result = deepcopy(original)
key = 'mlp.mean.quiet_velocity.high.weight'
assert result['actor_state_dict'][key].shape == (4, 2)
for i, j in ((2, 0), (3, 1)):
    result['actor_state_dict'][key][i, j].neg_()
result['infos']['high_wheel_diagonal_probe'] = dict(schema='high-wheel-diagonal-sign-probe-v1',
    source=str(source.resolve()), source_sha256=digest(source), tensor=key,
    changed_entries=[[2, 0], [3, 1]], factor=-1., new_PPO=0,
    hypothesis='Test the existing high-expert wheel/displacement diagonal signs; sensor derivatives alone do not establish physical closed-loop stability.')
result['infos']['hardware_ready'] = False
checkpoint = B / 'high_wheel_diagonal_restoring_probe001/model_probe.pt'
checkpoint.parent.mkdir(parents=True, exist_ok=False)
torch.save(result, checkpoint)
old, _ = load_mean(source)
new, _ = load_mean(checkpoint)
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation', 'diagnosis015_development21_baseline/evaluation'):
    path = B / name / 'trajectory.npz'
    frames = np.load(path)['observations'].reshape(-1, 35)
    legs_exact = outside_exact = True
    wheel_peak = 0.
    with torch.inference_mode():
        for values in np.array_split(frames, (len(frames) + 511) // 512):
            obs = torch.from_numpy(values.copy())
            a, b = old(obs), new(obs)
            eligible = (quiet_gate(obs).ravel() > 0.) & (obs[:, 3] > np.float32(.305) * 5.)
            legs_exact &= torch.equal(a[:, :4], b[:, :4])
            outside_exact &= torch.equal(a[~eligible], b[~eligible])
            wheel_peak = max(wheel_peak, float((a[:, 4:] - b[:, 4:]).abs().max()))
    rows.append(dict(trace=str(path), sha256=digest(path), frames=len(frames),
        legs_exact=legs_exact, outside_high_quiet_exact=outside_exact, wheel_target_delta_peak_nm=wheel_peak))
checks = dict(only_one_tensor_changed=all(torch.equal(v, result['actor_state_dict'][k])
    for k, v in original['actor_state_dict'].items() if k != key),
    common_leg_and_off_diagonal_weights_exact=torch.equal(original['actor_state_dict'][key][:2], result['actor_state_dict'][key][:2])
        and original['actor_state_dict'][key][2, 1] == result['actor_state_dict'][key][2, 1]
        and original['actor_state_dict'][key][3, 0] == result['actor_state_dict'][key][3, 0],
    critic_Adam_and_clock_exact=all(exact_state_equal(result[k], original[k])
        for k in ('critic_state_dict', 'optimizer_state_dict', 'iter')),
    recorded_legs_and_non_high_quiet_outputs_exact=all(x['legs_exact'] and x['outside_high_quiet_exact'] for x in rows),
    bounded_wheel_delta=all(x['wheel_target_delta_peak_nm'] <= 1. for x in rows))
checks = {k: bool(v) for k, v in checks.items()}
out = dict(schema='high-wheel-diagonal-zero-PPO-preflight-v1', checkpoint=str(checkpoint),
    sha256=digest(checkpoint), source=str(source), source_sha256=digest(source), checks=checks, rows=rows,
    gates=dict(development_high_all7=True, development_no_lost_previous_pass=True,
        high_wobble_XY_at_most_m=.1, high_wobble_tail_stable_at_least=.95,
        fixed_high_all7_required_before_composition=True, closure_at_most_m=.003,
        passive_motor_peak_nm=0., no_PPO=True),
    scope='A separate quiet component diagnosis, not a candidate replacing the successful entry policy.',
    new_PPO=0, actual_experiment_consumed=2605, qualification=None,
    hardware_started=False, created_unix=time.time())
(R / 'high_diagonal_restoring_preflight001.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
assert all(checks.values())
