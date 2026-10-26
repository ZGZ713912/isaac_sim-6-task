"""A single stronger smoothing coefficient; unchanged neural weights and ABI."""
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
from wheeled_algo.motion_target_damping import motion_damping_gate
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'motion_damping080002/model_probe.pt'
checkpoint = B / 'motion_damping060001/model_probe.pt'
old, oi = load_mean(source)
new, ni = load_mean(checkpoint)
a, b = [torch.load(path, map_location='cpu', weights_only=False) for path in (source, checkpoint)]
expected_info = deepcopy(a['infos'])
expected_info['settings']['motion_leg_damping_alpha'] = .6
checks = dict(all_saved_non_metadata_state_exact=exact_state_equal(
    {k: v for k, v in a.items() if k != 'infos'}, {k: v for k, v in b.items() if k != 'infos'}),
    only_alpha_metadata_changed=exact_state_equal(expected_info, b['infos']))
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation', 'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation', 'diagnosis024_entry_phase_fit_fixed/evaluation'):
    path = B / name / 'trajectory.npz'
    frames = np.load(path)['observations'].reshape(-1, 35)
    inactive = inactive_exact = quiet = quiet_exact = 0
    wheels_exact = finite = True
    change = error = before_power = after_power = 0.
    with torch.inference_mode():
        for values in np.array_split(frames, (len(frames) + 511) // 512):
            obs = torch.from_numpy(values.copy())
            x, y = old(obs), new(obs)
            gate = motion_damping_gate(obs).ravel()
            zero, stationary = gate == 0., (obs[:, 0] == 0.) & (obs[:, 2] == 0.)
            inactive += int(zero.sum())
            inactive_exact += int((x[zero] == y[zero]).all(-1).sum())
            quiet += int(stationary.sum())
            quiet_exact += int((x[stationary] == y[stationary]).all(-1).sum())
            wheels_exact &= torch.equal(x[:, 4:], y[:, 4:])
            finite &= bool(torch.isfinite(y).all())
            change = max(change, float(wrap((y[:, :4] - x[:, :4]) * .25).abs().max()))
            active = gate > 0.
            if active.any():
                error = max(error, float(wrap(y[active, :4] * .25 - obs[active, 10:14]).abs().max()))
            before_step, after_step = [wrap((v[:, :4] - obs[:, 22:26]) * .25) for v in (x, y)]
            normal = ((gate > .999) & (before_step.abs().amax(-1) < .03)
                & (wrap(x[:, :4] * .25 - obs[:, 10:14]).abs().amax(-1) < .04)
                & (wrap(y[:, :4] * .25 - obs[:, 10:14]).abs().amax(-1) < .04))
            before_power += float(before_step[normal].square().sum())
            after_power += float(after_step[normal].square().sum())
    rows.append(dict(trace=str(path), sha256=digest(path), frames=len(frames), inactive=inactive,
        inactive_exact=inactive_exact, quiet=quiet, quiet_exact=quiet_exact,
        wheels_exact=wheels_exact, finite=finite, target_change_peak_rad=change,
        active_target_error_peak_rad=error, step_RMS_ratio=(after_power / before_power)**.5 if before_power else None))
checks.update(inactive_and_quiet_exact=all(x['inactive'] == x['inactive_exact']
        and x['quiet'] == x['quiet_exact'] for x in rows),
    wheels_exact=all(x['wheels_exact'] for x in rows), finite=all(x['finite'] for x in rows),
    change_below_003_rad=all(x['target_change_peak_rad'] <= .03 for x in rows),
    existing_target_error_bound=all(x['active_target_error_peak_rad'] <= .12001 for x in rows),
    normal_step_reduced=all(x['step_RMS_ratio'] is None or x['step_RMS_ratio'] <= .755 for x in rows))
out = dict(schema='motion-damping-coefficient-preflight-v1', checkpoint=str(checkpoint),
    sha256=digest(checkpoint), source=str(source), source_sha256=digest(source),
    alpha_before=.8, alpha_after=.6, checks=checks, rows=rows,
    hypothesis='A further bounded reduction of motion leg target increments may prevent the short support breaks that interrupt strict post-recovery tracking.',
    gate=dict(standing_at_least=19, motion_MAE_at_most=.135, entry_recovery_at_least=8,
        entry_followup_at_least=8, fallen_recovery_at_least=4, fallen_followup_at_least=2,
        entry_peak_at_most_rad_s=8.911178255081177, fallen_peak_at_most_rad_s=24.11665153503418,
        impacts_at_most=1, improvement_required='entry_followup>=9 OR fallen_followup>=3 OR fallen_recovery>=5',
        no_PPO=True),
    new_PPO=0, actual_experiment_consumed=2605, qualification=None,
    hardware_started=False, created_unix=time.time())
(R / 'motion_damping060_preflight001.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=2), flush=True)
assert all(checks.values()), checks
