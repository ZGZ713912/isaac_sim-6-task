"""Check causal paired constraints and the actual native-evaluation wrapper."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import time
import types

sys.path[:0] = ['src', 'scripts', 'reports/robust_policy_repair_20261024']
import numpy as np
import torch
import export_robust_getup as export
from train_robust_getup import digest, write_json
from wheeled_algo.motion_target_damping import motion_damping_gate
from wheeled_algo.paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap
from motion_knee_damping_probe001 import MotionKneeDamping

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
cp = B/'motion_damping080002/model_probe.pt'
reference, _ = export.load_mean(cp)
initial_state = deepcopy(reference.state_dict())
probe = MotionKneeDamping(reference)
zero = MotionKneeDamping(deepcopy(reference), alpha=1.)
rows = []
first = None
with torch.inference_mode():
    for name in ('diagnosis025_motion_damping080_fixed/evaluation',
                 'diagnosis026_motion_damping080_rmcs/evaluation',
                 'diagnosis039_current9a_development/evaluation'):
        trace = B/name/'trajectory.npz'
        frames = np.load(trace)['observations'].reshape(-1, 35)
        if first is None:
            first = torch.from_numpy(frames[:2048].copy())
        row = dict(trace=str(trace), sha256=digest(trace), frames=len(frames),
            inactive_frames=0, inactive_different=0, quiet_different=0, wheels_different=0,
            zero_different=0, changed_frames=0, common_change_peak_rad=0.,
            active_error_peak_rad=0., bounds_excess_peak_rad=0.,
            change_peak_rad=0., knee_step_increase_peak_rad=0., action_bound_excess_peak=0.)
        for array in np.array_split(frames, (len(frames)+511)//512):
            obs = torch.from_numpy(array.copy())
            old, out, neutral = reference(obs), probe(obs), zero(obs)
            assert torch.isfinite(out).all()
            gate = motion_damping_gate(obs).ravel()
            inactive = gate == 0.
            quiet = (obs[:, 0] == 0.) & (obs[:, 2] == 0.)
            row['inactive_frames'] += int(inactive.sum())
            row['inactive_different'] += int((old[inactive] != out[inactive]).any(-1).sum())
            row['quiet_different'] += int((old[quiet] != out[quiet]).any(-1).sum())
            row['wheels_different'] += int((old[:, 4:] != out[:, 4:]).any(-1).sum())
            row['zero_different'] += int((old != neutral).any(-1).sum())
            changed = (old != out).any(-1)
            row['changed_frames'] += int(changed.sum())
            a, b = [(x[:, :4]*.25).unflatten(-1, (2, 2)) for x in (old, out)]
            row['common_change_peak_rad'] = max(row['common_change_peak_rad'], float((b.mean(-1)-a.mean(-1)).abs().max()))
            row['change_peak_rad'] = max(row['change_peak_rad'], float((b-a).abs().max()))
            if (gate > 0.).any():
                row['active_error_peak_rad'] = max(row['active_error_peak_rad'], float(wrap(out[gate>0., :4]*.25-obs[gate>0., 10:14]).abs().max()))
            row['action_bound_excess_peak'] = max(row['action_bound_excess_peak'], float((out.abs()-reference.action_bound).clamp_min(0.).max()))
            if changed.any():
                sign, nominal, bounds = b.new_tensor([1., -1.]), b.new_tensor(NOMINAL_DELTA), b.new_tensor(DELTA_BOUNDS)
                delta = sign*(nominal+b[changed, :, 1]-b[changed, :, 0])
                row['bounds_excess_peak_rad'] = max(row['bounds_excess_peak_rad'], float(torch.maximum(bounds[:, 0]-delta, delta-bounds[:, 1]).clamp_min(0.).max()))
                previous = (obs[:, 22:26]*.25).unflatten(-1, (2, 2))
                da, db = wrap(a-previous), wrap(b-previous)
                ka, kb = [.5*(d[..., 1]-d[..., 0]) for d in (da, db)]
                row['knee_step_increase_peak_rad'] = max(row['knee_step_increase_peak_rad'], float((kb[changed].abs()-ka[changed].abs()).clamp_min(0.).max()))
        rows.append(row)

calls = []
module = types.ModuleType('evaluate_gap_robustness')
module.digest = digest
module.write_json = lambda p, v: calls.append((str(p), deepcopy(v)))
native = export.load_mean
outdir = R/'motion_knee_wrapper_smoke001'
outdir.mkdir(exist_ok=False)
def main():
    actual, _ = export.load_mean(cp)
    with torch.inference_mode():
        assert torch.equal(actual(first), probe(first))
    module.write_json(outdir/'task_contract.json', dict(evaluation_overrides={}))
    module.write_json(outdir/'progress.json', dict(source_sha256={}, evaluation_overrides={}))
module.main = main
sys.modules['evaluate_gap_robustness'] = module
wrapper = R/'evaluate_motion_knee_probe001.py'
sys.argv = [str(wrapper), '--checkpoint', str(cp), '--output', str(outdir)]
runpy.run_path(str(wrapper), run_name='__main__')
runtime = next(v for p, v in calls if p.endswith('runtime_function_readback.json'))
progress = next(v for p, v in calls if p.endswith('progress.json'))
checks = dict(zero_alpha_endpoint_exact=all(r['zero_different']==0 for r in rows),
    inactive_and_quiet_exact=all(r['inactive_different']==r['quiet_different']==0 for r in rows),
    wheels_exact=all(r['wheels_different']==0 for r in rows),
    common_rotation_retained=all(r['common_change_peak_rad']<=2e-6 for r in rows),
    native_target_error_limit=all(r['active_error_peak_rad']<=.12001 for r in rows),
    paired_geometry=all(r['bounds_excess_peak_rad']<=2e-6 for r in rows),
    action_bounds=all(r['action_bound_excess_peak']==0. for r in rows),
    knee_increment_not_increased=all(r['knee_step_increase_peak_rad']<=2e-6 for r in rows),
    nonempty_intervention=all(r['changed_frames']>0 for r in rows),
    source_parameters_untouched=all(torch.equal(v, reference.state_dict()[k]) for k,v in initial_state.items()),
    actual_wrapper_ran=True, decoder_restored=export.load_mean is native,
    sources_archived=len(progress['source_sha256'])==2,
    qualification_excluded=runtime['eligible_for_checkpoint_mean_qualification'] is False)
out = dict(schema='motion-knee-only-CPU-preflight-v1', checks=checks, rows=rows,
    source_checkpoint=str(cp), source_sha256=digest(cp),
    wrapper_sha256=digest(wrapper), probe_sha256=digest(R/'motion_knee_damping_probe001.py'),
    new_PPO=0, simulated_steps=0, qualification=None, created_unix=time.time())
write_json(R/'motion_knee_preflight001.json', out)
print(json.dumps(out, indent=2))
assert all(checks.values()), checks
