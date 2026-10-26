"""Check exact endpoints and actual wrapper execution before one native scope counterfactual."""
from copy import deepcopy
from pathlib import Path
import json
import runpy
import sys
import time
import types

sys.path[:0] = ['src', 'scripts', 'reports/robust_policy_repair_20261024']
import numpy as np
import torch
import export_robust_getup as export
from train_robust_getup import write_json, digest
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from fallen_wheel_entry_scope_probe001 import EntryProtectedFallenWheels

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'fallen_output_root001/model_initial.pt'
target = B / 'pilot014_fallen_output/model_final.pt'
reference, _ = export.load_mean(source)
trained, _ = export.load_mean(target)
zero = EntryProtectedFallenWheels(deepcopy(reference), deepcopy(trained), strength=0.)
scoped = EntryProtectedFallenWheels(deepcopy(reference), deepcopy(trained))
trace = B / 'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
frames = np.load(trace)['observations'].reshape(-1, 35)
counts = dict(zero_different=0, legs_different=0, protected_count=0, protected_different=0,
    no_entry_count=0, no_entry_wheel_different=0, changed_frames=0)
with torch.inference_mode():
    for array in np.array_split(frames, (len(frames) + 511) // 512):
        x = torch.from_numpy(array.copy())
        old, neutral, out = reference(x), zero(x), scoped(x)
        unscoped = scoped.learned(x)
        entry = handoff_entry_gate(x).ravel()
        protected, free = entry >= .05, entry == 0.
        counts['zero_different'] += int((neutral != old).any(-1).sum())
        counts['legs_different'] += int((out[:, :4] != old[:, :4]).any(-1).sum())
        counts['protected_count'] += int(protected.sum())
        counts['protected_different'] += int((out[protected] != old[protected]).any(-1).sum())
        counts['no_entry_count'] += int(free.sum())
        counts['no_entry_wheel_different'] += int((out[free, 4:] != unscoped[free, 4:]).any(-1).sum())
        counts['changed_frames'] += int((out != old).any(-1).sum())
outdir = R / 'fallen_wheel_scope_wrapper_smoke001'
outdir.mkdir(exist_ok=False)
calls = []
module = types.ModuleType('evaluate_gap_robustness')
module.digest = digest
module.write_json = lambda p, v: calls.append((str(p), deepcopy(v)))
native = export.load_mean
def main():
    actual, _ = export.load_mean(target)
    with torch.inference_mode():
        x = torch.from_numpy(frames[:2048].copy())
        assert torch.equal(actual(x), scoped(x))
    module.write_json(outdir / 'task_contract.json', dict(evaluation_overrides={}))
    module.write_json(outdir / 'progress.json', dict(source_sha256={}, evaluation_overrides={}))
module.main = main
sys.modules['evaluate_gap_robustness'] = module
wrapper = R / 'evaluate_fallen_wheel_scope001.py'
sys.argv = [str(wrapper), '--checkpoint', str(target), '--output', str(outdir)]
runpy.run_path(str(wrapper), run_name='__main__')
progress = next(v for p, v in calls if p.endswith('progress.json'))
runtime = next(v for p, v in calls if p.endswith('runtime_function_readback.json'))
checks = dict(zero_strength_bit_exact=counts['zero_different'] == 0,
    all_legs_bit_exact=counts['legs_different'] == 0,
    protected_entry_bit_exact=counts['protected_count'] > 0 and counts['protected_different'] == 0,
    no_entry_retains_wheel_update_exact=counts['no_entry_count'] > 0 and counts['no_entry_wheel_different'] == 0,
    nonempty_intervention=counts['changed_frames'] > 0,
    actual_wrapper_ran=True, decoder_restored=export.load_mean is native,
    sources_archived=len(progress['source_sha256']) == 3,
    qualification_excluded=runtime['eligible_for_checkpoint_mean_qualification'] is False)
result = dict(schema='fallen-wheel-entry-scope-CPU-preflight-v1', checks=checks, counts=counts,
    frames=len(frames), source_sha256=digest(source), trained_sha256=digest(target),
    wrapper_sha256=digest(wrapper), probe_sha256=digest(R / 'fallen_wheel_entry_scope_probe001.py'),
    new_PPO=0, simulated_steps=0, qualification=None, created_unix=time.time())
write_json(R / 'fallen_wheel_scope_preflight001.json', result)
print(json.dumps(result, indent=2))
assert all(checks.values()), checks
