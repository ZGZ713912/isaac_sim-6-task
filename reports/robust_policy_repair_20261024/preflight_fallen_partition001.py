"""Check actual authenticated means and execute the native wrapper with a CPU evaluator substitute."""
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
from fallen_output_partition_probe001 import partition

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'fallen_output_root001/model_initial.pt'
target = B / 'pilot014_fallen_output/model_final.pt'
baseline, _ = export.load_mean(source)
trained, _ = export.load_mean(target)
models = {key: deepcopy(trained) for key in ('none', 'all', 'legs', 'wheels')}
for key, mean in models.items():
    partition(mean, baseline, key)
trace = B / 'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
frames = np.load(trace)['observations'].reshape(-1, 35)
different = {key: 0 for key in ('none', 'all', 'legs_wheel_output', 'wheels_leg_output')}
active_changes = dict(legs=0, wheels=0)
with torch.inference_mode():
    for array in np.array_split(frames, (len(frames) + 511) // 512):
        x = torch.from_numpy(array.copy())
        old, new = baseline(x), trained(x)
        values = {k: m(x) for k, m in models.items()}
        different['none'] += int((values['none'] != old).any(-1).sum())
        different['all'] += int((values['all'] != new).any(-1).sum())
        different['legs_wheel_output'] += int((values['legs'][:, 4:] != old[:, 4:]).any(-1).sum())
        different['wheels_leg_output'] += int((values['wheels'][:, :4] != old[:, :4]).any(-1).sum())
        for k in ('legs', 'wheels'):
            active_changes[k] += int((values[k] != old).any(-1).sum())
native_load = export.load_mean
wrapper_checks = []
for mode in ('legs', 'wheels'):
    out = R / ('fallen_partition_wrapper_smoke_' + mode + '001')
    out.mkdir(exist_ok=False)
    calls = []
    module = types.ModuleType('evaluate_gap_robustness')
    module.digest = digest
    module.write_json = lambda p, v: calls.append((str(p), deepcopy(v)))
    def main():
        decoded, _ = export.load_mean(target)
        module.write_json(out / 'task_contract.json', dict(evaluation_overrides={}))
        module.write_json(out / 'progress.json', dict(source_sha256={}, evaluation_overrides={}))
        with torch.inference_mode():
            x = torch.from_numpy(frames[:512].copy())
            assert torch.equal(decoded(x), models[mode](x))
    module.main = main
    sys.modules['evaluate_gap_robustness'] = module
    wrapper = R / 'evaluate_fallen_partition_probe001.py'
    sys.argv = [str(wrapper), '--checkpoint', str(target), '--output', str(out), '--retained-fallen-output-rows', mode]
    runpy.run_path(str(wrapper), run_name='__main__')
    metadata = next(v for p, v in calls if p.endswith('progress.json'))
    runtime = next(v for p, v in calls if p.endswith('runtime_function_readback.json'))
    wrapper_checks.append(export.load_mean is native_load and len(metadata['source_sha256']) == 2
        and metadata['evaluation_overrides']['fallen_output_partition_probe']['eligible_for_checkpoint_mean_qualification'] is False
        and runtime['retained_learned_rows'] == mode and runtime['checkpoint_file_changed'] is False)
checks = dict(none_exact_source=different['none'] == 0, all_exact_trained=different['all'] == 0,
    legs_only_wheel_output_exact=different['legs_wheel_output'] == 0,
    wheels_only_leg_output_exact=different['wheels_leg_output'] == 0,
    both_interventions_nonempty=all(v > 0 for v in active_changes.values()), actual_wrapper_both_modes=all(wrapper_checks))
out = dict(schema='fallen-output-partition-CPU-preflight-v1', checks=checks,
    source_sha256=digest(source), checkpoint_sha256=digest(target), frames=len(frames),
    differing_frames=different, changed_frames=active_changes,
    probe_sha256=digest(R / 'fallen_output_partition_probe001.py'), wrapper_sha256=digest(R / 'evaluate_fallen_partition_probe001.py'),
    simulated_steps=0, new_PPO=0, qualification=None, created_unix=time.time())
write_json(R / 'fallen_partition_preflight001.json', out)
print(json.dumps(out, indent=2))
assert all(checks.values()), checks
