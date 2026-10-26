"""Measure deployable entry authority where the wheel-row update changes recorded actions."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts', 'reports/robust_policy_repair_20261024']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import tilt_gate
from fallen_output_partition_probe001 import partition

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'fallen_output_root001/model_initial.pt'
target = B / 'pilot014_fallen_output/model_final.pt'
before, _ = load_mean(source)
after, _ = load_mean(target)
partition(after, before, 'wheels')
path = B / 'diagnosis025_motion_damping080_fixed/evaluation'
report = json.loads((path / 'report.json').read_text())
trace = path / 'trajectory.npz'
assert digest(trace) == report['trajectory_sha256']
z = np.load(trace)
rows = []
with torch.inference_mode():
    for lane in report['lanes']:
        if not lane['name'].startswith('real_start'):
            continue
        i = lane['lane']
        valid = z['alive'][:, i] & ~z['passive'][:, i]
        ids = np.flatnonzero(valid)
        x = torch.from_numpy(z['observations'][ids, i].copy())
        old, new = before(x), after(x)
        delta = (old[:, 4:] - new[:, 4:]).abs().amax(-1)
        changed = delta > 1e-6
        entry = handoff_entry_gate(x).ravel()
        fallen = tilt_gate(x, True, True).ravel()
        changed_ids = torch.flatnonzero(changed).ravel() if hasattr(torch, 'flatnonzero') else changed.nonzero().ravel()
        rows.append(dict(name=lane['name'], frames=len(ids), changed_frames=int(changed.sum()),
            wheel_target_delta_peak_rad_s=float(delta.max() * 10.),
            changed_entry_at_least005=int((changed & (entry >= .05)).sum()),
            changed_entry_exact_zero=int((changed & (entry == 0.)).sum()),
            first_changed_time_s=float(z['time'][ids[int(changed_ids[0])], i]) if len(changed_ids) else None,
            first_changed_entry_gate=float(entry[changed_ids[0]]) if len(changed_ids) else None,
            first_changed_fallen_gate=float(fallen[changed_ids[0]]) if len(changed_ids) else None))
out = dict(schema='fallen-wheel-observed-entry-scope-v1', rows=rows, trace=str(trace), trace_sha256=digest(trace),
    source_sha256=digest(source), target_sha256=digest(target),
    interpretation='Frozen observed inputs only; indicates which sensor-only authority region sees different wheel targets. This is not PPO credit or a new closed-loop counterfactual.',
    new_PPO=0, qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'fallen_wheel_entry_scope001.json', out)
print(json.dumps(out, indent=2))
