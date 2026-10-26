"""Measure which offline path constraints the real two-update head retained."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits
from wheeled_algo.paired_recovery import wrap

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
dataset = R / 'entry_phase_path_dataset001.npz'
z = np.load(dataset)
frames = z['observations']
desired = z['desired_common']
phase = z['phase'][z['pose_ids']]
far, near = np.abs(phase) >= .75, np.abs(phase) < 1e-6
paths = [B / 'motion_damping080002/model_probe.pt', B / 'pilot009_contract002/model_migrated.pt']
models = [load_mean(path)[0] for path in paths]
outputs = [[], []]
logits = [[], []]
gates = []
with torch.inference_mode():
    for values in np.array_split(frames, (len(frames) + 511) // 512):
        obs = torch.from_numpy(values.copy())
        frame = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        gates.append(handoff_entry_gate(obs).numpy())
        for i, model in enumerate(models):
            logits[i].append(periodic_canonical_entry_logits(model.entry_recovery, frame, model.feature_scale).numpy())
            outputs[i].append(model(obs).numpy())
outputs = [np.concatenate(v) for v in outputs]
logits = [np.concatenate(v) for v in logits]
gate = np.concatenate(gates)
far &= gate > .99
near &= gate > .99
rows = []
for path, output, raw in zip(paths, outputs, logits, strict=True):
    common = raw[:, :4].reshape(-1, 2, 2).mean(-1)
    error = torch.tensor(output[:, :4] * .25 - frames[:, 10:14])
    lead = wrap(error).numpy().reshape(-1, 2, 2).mean(-1)
    rows.append(dict(checkpoint=str(path), sha256=digest(path),
        far_common_signed_min=float((common * np.sign(desired))[far].min()),
        far_actual_signed_lead_min_rad=float((lead * np.sign(desired))[far].min()),
        far_actual_direction_correct_fraction=float(((lead * np.sign(desired))[far] > .003).mean()),
        stopped_common_peak=float(np.abs(common[near]).max()), stopped_actual_target_peak_rad=float(np.abs(lead[near]).max())))
delta_raw = logits[1] - logits[0]
delta = outputs[1] - outputs[0]
stats = dict(common_logit_delta_peak=float(np.abs(delta_raw[:, :4].reshape(-1, 2, 2).mean(-1)).max()),
    knee_logit_delta_peak=float(np.abs(delta_raw[:, [1, 3]] - delta_raw[:, [0, 2]]).max()),
    wheel_logit_delta_peak=float(np.abs(delta_raw[:, 4:]).max()),
    physical_wheel_target_delta_peak_nm=float(np.abs(delta[:, 4:]).max()),
    physical_leg_target_delta_peak_rad=float(wrap(torch.tensor(delta[:, :4] * .25)).abs().max()))
out = dict(schema='real-partial-PPO-phase-constraint-audit-v1', dataset=str(dataset), dataset_sha256=digest(dataset),
    frames=len(frames), rows=rows, changes=stats,
    scope='Offline original development constraints after real native PPO; no additional physical trajectory, causal claim or new training.',
    new_PPO=0, actual_experiment_consumed=2605, qualification=None, hardware_started=False, created_unix=time.time())
(R / 'partial_PPO_phase_audit001.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2), flush=True)
