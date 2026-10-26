"""Recorded-state seam audit; no native simulation or optimizer changes."""
from pathlib import Path
from types import MethodType
import hashlib
import json
import math
import sys

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.paired_recovery import wrap

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
source = B / 'entry_optimizer_support0_source001/model_migrated.pt'
old, info = load_mean(source)
probe, _ = load_mean(source)
original = probe.entry_recovery.forward
width = .35


def periodic_forward(self, encoded):
    # Temporary offline-only implementation. Input here is tanh(scaled frame).
    angles = torch.atanh(encoded[:, 10:14].clamp(-.999999, .999999))
    phase = ((angles.abs() - (math.pi-width))/width).clamp(0., 1.)
    weights = .5 * phase.square() * (3.-2.*phase)
    base = original(encoded)
    result = torch.zeros_like(base)
    for mask in range(16):
        chart = encoded.clone()
        probability = torch.ones_like(encoded[:, :1])
        for j in range(4):
            alternate = bool(mask & (1 << j))
            probability = probability * (weights[:, j:j+1] if alternate else 1.-weights[:, j:j+1])
            if alternate:
                chart[:, 10+j] = torch.tanh(angles[:, j]-2.*math.pi*angles[:, j].sign())
        result += probability * (base if mask == 0 else original(chart))
    return torch.where(weights.amax(-1, keepdim=True) > 0., result, base)


probe.entry_recovery.forward = MethodType(periodic_forward, probe.entry_recovery)
run = B / 'diagnosis012_final_arithmetic_nominal/robust'
report = json.loads((run / 'report.json').read_text())
z = np.load(run / 'trajectory.npz')
rows = []
for lane in report['lanes']:
    if not lane.get('recovery') or lane['recovered'] or lane['lane'] >= 35:
        continue
    i = lane['lane']
    valid = z['alive'][:, i] & ~z['passive'][:, i] & (z['time'][:, i] >= 10.)
    obs = torch.tensor(z['observations'][valid, i])
    frame = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
    with torch.inference_mode():
        raw = canonical_recovery_logits(old.entry_recovery, frame, old.feature_scale)
        common = raw[:, :4].unflatten(-1, (2, 2)).mean(-1)
    conditional = {}
    for label, selected in [('positive_hip', obs[:, 10] > 3.), ('negative_hip', obs[:, 10] < -3.)]:
        conditional[label] = dict(frames=int(selected.sum()), common_logit_mean=common[selected].mean(0).tolist())
    seam_tests = []
    for encoder in range(10, 14):
        for eps in (.001, .00001):
            pair = obs[-1:].repeat(2, 1)
            pair[0, encoder] = math.pi-eps
            pair[1, encoder] = -math.pi+eps
            with torch.inference_mode():
                a, b = old(pair), probe(pair)
            seam_tests.append(dict(encoder_index=encoder, physical_input_difference_rad=2*eps,
                old_physical_target_seam_max_rad=float(wrap((a[0, :4]-a[1, :4])*.25).abs().max()),
                periodic_probe_physical_target_seam_max_rad=float(wrap((b[0, :4]-b[1, :4])*.25).abs().max())))
    rows.append(dict(name=lane['name'], actual_family=lane['post_settle_family'],
        survived=lane['survived'], recovered=lane['recovered'],
        mean_height_m=float(z['height'][valid, i].mean()),
        mean_wheel_clearance_m=z['wheel_clearance'][valid, i].mean(0).tolist(),
        observed_chart_conditioned_output=conditional, counterfactual_seam_tests=seam_tests))

# All ordinary non-seam entry values must remain exactly unchanged.
samples = torch.tensor(z['observations'][::20].reshape(-1, 35))
nonseam = samples[:, 10:14].abs().amax(-1) <= math.pi-width
with torch.inference_mode():
    x, y = old(samples), probe(samples)
all_tests = [test for row in rows for test in row['counterfactual_seam_tests']]
checks = dict(no_weight_or_clock_changes=all(torch.equal(v, probe.state_dict()[k]) for k, v in old.state_dict().items()),
    nonseam_actions_bit_exact=torch.equal(x[nonseam], y[nonseam]),
    finite=bool(torch.isfinite(y).all()),
    tiny_seam_probe_target_jump_under_1e_4_rad=all(test['periodic_probe_physical_target_seam_max_rad'] < 1e-4
        for test in all_tests if test['physical_input_difference_rad'] < .0001))
out = dict(schema='entry-measured-encoder-seam-audit-v1', source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    trace=str(run / 'trajectory.npz'), trace_sha256=hashlib.sha256((run/'trajectory.npz').read_bytes()).hexdigest(),
    probe='offline-only periodic chart average of the SAME frozen entry NN; seam width .35 rad',
    rows=rows, replay_frames=len(samples), nonseam_frames=int(nonseam.sum()), checks=checks,
    scope='Recorded-state association and identical-state function continuity. No proof of physical recovery or native behavior improvement.',
    new_PPO=0, qualification=None, hardware_started=False)
(R/'entry_encoder_seam_audit001.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
