"""Authenticate the explicit periodic candidate and replay all three banks."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import sys
import tempfile
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
source = B/'entry_optimizer_support0_source001/model_migrated.pt'
checkpoint = B/'periodic_entry_encoder001/model_probe.pt'
old, oi = load_mean(source)
new, ni = load_mean(checkpoint)
src = torch.load(source, map_location='cpu', weights_only=False)
dst = torch.load(checkpoint, map_location='cpu', weights_only=False)
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation',
             'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation'):
    trace = B/name/'trajectory.npz'
    z = np.load(trace)
    frames = z['observations'].reshape(-1, 35)
    invariant_count = exact_count = changed_count = 0
    all_finite = True
    max_delta = 0.
    max_target_delta = 0.
    for start in range(0, len(frames), 2048):
        obs = torch.tensor(frames[start:start+2048])
        invariant = (handoff_entry_gate(obs)[:, 0] == 0.) | (obs[:, 10:14].abs().amax(-1) <= math.pi-.35)
        with torch.inference_mode():
            a, b = old(obs), new(obs)
        exact = (a == b).all(-1)
        invariant_count += int(invariant.sum())
        exact_count += int(exact[invariant].sum())
        changed_count += int((~exact).sum())
        all_finite &= bool(torch.isfinite(b).all())
        max_delta = max(max_delta, float((a-b).abs().max()))
        difference = (a[:, :4]-b[:, :4])*.25
        max_target_delta = max(max_target_delta, float(torch.atan2(difference.sin(), difference.cos()).abs().max()))
    rows.append(dict(trace=str(trace), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        frames=len(frames), invariant_frames=invariant_count, invariant_frames_bit_exact=exact_count,
        changed_frames=changed_count, finite=all_finite, maximum_raw_action_difference=max_delta,
        maximum_physical_target_difference_rad=max_target_delta))
    print(name, rows[-1], flush=True)

# These are real complete checkpoints: factory must reject provenance tampering.
negatives = {}
with tempfile.TemporaryDirectory(prefix='periodic-entry-preflight-', dir=R) as directory:
    for case in ('weight', 'clock', 'source_sha', 'helper_sha', 'unrelated_setting', 'unknown_version', 'no_split'):
        changed = deepcopy(dst)
        if case == 'weight':
            changed['actor_state_dict']['mlp.mean.entry_recovery.4.bias'][0] += .01
        elif case == 'clock':
            changed['infos']['completed_updates'] += 1
        elif case == 'source_sha':
            changed['infos']['entry_encoder_migration']['source_sha256'] = '0'*64
        elif case == 'helper_sha':
            changed['infos']['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = '0'*64
        elif case == 'unrelated_setting':
            changed['infos']['settings']['recovery_support_weight'] = 1.
        elif case == 'unknown_version':
            changed['infos']['settings']['gap_entry_encoder_version'] = 'unknown-v1'
        else:
            changed['infos']['settings']['gap_entry_recovery_split_version'] = None
        path = Path(directory)/f'{case}.pt'
        torch.save(changed, path)
        try:
            load_mean(path)
        except (ValueError, RuntimeError, KeyError) as error:
            negatives[case] = dict(rejected=True, reason=str(error))
        else:
            negatives[case] = dict(rejected=False)

obs = torch.zeros(1, 35)
obs[:, 3] = 1.525
obs[:, 9] = -1.
obs[:, 10:14] = torch.tensor([3.05, -2.5, -3.05, 2.5])
obs[:, 22:26] = obs[:, 10:14]/.25
timing = {}
with torch.inference_mode():
    for label, mean in [('source', old), ('periodic_probe', new)]:
        for _ in range(20):
            mean(obs)
        start = time.perf_counter()
        for _ in range(300):
            mean(obs)
        timing[label] = (time.perf_counter()-start)*1000/300
checks = dict(all_tensors_and_state_exact=exact_state_equal({k:v for k,v in src.items() if k!='infos'},
                                                          {k:v for k,v in dst.items() if k!='infos'}),
    learning_clock_exact=oi['completed_updates']==ni['completed_updates']==1501,
    state_dict_exact=all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()),
    all_invariant_frames_bit_exact=all(x['invariant_frames']==x['invariant_frames_bit_exact'] for x in rows),
    all_frames_finite=all(x['finite'] for x in rows),
    all_negative_provenance_cases_rejected=all(x['rejected'] for x in negatives.values()))
out = dict(schema='periodic-entry-real-checkpoint-preflight-v1', checkpoint=str(checkpoint),
    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), checks=checks, replays=rows,
    negative_cases=negatives, one_thread_CPU_mean_inference_ms=timing,
    timing_scope='Local PyTorch research diagnostic; not ONNX/RMCS timing acceptance.',
    new_PPO=0, actual_experiment_consumed=2603, qualification=None, hardware_started=False)
(R/'periodic_entry_preflight001.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('replays','negative_cases')}, indent=2))
assert all(checks.values())
