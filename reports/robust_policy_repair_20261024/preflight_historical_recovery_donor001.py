"""Verify only learned recovery tensors change before any native step."""
from pathlib import Path
import json
import sys
import time

sys.path[:0] = ['src', 'scripts', 'reports/robust_policy_repair_20261024']
import numpy as np
import torch
from export_robust_getup import load_mean, digest
from train_robust_getup import write_json
from historical_recovery_donor_probe001 import apply, DONOR, DONOR_SHA

ROOT = Path.cwd()
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
torch.set_num_threads(1)
checkpoint = B / 'motion_damping080002/model_probe.pt'
mean, info = load_mean(checkpoint)
before = {k: v.clone() for k, v in mean.state_dict().items()}
rng = torch.get_rng_state().clone()
runtime = apply(mean, ROOT)
after = mean.state_dict()
with np.load(B / 'diagnosis026_motion_damping080_nominal_pair/robust/trajectory.npz') as data:
    observations = torch.as_tensor(data['observations'].reshape(-1, 35).copy())
    observations = observations[torch.isfinite(observations).all(-1)]
    observations = observations[::17]
with torch.random.fork_rng(devices=[]):
    baseline, _ = load_mean(checkpoint)
with torch.no_grad():
    changed = (mean(observations) != baseline(observations)).any(-1)
checks = dict(strict_current_mean_loaded=info['completed_updates'] == 1501,
    donor_file_hash=digest(DONOR) == DONOR_SHA,
    only_six_existing_recovery_tensors=runtime['transplanted_tensor_count'] == 6,
    all_nonrecovery_state_bitexact=all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith('recovery.')),
    feature_basis_bitexact=runtime['current_decoder_retained'],
    torch_rng_unchanged=torch.equal(rng, torch.get_rng_state()),
    recorded_actions_finite=bool(torch.isfinite(mean(observations)).all()),
    action_intervention_observed=bool(changed.any()),
    current_source_file_unchanged=digest(checkpoint) == '9a3610dc396c97d5bf3a0965f2be1217ac3e225e857604554c84b991d7f9eee9',
    old_function_not_claimed=runtime['historical_policy_function_reproduced'] is False,
    no_qualification=runtime['eligible_for_checkpoint_mean_qualification'] is False)
assert all(checks.values()), checks
write_json(R / 'historical_recovery_donor_preflight001.json', dict(
    schema='historical-recovery-donor-preflight-v1', checks=checks, runtime=runtime,
    checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
    observations_checked=len(observations), changed_observations=int(changed.sum()),
    wrapper_sha256=digest(R / 'evaluate_historical_recovery_donor001.py'),
    probe_sha256=digest(R / 'historical_recovery_donor_probe001.py'),
    batch_sha256=digest('scripts/evaluate_robust_getup_batch.py'),
    new_PPO=0, created_unix=time.time()))
print(json.dumps(checks))
