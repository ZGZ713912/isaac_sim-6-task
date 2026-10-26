"""Counterfactual learned tensor donor; retain the current action decoder."""
from pathlib import Path
import hashlib
import torch

VERSION = 'historical-recovery-neural-donor-current-decoder-v1'
DONOR = Path('runs/unified_robust_20261022/offline_recovery096/model_migrated.pt')
DONOR_SHA = 'eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f'
PREFIX = 'mlp.mean.recovery.'


def apply(mean, root):
    path = Path(root) / DONOR
    if hashlib.sha256(path.read_bytes()).hexdigest() != DONOR_SHA:
        raise ValueError('Historical learned tensor donor changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    # This raw read is a tensor donor, never an authenticated old policy load.
    # The old decoder has a different source contract and is not transplanted.
    donor = {k.removeprefix(PREFIX): v for k, v in state['actor_state_dict'].items()
             if k.startswith(PREFIX)}
    current = mean.recovery.state_dict()
    if current.keys() != donor.keys() or any(
            current[k].shape != v.shape or current[k].dtype != v.dtype
            or not torch.isfinite(v).all() for k, v in donor.items()):
        raise ValueError('Neural tensor layout or finiteness differs')
    if not torch.equal(mean.feature_scale.cpu(), state['actor_state_dict']['mlp.mean.feature_scale']):
        raise ValueError('Recovery feature basis differs')
    before = {k: v.detach().clone() for k, v in mean.state_dict().items()}
    mean.recovery.load_state_dict(donor, strict=True)
    after = mean.state_dict()
    changed = [k for k, v in before.items() if not torch.equal(v, after[k])]
    if not changed or any(not k.startswith('recovery.') for k in changed):
        raise ValueError('Donor changed tensors outside existing fallen network')
    return dict(schema=VERSION, donor_checkpoint=str(DONOR), donor_sha256=DONOR_SHA,
        donor_clock=state['infos']['completed_updates'], source_policy_clock=1501,
        changed_tensor_names=changed, transplanted_tensor_count=len(donor),
        current_decoder_retained=True, historical_policy_function_reproduced=False,
        historical_checkpoint_authenticated=False, source_checkpoint_file_changed=False,
        gaussian_mean_changed=True, sampling_std_changed=False,
        runtime_script_or_FSM_added=False, observer_changed=False, physics_changed=False,
        new_PPO=0, eligible_for_checkpoint_mean_qualification=False)
