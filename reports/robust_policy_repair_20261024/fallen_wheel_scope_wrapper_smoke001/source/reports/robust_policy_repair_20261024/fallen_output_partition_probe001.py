"""Explicit zero-PPO row partition of the rejected fallen-head update, for diagnosis only."""
import hashlib
import torch

VERSION = 'fallen-output-row-partition-probe-v1'


def partition(mean, source_mean, mode):
    if mode not in ('none', 'all', 'legs', 'wheels'):
        raise ValueError('Unknown output row partition')
    a, b = mean.state_dict(), source_mean.state_dict()
    allowed = {'recovery.4.weight', 'recovery.4.bias'}
    if a.keys() != b.keys() or any(not torch.equal(v, b[k]) for k, v in a.items() if k not in allowed):
        raise ValueError('Partition requires the exact same frozen mean function and feature basis')
    if type(mean) is not type(source_mean) or mean.quiet_stance_retained_authority != source_mean.quiet_stance_retained_authority:
        raise ValueError('Partition requires identical authenticated runtime functions')
    reset = {'none': slice(None), 'all': slice(0, 0), 'legs': slice(4, 6), 'wheels': slice(0, 4)}[mode]
    with torch.no_grad():
        mean.recovery[-1].weight[reset].copy_(source_mean.recovery[-1].weight[reset])
        mean.recovery[-1].bias[reset].copy_(source_mean.recovery[-1].bias[reset])
    h = hashlib.sha256()
    for name, value in mean.recovery[-1].state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return dict(schema=VERSION, retained_learned_rows=mode, final_layer_runtime_sha256=h.hexdigest(),
        runtime_weights_changed=mode != 'all', checkpoint_file_changed=False,
        gaussian_mean_changed=mode != 'all', sampling_std_changed=False,
        eligible_for_checkpoint_mean_qualification=False, new_PPO=0)
