"""A single factor3 intervention in existing quiet wheel/displacement weights."""
import math

import torch

VERSION = 'quiet-wheel-displacement-gain3-diagnostic-v1'


def apply_gain(mean, gain):
    if (type(gain) not in (int, float) or not math.isfinite(gain) or gain not in (1., 3.)
            or mean.quiet_velocity_feedback_mode != 'standing_displacement'
            or set(mean.quiet_velocity) != {'low', 'mid', 'high'}):
        raise ValueError('Require the existing displacement heads and declared factor1or3')
    records = []
    with torch.no_grad():
        for name, head in mean.quiet_velocity.items():
            if head.weight.shape != (4, 2):
                raise ValueError('Unexpected existing feedback layout')
            before = head.weight[2:4].detach().clone()
            head.weight[2:4].mul_(gain)
            records.append(dict(expert=name, before=before.tolist(), after=head.weight[2:4].tolist()))
    return records
