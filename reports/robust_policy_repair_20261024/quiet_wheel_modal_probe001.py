"""Preserve each wheel's common response while decoupling displacement columns."""
import torch

VERSION = 'quiet-wheel-common-preserved-diagonal-v1'


def apply_modal(mean, *, neutral=False):
    if mean.quiet_velocity_feedback_mode != 'standing_displacement':
        raise ValueError('Require the existing two wheel displacement observations')
    records = {}
    with torch.no_grad():
        for name in ('low', 'mid', 'high'):
            weight = mean.quiet_velocity[name].weight
            if weight.shape != (4, 2):
                raise ValueError('Require the existing4x2 feedback weight')
            before = weight.clone()
            common = before[2:].sum(-1)
            if not neutral:
                weight[2:].zero_()
                weight[2, 0] = common[0]
                weight[3, 1] = common[1]
            records[name] = dict(before=before.tolist(), after=weight.tolist(),
                common_row_sum=common.tolist(), changed_entries=int((before != weight).sum()))
    return records
