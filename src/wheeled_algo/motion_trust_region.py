"""Optional post-Adam mean projection against the actual rollout Gaussian.

Sampling, log probabilities, advantages and critic updates stay in PPO. The
projection is explicit: accepted actor parameters may interpolate an Adam
proposal; its moments remain unless that proposal is rejected completely.
"""
from copy import deepcopy
import math

import torch


class MotionTrustRegion:
    def __init__(self, actor, optimizer, max_kl, max_backtracks=12):
        if (not math.isfinite(max_kl) or not 0. < max_kl <= .05
                or type(max_backtracks) is not int or not 1 <= max_backtracks <= 16):
            raise ValueError('Invalid motion trust region')
        if any(p.requires_grad for p in actor.distribution.parameters()):
            raise ValueError('Motion trust region requires frozen exploration std')
        self.actor, self.optimizer = actor, optimizer
        self.max_kl, self.max_backtracks = max_kl, max_backtracks
        self.parameters = [p for p in actor.parameters() if p.requires_grad]
        self.before = [p.detach().clone() for p in self.parameters]
        self.before_adam = {p: deepcopy(optimizer.state.get(p)) for p in self.parameters}

    @torch.no_grad()
    def maximum_group_kl(self, frame, old_params):
        old_mean, old_std = old_params
        # The std is frozen, so the Gaussian KL is exactly the squared change
        # in the mean in standard-deviation units. This path consumes no RNG.
        new_mean = self.actor.mlp(frame)
        per_frame = .5 * ((new_mean - old_mean) / old_std).square().sum(-1)
        mean = per_frame.mean()
        groups = getattr(self.actor.mlp, 'trust_region_groups', None)
        if groups is not None:
            selected = groups(frame).to(per_frame.dtype)
            values = (selected * per_frame[:, None]).sum(0) / selected.sum(0).clamp_min(1.)
            mean = torch.maximum(mean, values.max())
        return float(mean)

    @torch.no_grad()
    def project(self, frame, old_params):
        if not self.parameters:
            return dict(kl=0., scale=1., backtracks=0.)
        proposal = [p.detach().clone() for p in self.parameters]
        for attempt in range(self.max_backtracks + 1):
            scale = .5 ** attempt
            for p, before, after in zip(self.parameters, self.before, proposal, strict=True):
                p.copy_(before + scale * (after - before))
            kl = self.maximum_group_kl(frame, old_params)
            if math.isfinite(kl) and kl <= self.max_kl:
                return dict(kl=kl, scale=scale, backtracks=float(attempt))
        for p, before in zip(self.parameters, self.before, strict=True):
            p.copy_(before)
            if self.before_adam[p] is None:
                self.optimizer.state.pop(p, None)
            else:
                self.optimizer.state[p] = self.before_adam[p]
        return dict(kl=self.maximum_group_kl(frame, old_params), scale=0.,
                    backtracks=float(self.max_backtracks + 1))
