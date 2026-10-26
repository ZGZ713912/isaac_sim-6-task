"""Diagnostic sensor-only paired knee damping; preserve common balance rotation."""
import math
import torch
from torch import nn
from wheeled_algo.motion_target_damping import motion_damping_gate
from wheeled_algo.paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap

VERSION = 'motion-knee-only-damping-probe-v1'


class MotionKneeDamping(nn.Module):
    def __init__(self, reference, alpha=.6):
        super().__init__()
        if isinstance(alpha, bool) or not math.isfinite(alpha) or not .05 <= alpha <= 1.:
            raise ValueError('Invalid knee increment alpha')
        self.reference = reference
        self.alpha = float(alpha)

    def forward(self, obs):
        old = self.reference(obs)
        if self.alpha == 1.:
            return old
        theta = (old[:, :4]*.25).unflatten(-1, (2, 2))
        previous = (obs[:, 22:26]*.25).unflatten(-1, (2, 2))
        step = wrap(theta-previous)
        knee_step = .5*(step[..., 1]-step[..., 0])
        shift = -(1.-self.alpha)*knee_step
        error = wrap(theta-obs[:, 10:14].unflatten(-1, (2, 2)))
        # Motor0 receives -shift, motor1 +shift. Restrict this scalar rather
        # than clipping motors independently, so common rotation is retained.
        lower = torch.maximum(error[..., 0]-.12, -.12-error[..., 1])
        upper = torch.minimum(error[..., 0]+.12, .12-error[..., 1])
        bound = (self.reference.action_bound[:4]*.25).unflatten(-1, (2, 2))
        lower = torch.maximum(lower, torch.maximum(theta[..., 0]-bound[..., 0], -bound[..., 1]-theta[..., 1]))
        upper = torch.minimum(upper, torch.minimum(theta[..., 0]+bound[..., 0], bound[..., 1]-theta[..., 1]))
        sign = theta.new_tensor([1., -1.])
        nominal = theta.new_tensor(NOMINAL_DELTA)
        bounds = theta.new_tensor(DELTA_BOUNDS)
        delta = sign*(nominal+theta[..., 1]-theta[..., 0])
        ends = (bounds-delta[..., None])/(2.*sign[..., None])
        lower = torch.maximum(lower, ends.amin(-1))
        upper = torch.minimum(upper, ends.amax(-1))
        feasible = (lower <= 0.) & (upper >= 0.)
        shift = torch.where(feasible, torch.minimum(torch.maximum(shift, lower), upper), torch.zeros_like(shift))
        gate = motion_damping_gate(obs, self.reference.stance_aware_recovery, self.reference.recovery_shape_aware)
        shift = shift*gate
        change = torch.stack((-shift, shift), -1).flatten(-2)/.25
        legs = torch.where(change != 0., old[:, :4]+change, old[:, :4])
        return torch.cat((legs, old[:, 4:]), -1)
