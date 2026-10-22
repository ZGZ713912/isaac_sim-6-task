"""Training-only vibration costs with a 5 cm position deadband.

The actor needs no position anchor or filter state at deployment. These are
privileged reward/critic quantities, not an extra runtime standing controller.
"""
import torch


class QuietStanding:
    def __init__(self, count, device, dt, sway_m=.05):
        self.alpha = dt / (.05 + dt)
        self.sway_m = sway_m
        self.active = torch.zeros(count, device=device, dtype=torch.bool)
        self.anchor = torch.zeros(count, 2, device=device)
        self.target_slow = torch.zeros(count, 6, device=device)
        self.velocity_slow = torch.zeros_like(self.target_slow)

    def reset(self, ids):
        self.active[ids] = False
        self.target_slow[ids] = 0.
        self.velocity_slow[ids] = 0.

    def displacement(self, xy):
        return torch.where(self.active[:, None], xy - self.anchor, 0.)

    def update(self, active, xy, target, velocity):
        start = active & ~self.active
        self.anchor[start] = xy[start]
        self.target_slow[start] = target[start]
        self.velocity_slow[start] = velocity[start]
        self.active.copy_(active)
        delta = target - self.target_slow
        legs = [0, 1, 3, 4]
        angle = delta[:, legs]
        delta[:, legs] = torch.atan2(angle.sin(), angle.cos())
        self.target_slow += self.alpha * delta
        velocity_delta = velocity - self.velocity_slow
        self.velocity_slow += self.alpha * velocity_delta
        high_target = (1. - self.alpha) * delta
        high_velocity = (1. - self.alpha) * velocity_delta
        target_scale = target.new_tensor([.003, .003, .3, .003, .003, .3])
        velocity_scale = target.new_tensor([.15, .15, .8, .15, .15, .8])
        target_power = (high_target / target_scale).square().mean(-1)
        velocity_power = (high_velocity / velocity_scale).square().mean(-1)
        radius = self.displacement(xy).norm(dim=-1)
        position_cost = ((radius - self.sway_m).clamp_min(0.) / self.sway_m).square()
        # Saturate the cost gently so an initially noisy standing policy still
        # earns more by balancing than by ending the episode immediately.
        cost = active * (.3 * torch.log1p(target_power) + .05 * torch.log1p(velocity_power)
                         + .5 * torch.log1p(position_cost))
        return cost, target_power, velocity_power, radius
