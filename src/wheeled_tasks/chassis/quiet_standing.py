"""Training-only vibration costs with a 5 cm position deadband.

The actor needs no position anchor or filter state at deployment. These are
privileged reward/critic quantities, not an extra runtime standing controller.
"""
import torch


def standing_request_mask(commands, passive, recovery_lane, ever_recovered):
    """Keep the standing anchor through a disturbance, until the request ends.

    A genuine fallen birth starts paying standing costs only after its first
    recovery. Motion births cannot erase displacement by leaning or falling.
    """
    requested = (commands[:, :2].abs() < commands.new_tensor([.1, .2])).all(-1)
    return requested & ~passive & (~recovery_lane | ever_recovered)


def standing_support_cost(clearance_peak, contact_fraction):
    air = ((clearance_peak - .003).clamp_min(0.) / .01).square().mean(-1)
    unloaded = (1. - contact_fraction).mean(-1)
    return 2. * torch.log1p(air) + .5 * unloaded


class QuietStanding:
    def __init__(self, count, device, dt, sway_m=.05, position_weight=.5):
        self.alpha = dt / (.05 + dt)
        self.sway_m = sway_m
        self.position_weight = position_weight
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
                         + self.position_weight * torch.log1p(position_cost))
        return cost, target_power, velocity_power, radius
