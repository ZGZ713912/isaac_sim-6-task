"""Terrain-contact risk proxies at physical rate; no claim of structural stress.

Flat-ground normal force is the nonnegative world-Z body resultant. The metric
does not measure internal contacts while native self-collision remains disabled.
Each body's fixed-time impulse is accumulated BEFORE reduction into part groups.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch

GROUPS = ("chassis", "gimbal", "legs", "wheels")


def validate(config, physics_dt=.001):
    c = deepcopy(config)
    if c.get("schema") != "wheel-leg-recovery-impact-v1":
        raise ValueError("Need the explicit recovery impact contract")
    window = c.get("window_seconds", 0.)
    if (not math.isfinite(window) or not physics_dt <= window <= .2
            or not math.isclose(window/physics_dt, round(window/physics_dt), abs_tol=1e-7)):
        raise ValueError("Impact window must contain an integral number of physical steps")
    for key in ("force_limits_n", "impulse_limits_ns"):
        if not isinstance(c[key],dict) or set(c[key]) != set(GROUPS):
            raise ValueError("Specify every part group, using null for an unmeasured structural limit")
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v <= 0)
               for v in c[key].values()):
            raise ValueError("Structural limits must be finite positive values or null")
    if (set(c["cost_weights"]) != set(GROUPS)
            or any(not math.isfinite(v) or v < 0 for v in c["cost_weights"].values())):
        raise ValueError("Invalid secondary impact weights")
    if c.get("scope") != "flat_ground_body_resultants_not_self_collision_or_structural_stress":
        raise ValueError("Contact proxy scope must remain explicit")
    return c


def body_groups(names):
    result = []
    for name in names:
        group = ("chassis" if name == "base_link" else "gimbal" if name.startswith("M_")
                 else "wheels" if name in ("L_link3", "R_link3") else "legs")
        result.append(GROUPS.index(group))
    return result


class RecoveryImpact:
    def __init__(self, count, device, body_names, config, physics_dt, mass_kg):
        self.config = validate(config, physics_dt)
        self.dt = physics_dt
        self.ids = torch.tensor(body_groups(body_names), device=device)
        self.members = [torch.tensor([i for i,g in enumerate(body_groups(body_names)) if g == group],
                                     device=device, dtype=torch.long) for group in range(len(GROUPS))]
        self.history = torch.zeros(round(config["window_seconds"]/physics_dt), count, len(body_names), device=device)
        self.running = torch.zeros(count, len(body_names), device=device)
        self.cursor = 0
        self.force_peak = torch.zeros(count, len(GROUPS), device=device)
        self.impulse_peak = self.force_peak.clone()
        self.episode_force_peak = self.force_peak.clone()
        self.episode_impulse_peak = self.force_peak.clone()
        self.cost_integral = torch.zeros(count, device=device)
        self.violation = torch.zeros(count, dtype=torch.bool, device=device)
        self.force_limits = self.force_peak.new_tensor([config["force_limits_n"][g] or float("inf") for g in GROUPS])
        self.impulse_limits = self.force_peak.new_tensor([config["impulse_limits_ns"][g] or float("inf") for g in GROUPS])
        self.weights = self.force_peak.new_tensor([config["cost_weights"][g] for g in GROUPS])
        self.weight_n = torch.as_tensor(mass_kg, device=device).expand(count)*9.81

    def reset(self, ids):
        self.history[:, ids] = 0.
        self.running[ids] = 0.
        self.episode_force_peak[ids] = self.episode_impulse_peak[ids] = 0.
        self.force_peak[ids] = self.impulse_peak[ids] = 0.
        self.cost_integral[ids] = 0.
        self.violation[ids] = False

    def begin_interval(self):
        # Fixed-time history crosses policy and flip-to-rise boundaries.
        self.force_peak.zero_()
        self.impulse_peak.zero_()
        self.cost_integral.zero_()
        self.violation.zero_()

    def observe(self, contact_force, active):
        force = contact_force[..., 2].clamp_min(0.)*active[:, None]
        impulse = force*self.dt
        self.running += impulse-self.history[self.cursor]
        self.history[self.cursor].copy_(impulse)
        self.cursor = (self.cursor+1) % len(self.history)
        current_f = torch.stack([force[:, members].amax(-1) if len(members)
                                 else force.new_zeros(force.shape[0]) for members in self.members], -1)
        current_j = torch.stack([self.running[:, members].clamp_min(0.).amax(-1) if len(members)
                                 else force.new_zeros(force.shape[0]) for members in self.members], -1)
        self.force_peak.copy_(torch.maximum(self.force_peak, current_f))
        self.impulse_peak.copy_(torch.maximum(self.impulse_peak, current_j))
        self.episode_force_peak.copy_(torch.maximum(self.episode_force_peak, current_f))
        self.episode_impulse_peak.copy_(torch.maximum(self.episode_impulse_peak, current_j))
        self.cost_integral += ((current_f/self.weight_n[:, None]).square()*self.weights).sum(-1)*self.dt
        self.violation |= active & ((current_f > self.force_limits).any(-1) | (current_j > self.impulse_limits).any(-1))
