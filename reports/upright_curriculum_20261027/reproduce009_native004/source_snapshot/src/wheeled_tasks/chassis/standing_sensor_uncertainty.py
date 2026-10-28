"""Bounded measurement projection at the frozen physical geometry endpoints."""
from copy import deepcopy

import torch
from torch import nn

from . import standing_sensor_winding as winding

SCHEMA = "wheel-leg-standing-displacement-v6-bounded-sensor-projection"
PROJECTION = dict(
    encoder_bias_bound_rad=.005,
    encoder_noise_sigma_rad=.00035,
    relative_error_bound_rad=.012,
    derivation="two_bias_bounds_plus_four_sigma_of_independent_encoder_difference",
    mapping="nearest_relative_endpoint_preserving_measured_pair_common_angle",
    numerical_interior_margin="sixteen_float32_eps_times_one_plus_max_abs_leg_angle",
    outside_bound="invalid_zero_output_preserve_reference",
    geometry="unchanged_40_to_107_degree_physical_table",
    simulator_state_writes=False,
)


def extend_settings(settings):
    winding.validate_observation_contract(settings)
    if (settings.get("standing_position_observation") or {}).get("schema") != winding.SCHEMA:
        raise ValueError("Sensor projection requires the continuous v5 observation contract")
    result = deepcopy(settings)
    result["standing_position_observation"].update(schema=SCHEMA, sensor_projection=deepcopy(PROJECTION))
    validate_observation_contract(result)
    return result


def base_settings(settings):
    if (settings.get("standing_position_observation") or {}).get("schema") != SCHEMA:
        winding.validate_observation_contract(settings)
        return settings
    result = deepcopy(settings)
    contract = result["standing_position_observation"]
    if contract.pop("sensor_projection", None) != PROJECTION:
        raise ValueError("Standing sensor uncertainty contract differs")
    contract["schema"] = winding.SCHEMA
    if (result.get("encoder_bias_rad", 0.) > PROJECTION["encoder_bias_bound_rad"] + 1e-12
            or result.get("encoder_noise_rad", 0.) > PROJECTION["encoder_noise_sigma_rad"] + 1e-12):
        raise ValueError("Sensor randomization exceeds the declared observation uncertainty")
    winding.validate_observation_contract(result)
    return result


def validate_observation_contract(settings):
    return winding.validate_observation_contract(base_settings(settings))


class BoundedSensorCoordinate(nn.Module):
    """Project noisy measurements only; never change physical or control limits."""
    def __init__(self, coordinate):
        super().__init__()
        self.coordinate = coordinate

    @property
    def geometry(self):
        return self.coordinate.geometry

    def forward(self, positions, gravity):
        q = positions[:, :4].unflatten(-1, (2, 2))
        difference = q[..., 1] - q[..., 0]
        difference = torch.atan2(difference.sin(), difference.cos())
        lower = torch.stack([getattr(self.geometry, f"side{i}_delta_rad")[0] for i in (0, 1)])
        upper = torch.stack([getattr(self.geometry, f"side{i}_delta_rad")[-1] for i in (0, 1)])
        clipped = torch.maximum(torch.minimum(difference, upper), lower)
        overshoot = (difference - clipped).abs()
        accepted = (torch.isfinite(positions).all(-1, keepdim=True)
                    & (overshoot <= PROJECTION["relative_error_bound_rad"]).all(-1, keepdim=True))
        outside = overshoot > 0.
        # Only projected samples receive a tiny inward roundoff margin. All
        # in-domain sensor frames pass through bit for bit unchanged.
        margin = 16. * torch.finfo(torch.float32).eps * (1. + q.abs().amax(-1))
        interior = torch.maximum(torch.minimum(difference, upper - margin), lower + margin)
        error = torch.where(outside, difference - interior, torch.zeros_like(difference))
        projected_legs = torch.stack((q[..., 0] + .5 * error, q[..., 1] - .5 * error), -1).flatten(-2)
        projected = torch.cat((torch.where(outside.repeat_interleave(2, -1), projected_legs, positions[:, :4]),
                               positions[:, 4:]), -1)
        result, valid = self.coordinate(projected, gravity)
        valid = valid & accepted
        self.last_projected = outside.any(-1) & accepted[:, 0]
        self.last_rejected = ~accepted[:, 0]
        self.last_projection_abs_rad = torch.where(accepted, overshoot, torch.zeros_like(overshoot)).amax(-1)
        return torch.where(valid, result, torch.zeros_like(result)), valid


def observer_from_settings(settings, count, device, dt=.005):
    return winding.observer_from_settings(base_settings(settings), count, device, dt)


def coordinate_from_settings(settings):
    coordinate = winding.coordinate_from_settings(base_settings(settings))
    if (settings.get("standing_position_observation") or {}).get("schema") == SCHEMA:
        return BoundedSensorCoordinate(coordinate)
    return coordinate


def winding_from_settings(settings, count, device, coordinate):
    return winding.winding_from_settings(base_settings(settings), count, device, coordinate)
