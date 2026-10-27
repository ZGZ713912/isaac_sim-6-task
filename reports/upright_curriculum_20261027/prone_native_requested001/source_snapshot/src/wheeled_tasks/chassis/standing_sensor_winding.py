"""Continuous coordinate angles from one delayed/noisy encoder stream only."""
from copy import deepcopy

from wheeled_algo import standing_reference_lifecycle as legacy
from . import standing_geometry107 as geometry
from .continuous_encoders import ContinuousEncoderAngles
from .scut_observation import CONTROL_FROM_POLICY, POLICY_FROM_CONTROL

SCHEMA = "wheel-leg-standing-displacement-v5-continuous-sensors"
WINDING = dict(source="same_delayed_noisy_six_encoder_policy_stream",
               update="once_per_new_sensor_frame_idempotent_on_repeated_samples",
               reset="episode_reset_only",
               seam="nearest_integer_2pi_turns",
               max_physical_increment_rad="strictly_less_than_pi_between_samples",
               nonfinite="invalid_until_episode_reset",
               simulator_state_writes=False)


def extend_settings(settings):
    geometry.validate_observation_contract(settings)
    if (settings.get("standing_position_observation") or {}).get("schema") != geometry.SCHEMA:
        raise ValueError("Continuous sensor revision requires the authenticated 107 degree table")
    result = deepcopy(settings)
    result["standing_position_observation"].update(schema=SCHEMA, encoder_winding=deepcopy(WINDING))
    validate_observation_contract(result)
    return result


def geometry_settings(settings):
    declared = settings.get("standing_position_observation") or {}
    if declared.get("schema") != SCHEMA:
        if declared.get("schema") == geometry.SCHEMA:
            geometry.validate_observation_contract(settings)
        else:
            legacy.validate_observation_contract(settings)
        return settings
    result = deepcopy(settings)
    contract = result["standing_position_observation"]
    if contract.pop("encoder_winding", None) != WINDING:
        raise ValueError("Continuous standing encoder contract differs")
    contract["schema"] = geometry.SCHEMA
    geometry.validate_observation_contract(result)
    return result


def validate_observation_contract(settings):
    base = geometry_settings(settings)
    if (base.get("standing_position_observation") or {}).get("schema") == geometry.SCHEMA:
        return geometry.validate_observation_contract(base)
    return legacy.validate_observation_contract(base)


def observer_from_settings(settings, count, device, dt=.005):
    base = geometry_settings(settings)
    if (base.get("standing_position_observation") or {}).get("schema") == geometry.SCHEMA:
        return geometry.observer_from_settings(base, count, device, dt)
    return legacy.observer_from_settings(base, count, device, dt)


def coordinate_from_settings(settings):
    return geometry.coordinate_from_settings(geometry_settings(settings))


class StandingSensorWinding:
    """No control outputs, physics state, velocity integration, or mode switch."""
    def __init__(self, count, device, coordinate):
        limits = [[getattr(coordinate.geometry, f"side{i}_delta_rad")[j].item()
                   for j in (0, -1)] for i in (0, 1)]
        self.encoders = ContinuousEncoderAngles(count, device, limits)

    def reset(self, ids):
        # The first frame AFTER the sensor transport/noise reset owns the new
        # winding. Ideal simulator angles never initialize this provider.
        self.encoders.initialized[ids] = False

    def update(self, sensor_positions_policy):
        raw = sensor_positions_policy[:, CONTROL_FROM_POLICY]
        missing = (~self.encoders.initialized).nonzero().flatten()
        if len(missing):
            self.encoders.reset(missing, raw)
        return self.encoders.update(raw)[:, POLICY_FROM_CONTROL]


def winding_from_settings(settings, count, device, coordinate):
    validate_observation_contract(settings)
    if (settings.get("standing_position_observation") or {}).get("schema") == SCHEMA:
        return StandingSensorWinding(count, device, coordinate)
    return None
