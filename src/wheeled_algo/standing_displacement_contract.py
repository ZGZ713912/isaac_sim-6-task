"""Explicit observation revisions around the frozen v1 encoder estimator.

Old checkpoints keep the exact v1 contract and 50 ms filter. A changed filter
must be declared in both settings and the observation contract; it cannot be
silently applied by an evaluator or by loading a different source version.
"""
import math

from .standing_displacement import (
    StandingDisplacement,
    observation_contract as v1_observation_contract,
    validate_observation_contract as validate_v1,
)


def observation_contract(filter_seconds=None):
    contract = v1_observation_contract()
    if filter_seconds is not None:
        if (isinstance(filter_seconds, bool) or not isinstance(filter_seconds, (int, float))
                or not math.isfinite(filter_seconds) or not .05 <= filter_seconds <= .2):
            raise ValueError("Standing position filter must be a finite number within [.05,.2] seconds")
        contract.update(schema="wheel-leg-standing-displacement-v2", filter_seconds=filter_seconds)
    return contract


def validate_observation_contract(settings):
    if "standing_position_filter_seconds" not in settings:
        return validate_v1(settings)
    if (settings.get("quiet_velocity_feedback_mode") != "standing_displacement"
            or not settings.get("quiet_velocity_feedback")):
        raise ValueError("Standing position filter requires the explicit standing displacement mode")
    seconds = settings["standing_position_filter_seconds"]
    if seconds is None:
        raise ValueError("Standing position filter must be explicitly numeric")
    expected = observation_contract(seconds)
    if settings.get("standing_position_observation") != expected:
        raise ValueError("Standing position filter requires the exact v2 observation contract")
    return True


def observer_from_settings(settings, count, device, dt=.005):
    if not validate_observation_contract(settings):
        return None
    return StandingDisplacement(count, device, dt=dt,
                                filter_seconds=settings.get("standing_position_filter_seconds", .05))
