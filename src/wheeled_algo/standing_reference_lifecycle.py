"""Sensor-only reference revisions, separate from the frozen coordinate/filter.

The reference sets the origin of a policy observation. It never creates a
motor command or reads physical simulator position/contact. Legacy v1/v2
retain their capture delay and reference lifecycle exactly.
"""
import math

import torch

from .standing_displacement import StandingDisplacement
from .standing_displacement_contract import (
    observation_contract as filter_contract,
    validate_observation_contract as validate_filter_contract,
)


def reference_settings(capture_delay_seconds=.2, reset_on_severe_tilt=False):
    if (isinstance(capture_delay_seconds, bool) or not isinstance(capture_delay_seconds, (int, float))
            or not math.isfinite(capture_delay_seconds) or not 0. <= capture_delay_seconds <= .2):
        raise ValueError("Standing reference capture delay must be numeric within [0,.2] seconds")
    if not isinstance(reset_on_severe_tilt, bool):
        raise ValueError("Standing reference severe-tilt reset requires an explicit boolean")
    return dict(capture_delay_seconds=capture_delay_seconds, reset_on_severe_tilt=reset_on_severe_tilt)


def observation_contract(filter_seconds=None, capture_delay_seconds=.2, reset_on_severe_tilt=False):
    reference = reference_settings(capture_delay_seconds, reset_on_severe_tilt)
    contract = filter_contract(filter_seconds)
    contract.update(schema="wheel-leg-standing-displacement-v3",
                    capture="once_after_declared_delay_active_quiet_gate_ge_.999_and_recovery_gate_le_.05",
                    capture_delay_seconds=reference["capture_delay_seconds"],
                    severe_tilt_reset=dict(enabled=reference["reset_on_severe_tilt"], normalized_gz_gt=-.5,
                                           reference_age="restart_while_severe"))
    if reset_on_severe_tilt:
        contract["reset"] += "_or_valid_normalized_gravity_z_gt_-.5"
    return contract


def validate_observation_contract(settings):
    if "standing_position_reference" not in settings:
        return validate_filter_contract(settings)
    reference = settings["standing_position_reference"]
    if not isinstance(reference, dict) or set(reference) != {"capture_delay_seconds", "reset_on_severe_tilt"}:
        raise ValueError("Standing reference requires its complete explicit settings")
    reference_settings(**reference)
    seconds = settings.get("standing_position_filter_seconds")
    expected = observation_contract(seconds, **reference)
    if settings.get("standing_position_observation") != expected:
        raise ValueError("Standing reference requires the exact v3 observation contract")
    base = dict(settings)
    del base["standing_position_reference"]
    base["standing_position_observation"] = filter_contract(seconds)
    return validate_filter_contract(base)


class StandingReference(StandingDisplacement):
    def __init__(self, count, device, dt, filter_seconds, reference):
        super().__init__(count, device, dt, filter_seconds)
        self.capture_delay_seconds = reference["capture_delay_seconds"]
        self.reset_on_severe_tilt = reference["reset_on_severe_tilt"]
        self.severe_reset_count = torch.zeros(count, dtype=torch.long, device=device)

    def reset(self, ids):
        super().reset(ids)
        self.severe_reset_count[ids] = 0

    def disarm_severe_tilt(self, gravity):
        if not self.reset_on_severe_tilt:
            return None
        norm = gravity.norm(dim=-1)
        valid = torch.isfinite(gravity).all(-1) & (norm > .5) & (norm < 1.5)
        severe = valid & (gravity[:, 2] / norm.clamp_min(1e-9) > -.5)
        # Count a lost active reference once, even if the same sensor frame is
        # read repeatedly. Resetting the age is handled by the caller every
        # severe tick, so air-spin cannot establish a new standing reference.
        self.severe_reset_count += (severe & self.active).long()
        super().reset(severe.nonzero().flatten())
        return severe


def observer_from_settings(settings, count, device, dt=.005):
    if not validate_observation_contract(settings):
        return None
    reference = settings.get("standing_position_reference", reference_settings())
    return StandingReference(count, device, dt, settings.get("standing_position_filter_seconds", .05), reference)
