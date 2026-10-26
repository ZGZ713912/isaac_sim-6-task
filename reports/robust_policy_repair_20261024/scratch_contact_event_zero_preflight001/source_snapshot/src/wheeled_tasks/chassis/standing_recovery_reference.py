"""Capture the standing reference after both learned recovery gates release."""
from copy import deepcopy

import torch

from wheeled_algo.robust_getup import tilt_gate
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from . import standing_sensor_uncertainty as sensors

SCHEMA = "wheel-leg-standing-displacement-v7-recovery-reference"
CAPTURE = "once_after_declared_delay_active_quiet_gate_ge_.999_and_max_fallen_entry_gate_le_.05"
RECOVERY_GATE = dict(fallen="stance_and_shape_aware_tilt_gate", entry="orbit-v2_handoff_entry_gate",
                     composition="maximum", threshold=.05, scope="standing_reference_capture_only")


def extend_settings(settings):
    sensors.validate_observation_contract(settings)
    if (settings.get("standing_position_observation") or {}).get("schema") != sensors.SCHEMA:
        raise ValueError("Recovery reference revision requires the bounded v6 sensor observation")
    result = deepcopy(settings)
    result["standing_position_observation"].update(schema=SCHEMA, capture=CAPTURE,
                                                  recovery_reference_gate=deepcopy(RECOVERY_GATE))
    validate_observation_contract(result)
    return result


def base_settings(settings):
    if (settings.get("standing_position_observation") or {}).get("schema") != SCHEMA:
        sensors.validate_observation_contract(settings)
        return settings
    result = deepcopy(settings)
    contract = result["standing_position_observation"]
    if (contract.pop("recovery_reference_gate", None) != RECOVERY_GATE
            or contract.get("capture") != CAPTURE
            or result.get("gap_entry_handoff_version") != "orbit-v2"
            or not result.get("upright_entry_recovery")):
        raise ValueError("Standing recovery reference gate differs from the actual actor handoff")
    contract.update(schema=sensors.SCHEMA,
                    capture="once_after_declared_delay_active_quiet_gate_ge_.999_and_recovery_gate_le_.05")
    sensors.validate_observation_contract(result)
    return result


def validate_observation_contract(settings):
    return sensors.validate_observation_contract(base_settings(settings))


def recovery_capture_gate(settings, frame):
    fallen = tilt_gate(frame, settings.get("stance_aware_recovery", False),
                       settings.get("recovery_shape_aware", False))
    if (settings.get("standing_position_observation") or {}).get("schema") == SCHEMA:
        return torch.maximum(fallen, handoff_entry_gate(frame))
    return fallen


def observer_from_settings(settings, count, device, dt=.005):
    return sensors.observer_from_settings(base_settings(settings), count, device, dt)


def coordinate_from_settings(settings):
    return sensors.coordinate_from_settings(base_settings(settings))


def winding_from_settings(settings, count, device, coordinate):
    return sensors.winding_from_settings(base_settings(settings), count, device, coordinate)
