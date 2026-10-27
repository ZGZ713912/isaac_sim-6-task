"""Explicit observation-only extension to the existing physical knee stop.

The old reference lifecycle, coordinate equations and all original table knots
are retained. Legacy settings continue to select the original frozen table.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from wheeled_algo import standing_reference_lifecycle as legacy
from wheeled_algo.rolling_velocity import RollingBaseVelocity, TABLE_SHA256
from wheeled_algo.standing_displacement import WheelAngleCoordinate

TABLE_RELATIVE = "contracts/standing_geometry107_v1.json"
EXTENDED_SHA256 = "7b313c807aefe51058567b1b2b8ce5300be85ccc8d9ec091d0aa7ae1708b9216"
SCHEMA = "wheel-leg-standing-displacement-v4-geometry107"


def table_path():
    path = Path(__file__).resolve().parents[3] / TABLE_RELATIVE
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXTENDED_SHA256:
        raise ValueError("Extended standing geometry differs from its authenticated table")
    return path


def extend_settings(settings):
    legacy.validate_observation_contract(settings)
    if "standing_position_reference" not in settings:
        raise ValueError("Geometry extension requires the complete v3 reference contract")
    result = deepcopy(settings)
    contract = result["standing_position_observation"]
    contract.update(schema=SCHEMA, geometry_table_sha256=EXTENDED_SHA256,
                    original_geometry_table_sha256=TABLE_SHA256,
                    passive_knee_domain_deg=[40., 107.], original_knots_unchanged=True)
    validate_observation_contract(result)
    return result


def legacy_settings(settings):
    if settings.get("standing_position_observation", {}).get("schema") != SCHEMA:
        legacy.validate_observation_contract(settings)
        return settings
    result = deepcopy(settings)
    contract = result["standing_position_observation"]
    expected = legacy.observation_contract(
        result.get("standing_position_filter_seconds"), **result["standing_position_reference"])
    extended = dict(expected, schema=SCHEMA, geometry_table_sha256=EXTENDED_SHA256,
                    original_geometry_table_sha256=TABLE_SHA256,
                    passive_knee_domain_deg=[40., 107.], original_knots_unchanged=True)
    if contract != extended:
        raise ValueError("Standing geometry extension requires its exact v4 contract")
    table_path()
    result["standing_position_observation"] = expected
    legacy.validate_observation_contract(result)
    return result


def validate_observation_contract(settings):
    return legacy.validate_observation_contract(legacy_settings(settings))


def observer_from_settings(settings, count, device, dt=.005):
    return legacy.observer_from_settings(legacy_settings(settings), count, device, dt)


def coordinate_from_settings(settings):
    validate_observation_contract(settings)
    coordinate = WheelAngleCoordinate()
    if settings.get("standing_position_observation", {}).get("schema") == SCHEMA:
        coordinate.geometry = RollingBaseVelocity(json.loads(table_path().read_text()))
    return coordinate
