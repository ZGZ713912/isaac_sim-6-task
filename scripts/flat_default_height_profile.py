"""Additional RMCS moving cases at the packaged default body height."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import yaml


SOURCE = "rmcs_bringup/config/wheel-leg-infantry-rl.yaml"
BASE_FIXED = "flat-base-paired107-rmcs-reference-fixed-v1"
BASE_ENVELOPE = "flat-base-paired107-rmcs-reference-envelope-v1"
FIXED = "flat-base-paired107-rmcs-default-height-fixed-v1"
ENVELOPE = "flat-base-paired107-rmcs-default-height-envelope-v1"


def read_default_height_contract(rmcs, reference):
    """The ordinary-reference reader authenticates the same production YAML."""
    path = Path(rmcs) / "rmcs_ws/src" / SOURCE
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != reference["source_sha256"][SOURCE]:
        raise ValueError("Default height YAML differs from the authenticated RMCS reference")
    settings = yaml.safe_load(path.read_text())["wheel_leg_chassis_controller"]["ros__parameters"]
    height, low, high = (settings[k] for k in
        ("default_command_height", "command_height_min", "command_height_max"))
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (height, low, high)) or not .23 <= low <= height <= high <= .43:
        raise ValueError("Packaged default height is outside the original actor envelope")
    return dict(schema="flat-rmcs-packaged-default-height-v1", height_m=height,
        command_height_bounds_m=[low, high], source_file=SOURCE, source_sha256=actual,
        scope="moving requests use the packaged default height; original static heights and height scan remain separate")


def default_height_settings(settings, contract):
    """Preserve scenarios, duration, thresholds and seeds; change moving height."""
    protocols = {BASE_FIXED:FIXED, BASE_ENVELOPE:ENVELOPE}
    if settings.get("protocol_id") not in protocols:
        raise ValueError("Default-height coverage requires an ordinary RMCS reference suite")
    height = contract.get("height_m")
    if (contract.get("schema") != "flat-rmcs-packaged-default-height-v1"
            or type(height) not in (int, float) or not math.isfinite(height) or not .23 <= height <= .43):
        raise ValueError("Invalid authenticated default-height contract")
    result = deepcopy(settings)
    result["protocol_id"] = protocols[settings["protocol_id"]]
    result["packaged_default_height"] = deepcopy(contract)
    changed = []
    for case in result["cases"]:
        if not any(abs(v) > 0 for v in case["command"][:2]):
            continue
        case["command"][2] = height
        if "command" in case.get("skill", {}):
            case["skill"]["command"][2] = height
        changed.append(case["name"])
    if not changed:
        raise ValueError("Default-height suite contains no moving requests")
    result["default_height_moving_cases"] = changed
    return result
