"""Cover moving commands at the authenticated RMCS upper height boundary."""
from copy import deepcopy
import math

from flat_default_height_profile import (
    BASE_ENVELOPE, BASE_FIXED, read_default_height_contract,
)

FIXED = "flat-base-paired107-rmcs-maximum-height-fixed-v1"
ENVELOPE = "flat-base-paired107-rmcs-maximum-height-envelope-v1"


def read_maximum_height_contract(rmcs, reference):
    default = read_default_height_contract(rmcs, reference)
    return dict(schema="flat-rmcs-packaged-maximum-height-v1",
        height_m=default["command_height_bounds_m"][1],
        command_height_bounds_m=list(default["command_height_bounds_m"]),
        source_file=default["source_file"], source_sha256=default["source_sha256"],
        scope="moving requests use the packaged maximum height; original static heights and height scan remain separate")


def maximum_height_settings(settings, contract):
    protocols = {BASE_FIXED: FIXED, BASE_ENVELOPE: ENVELOPE}
    if settings.get("protocol_id") not in protocols:
        raise ValueError("Maximum-height coverage requires an ordinary RMCS reference suite")
    bounds = contract.get("command_height_bounds_m")
    height = contract.get("height_m")
    if (contract.get("schema") != "flat-rmcs-packaged-maximum-height-v1"
            or not isinstance(bounds, list) or len(bounds) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in [height, *bounds])
            or not .23 <= bounds[0] <= height == bounds[1] <= .43):
        raise ValueError("Invalid authenticated maximum-height contract")
    result = deepcopy(settings)
    result["protocol_id"] = protocols[settings["protocol_id"]]
    result["packaged_maximum_height"] = deepcopy(contract)
    changed = []
    for case in result["cases"]:
        if not any(abs(v) > 0 for v in case["command"][:2]):
            continue
        case["command"][2] = height
        if "command" in case.get("skill", {}):
            case["skill"]["command"][2] = height
        changed.append(case["name"])
    if not changed:
        raise ValueError("Maximum-height suite contains no moving requests")
    result["maximum_height_moving_cases"] = changed
    return result
