#!/usr/bin/env python3
"""Freeze a new 15.62 control/design pair without rewriting historical evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "flat_base_200hz_control_v2.json"
DESIGN = "scratch_standing_base_design_v2.json"
MECHANISM = "scratch_standing_base_mechanism_v2.json"


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output is not None else root / "contracts"
    parents = {name: (root / "contracts" / name).read_bytes() for name in (
        "flat_base_200hz_control_v1.json", "scratch_standing_base_design_v1.json",
        "scratch_standing_base_mechanism_v1.json")}
    control = json.loads(parents["flat_base_200hz_control_v1.json"])
    config = json.loads(parents["scratch_standing_base_mechanism_v1.json"])
    design = json.loads(parents["scratch_standing_base_design_v1.json"])
    if (control["actuators"]["wheel"]["gear_ratio"] != 15.8
            or control["actuators"]["wheel"]["curve_side"] != "wheel_output"
            or config["control_math_sha256"] != digest(parents["flat_base_200hz_control_v1.json"])
            or config["design_preflight"]["sha256"] != digest(parents["scratch_standing_base_design_v1.json"])):
        raise ValueError("Historical mechanism/control/design bindings changed")

    control["control_id"] = "flat-base-policy-200hz-control-v2"
    control["actuators"]["wheel"].update(gear_ratio=15.62,
        source_note="Full rotor-to-wheel ratio15.62 matches RMCS and the new wheel. PC velocity P0.6 and cap4.5Nm are wheel-output quantities; do not apply the ratio twice. Native response parameters remain a historical output-side prior, not identification of the new wheel.")
    control["uncertainty_note"] = (
        "New wheel ratio15.62; inherited wheel-output inertia/resistance are an unvalidated prior for replacement hardware. "
        "No new wheel response, loaded torque-speed or hardware qualification is claimed.")
    control_bytes = encode(control)
    reduction = dict(schema="wheel-leg-wheel-reduction-v1", gear_ratio=15.62,
        source="user_confirmed_20261027_and_RMCS_explicit_driver",
        reference="wheel_output", motor_angle_and_speed="divide_by_ratio_once",
        motor_torque="multiply_by_ratio_once", output_torque_cap_nm=4.5,
        legacy_asset_metadata_gear_ratio=15.8, legacy_response_gear_ratio=15.8,
        response_scope="historical_wheel_output_prior_pending_new_wheel_identification",
        new_wheel_response_identified=False)
    config.update(control_math_source="contracts/" + CONTROL,
                  control_math_sha256=digest(control_bytes), wheel_reduction=reduction)
    row = design["stages"]["speed"]["roles"]["train"]
    row["contract"].update(control_math_source=config["control_math_source"],
        control_math_sha256=config["control_math_sha256"], wheel_reduction=deepcopy(reduction))
    row["control"]["actuators.wheel"] = deepcopy(control["actuators"]["wheel"])
    row["runtime"]["wheel_reduction"] = deepcopy(reduction)
    design.update(scope="scratch_standing_new_wheel_15p62",
        review_basis=dict(parent_files_sha256={"contracts/" + name: digest(data) for name, data in parents.items()},
            change="Explicit new-wheel ratio15.62; wheel-output PD/cap and historical response prior unchanged",
            new_native_validation=False, new_hardware_identification=False))
    design_bytes = encode(design)
    config["design_preflight"] = dict(file="contracts/" + DESIGN,
        sha256=digest(design_bytes), stage="speed")
    outputs = {CONTROL: control_bytes, DESIGN: design_bytes, MECHANISM: encode(config)}
    output.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (output / name).write_bytes(data)
    return dict(schema="wheel-reduction-freeze-v1", gear_ratio=15.62,
        parent_files_sha256={"contracts/" + name: digest(data) for name, data in parents.items()},
        files_sha256={"contracts/" + name: digest(data) for name, data in outputs.items()},
        new_native_validation=False, new_hardware_identification=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "contracts")
    args = parser.parse_args()
    print(json.dumps(build(output=args.output), ensure_ascii=False, indent=2))
