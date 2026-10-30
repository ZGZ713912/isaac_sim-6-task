"""Authenticate the explicit moving-gimbal derivative of the full CAD vehicle."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .full_robot_v202 import FullRobotV202, digest
from .gimbal_motion import motion_contract


def authenticate(config, root):
    root = Path(root)
    declaration = config["full_vehicle"]
    recipe_path = root/declaration["file"]
    if digest(recipe_path) != declaration["sha256"]:
        raise ValueError("Full-vehicle training recipe changed")
    recipe = json.loads(recipe_path.read_text())
    if (recipe.get("schema") != "full-vehicle-training-derivative-v1"
            or recipe.get("zero_friction_bodies") != ["base_link"]
            or recipe.get("zero_friction_combine_mode") != "min"):
        raise ValueError("Unknown full-vehicle training derivative")
    source = FullRobotV202(root/recipe["source_bundle"])
    if (digest(source.bundle/"manifest.json") != recipe["source_manifest_sha256"]
            or source.spec_sha256 != recipe["source_spec_sha256"]):
        raise ValueError("Parent full-vehicle asset changed")
    directory = root/config["asset_directory"]
    for name, expected in recipe["files_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("Full-vehicle derivative dependency changed: "+name)
    manifest = json.loads((directory/"manifest.json").read_text())
    if digest(directory/"manifest.json") != recipe["asset_manifest_sha256"] or config["asset_manifest_sha256"] != recipe["asset_manifest_sha256"]:
        raise ValueError("Training manifest is not bound to the derivative")
    for name, expected in manifest["files_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError("Training asset dependency changed: "+name)
    spec = deepcopy(source.spec)
    for j in spec["joints"]:
        if j["name"] == "M_joint1":
            j.update(type="continuous", limit=dict(effort="2", velocity=str(math.pi)))
        if j["name"] == "M_joint2":
            j.update(type="revolute", limit=dict(lower=str(math.radians(-30.)), upper=str(math.radians(35.)), effort="10", velocity="100"))
    if json.loads((directory/"model_spec.json").read_text()) != spec or digest(directory/"model_spec.json") != recipe["model_spec_sha256"]:
        raise ValueError("Only the two declared gimbal joint overrides are allowed")
    if (manifest["rigid_body_names"] != source.manifest["rigid_body_names"]
            or manifest["tree_joint_names"] != [j["name"] for j in spec["joints"]]
            or manifest["nominal_joint_pos"] != {**source.manifest["nominal_joint_pos"], "M_joint1": 0., "M_joint2": 0.}
            or manifest["control_joint_names"] != source.manifest["control_joint_names"]
            or manifest["gimbal_mode"] != "independent_servo"
            or config.get("gimbal_motion") != motion_contract()
            or config.get("zero_friction_bodies") != ["base_link"]
            or config.get("critic_dim") != 87):
        raise ValueError("Full-vehicle motor, gimbal, friction or observation contract differs")
    for name in ("gas_spring_model.json", "encoder_geometry.json", "wheel_model.json"):
        expected = json.loads((source.bundle/name).read_text())
        if name == "wheel_model.json":
            expected["asset_rebinding"]["model_spec_sha256"] = recipe["model_spec_sha256"]
        else:
            expected["model_spec_sha256"] = recipe["model_spec_sha256"]
        if json.loads((directory/name).read_text()) != expected:
            raise ValueError("Inherited full-vehicle component changed beyond identity: "+name)
    gas = deepcopy(source.runtime["v5_control"]["gas_spring_compensation"])
    gas.update(model_file=str((directory/"gas_spring_model.json").relative_to(root)),
        model_sha256=digest(directory/"gas_spring_model.json"))
    if config["v5_control"]["gas_spring_compensation"] != gas:
        raise ValueError("Full-vehicle gas configuration differs from its inherited component")
    wheels = config.get("identified_wheel_response")
    if wheels is not None and wheels != dict(enabled=True,
            model_path=str((directory/"wheel_model.json").relative_to(root)),
            model_sha256=digest(directory/"wheel_model.json"), apply_aggregate_delay=True):
        raise ValueError("Full-vehicle wheel configuration differs from its inherited component")
    for body in spec["bodies"]:
        for collision in body["collisions"]:
            if collision["type"] == "mesh" and digest(directory/collision["file"]) != digest(source.bundle/collision["file"]):
                raise ValueError("Full-vehicle collision geometry changed")
    bank = json.loads((directory/"pose_bank.json").read_text())
    if bank["asset_manifest_sha256"] != recipe["asset_manifest_sha256"] or bank["model_spec_sha256"] != recipe["model_spec_sha256"]:
        raise ValueError("Full-vehicle pose bank has another geometry identity")
    binding = recipe["parent_component_validation"]
    if digest(root/binding["file"]) != binding["sha256"]:
        raise ValueError("Inherited component receipt changed")
    receipt = json.loads((root/binding["file"]).read_text())
    if receipt.get("component_migration_passed") is not True or receipt["model_spec_sha256"] != source.spec_sha256:
        raise ValueError("Full-vehicle components lack a matching reviewed migration")
    for run in receipt["native_runs"]:
        if digest(root/run["path"]) != run["sha256"] or run.get("passed") is not True:
            raise ValueError("Inherited full-vehicle native component report changed")
    return dict(status="authenticated_geometry_derivation", source_spec_sha256=source.spec_sha256,
        model_spec_sha256=recipe["model_spec_sha256"], recipe_sha256=declaration["sha256"],
        knee_inner_limits_deg=source.limits["knee_inner_limits_deg"],
        hardware_zero_reference_inner_deg=source.limits["hardware_zero_reference_inner_deg"],
        component_scope="unchanged chassis geometry, mass, spring curves and wheel dynamics; only two free gimbal joints",
        moving_gimbal_behavior_qualified=False, hardware_release=False)


def encoder_geometry(config, root):
    from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
    directory = Path(root)/config["asset_directory"]
    table = json.loads((directory/"encoder_geometry.json").read_text())
    return EncoderLegGeometry(table, expected_spec_sha256=digest(directory/"model_spec.json"))
