#!/usr/bin/env python3
"""Rebind the full vehicle to the confirmed 110-degree zero and 40..110 stops.

First rebuild the parent asset, run its zero-PPO component probe, then rebuild
the moving-gimbal training derivative with the new component receipt. Existing
source CAD, recorded measurements, checkpoints and historical receipts stay intact.
"""
from copy import deepcopy
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"tools")]
import full_robot_v202 as geometry


def load(path):
    return json.loads(path.read_text())


def checked_files(directory, manifest):
    for name, expected in manifest["files_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or geometry.digest(path) != expected:
            raise ValueError("Asset differs from its manifest: "+name)


def sync_parent():
    directory = geometry.TARGET
    bundle = directory/"isaac"
    manifest = load(bundle/"manifest.json")
    previous_manifest = geometry.digest(bundle/"manifest.json")
    previous_build = load(directory/"build_receipt.json")
    if previous_build["asset_manifest_sha256"] != previous_manifest:
        raise ValueError("Parent build receipt differs")
    checked_files(bundle, manifest)
    before = dict(manifest["files_sha256"])
    provenance = load(directory/"source_provenance.json")
    for name, expected in provenance["source_files_sha256"].items():
        if geometry.digest(geometry.SOURCE/name) != expected:
            raise ValueError("Reviewed source CAD changed")
    spec = load(bundle/"model_spec.json")
    previous = deepcopy(spec)
    knees = ("L_joint2", "R_jonit2")
    for name, sign in zip(knees, (1., -1.)):
        joint = next(j for j in spec["joints"] if j["name"] == name)
        bounds = sorted((np.deg2rad(geometry.KNEE_INNER_DEG)
            -spec["passive_knee_zero_inner_rad"][name])/sign)
        joint["limit"].update(lower=str(bounds[0]), upper=str(bounds[1]))
    # Keep every body, frame, constraint and actuator property apart from the
    # two passive knee stops, including the actual inclined gimbal screw axes.
    restored = deepcopy(spec)
    for name in knees:
        next(j for j in restored["joints"] if j["name"] == name)["limit"] = next(
            j["limit"] for j in previous["joints"] if j["name"] == name)
    if restored != previous:
        raise ValueError("The limit update changed unrelated CAD data")
    text = (bundle/"robot.usda").read_text()
    for name in knees:
        joint = next(j for j in spec["joints"] if j["name"] == name)
        pattern = r'(def PhysicsRevoluteJoint "'+name+r'"\s*\{)(.*?)(\n\s*\})'
        def replace(match):
            body = match[2]
            for key in ("lower", "upper"):
                body, count = re.subn(r'(float physics:'+key+r'Limit = )[^\n]+',
                    lambda m:m[1]+format(np.rad2deg(float(joint["limit"][key])), ".17g"), body)
                if count != 1:
                    raise ValueError("Expected one native USD knee stop")
            return match[1]+body+match[3]
        text, count = re.subn(pattern, replace, text, flags=re.S)
        if count != 1:
            raise ValueError("Expected one native USD knee joint")
    tree = ET.parse(bundle/"robot.urdf")
    for name in knees:
        joint = next(j for j in spec["joints"] if j["name"] == name)
        tree.find(f"joint[@name='{name}']/limit").attrib.update(joint["limit"])
    old_curve = load(bundle/"gas_spring_model.json")
    geometry.save(bundle/"model_spec.json", spec)
    (bundle/"robot.usda").write_text(text)
    ET.indent(tree, space="  ")
    tree.write(bundle/"robot.urdf", encoding="utf-8", xml_declaration=True)
    geometry.finish_bundle(directory, directory, spec)
    new_curve = load(bundle/"gas_spring_model.json")
    for key in ("compression_knots_m", "force_knots_n", "source_sha256", "measured_beta_domain_deg"):
        if old_curve[key] != new_curve[key]:
            raise ValueError("The limit update changed measured spring data")
    provenance.setdefault("previous_generators", []).append(dict(
        sha256=provenance["generator_sha256"], scope="original CAD frame and collision generation"))
    provenance.update(generator_sha256=geometry.digest(geometry.__file__),
        knee_limit_update=dict(generator="scripts/sync_full_vehicle_knee_limits.py",
            generator_sha256=geometry.digest(__file__),
            source="user confirmed 110-degree zero and 40-110 degree stops 2026-10-30",
            cad_frames_masses_and_collisions_unchanged=True))
    geometry.save(directory/"source_provenance.json", provenance)
    after = load(bundle/"manifest.json")
    receipt = dict(schema="full-vehicle-knee-limits-sync-v1", knee_inner_limits_deg=list(geometry.KNEE_INNER_DEG),
        hardware_zero_reference_inner_deg=geometry.ENCODER_ZERO_INNER_DEG,
        previous_manifest_sha256=previous_manifest,
        previous_model_spec_sha256=previous_build["new_model_spec_sha256"],
        asset_manifest_sha256=geometry.digest(bundle/"manifest.json"),
        model_spec_sha256=geometry.digest(bundle/"model_spec.json"),
        changes={n:dict(before=before.get(n), after=v) for n,v in after["files_sha256"].items() if before.get(n)!=v},
        measured_force_knots_unchanged=True, measured_force_angles_reidentified=False,
        bodies_frames_constraints_and_actuator_parameters_unchanged=True,
        hardware_zero_commands=False, new_PPO=0, behavior_qualified=False)
    geometry.save(directory/"knee_limits_sync_receipt.json", receipt)
    build = load(directory/"build_receipt.json")
    build["knee_limits_sync"] = dict(file=str((directory/"knee_limits_sync_receipt.json").relative_to(ROOT)),
        sha256=geometry.digest(directory/"knee_limits_sync_receipt.json"))
    geometry.save(directory/"build_receipt.json", build)
    print(json.dumps(dict(stage="parent_rebound_needs_new_native_probe", **receipt), indent=2))


def refresh_compensation_binding():
    """Retain old V7 evidence while binding the backward-compatible loader edit."""
    directory = ROOT/"model/gas_spring_compensation_unified_20261029"
    config_path = directory/"mechanism.json"
    config = load(config_path)
    previous = config["gas_spring_compensation_evidence"]
    evidence = load(ROOT/previous["file"])
    changed_source = "src/wheeled_tasks/chassis/gas_spring_compensation.py"
    if evidence["files_sha256"][changed_source] == geometry.digest(ROOT/changed_source):
        return
    for name, expected in evidence["files_sha256"].items():
        if name != changed_source and geometry.digest(ROOT/name) != expected:
            raise ValueError("Unrelated inherited compensation evidence changed: "+name)
    evidence["files_sha256"][changed_source] = geometry.digest(ROOT/changed_source)
    evidence["previous_binding"] = previous
    evidence.update(scope="Inherited static V7 force and compensation evidence; only loader admission now checks the authenticated model's relative stops instead of hard-coded V7 values. V7 numerical control is unchanged. New 40-110 degree components and moving-gimbal wiring require separate native reports.",
        source_binding_update_only=True, new_PPO=0, hardware_release=False)
    path = directory/"evidence_full_vehicle_limits110_20261031.json"
    if path.exists():
        raise ValueError("Preserve the existing source-binding receipt before another change")
    geometry.save(path,evidence)
    binding = dict(file=str(path.relative_to(ROOT)),sha256=geometry.digest(path))
    design_path = ROOT/config["design_preflight"]["file"]
    design = load(design_path)
    design["stages"]["speed"]["roles"]["train"]["contract"]["gas_spring_compensation_evidence"] = binding
    design["review_basis"]["limits110_source_binding"] = dict(previous_evidence=previous,
        current_evidence=binding, legacy_geometry_force_and_limits_unchanged=True)
    geometry.save(design_path,design)
    config["gas_spring_compensation_evidence"] = binding
    config["design_preflight"]["sha256"] = geometry.digest(design_path)
    geometry.save(config_path,config)


def sync_training(validation):
    from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202
    asset = FullRobotV202(geometry.TARGET/"isaac")
    validation = validation.resolve()
    if not validation.is_relative_to(ROOT):
        raise ValueError("The component receipt must be archived inside the repository")
    evidence = load(validation)
    if not evidence.get("component_migration_passed") or evidence["model_spec_sha256"] != asset.spec_sha256:
        raise ValueError("New geometry requires a matching component probe")
    if not evidence.get("native_runs"):
        raise ValueError("At least one actual native component probe is required")
    for run in evidence["native_runs"]:
        result = load(ROOT/run["path"])
        if (not run.get("passed") or geometry.digest(ROOT/run["path"]) != run["sha256"]
                or not result.get("passed") or result["model_spec_sha256"] != asset.spec_sha256):
            raise ValueError("Native component evidence differs from the updated model")
    build_path = asset.bundle.parent/"build_receipt.json"
    build = load(build_path)
    build["component_validation"] = dict(file=str(validation.relative_to(ROOT)), sha256=geometry.digest(validation))
    geometry.save(build_path, build)
    refresh_compensation_binding()
    provenance_path = asset.bundle.parent/"source_provenance.json"
    provenance = load(provenance_path)
    provenance["knee_limit_update"]["generator_sha256"] = geometry.digest(__file__)
    geometry.save(provenance_path,provenance)
    definition = importlib.util.spec_from_file_location("rebuild_full_vehicle_training", ROOT/"scripts/build_full_vehicle_training.py")
    builder = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(builder)
    directory = builder.DIRECTORY
    checked_files(directory, load(directory/"manifest.json"))
    plan_path = ROOT/"contracts/wheel_leg_prone_motion_v4.json"
    previous_plan = plan_path.read_bytes()
    # The ordinary builder still refuses overwrites. This explicit migration
    # temporarily preserves the old generated tree and restores it on failure.
    with tempfile.TemporaryDirectory(prefix=".limits110-", dir=directory.parent) as tmp:
        backup = Path(tmp)/"training"
        directory.rename(backup)
        try:
            builder.main()
            plan = json.loads(previous_plan)
            mechanism = load(directory/"mechanism.json")
            for key, name in (("full_vehicle", "recipe.json"), ("mechanism", "mechanism.json"), ("pose_bank", "pose_bank.json")):
                plan[key] = dict(file=str((directory/name).relative_to(ROOT)), sha256=geometry.digest(directory/name))
            plan["gas_spring_model"]["sha256"] = geometry.digest(directory/"gas_spring_model.json")
            plan["wheel_model"]["model"]["sha256"] = geometry.digest(directory/"wheel_model.json")
            plan["gas_spring_feedforward"] = deepcopy(mechanism["v5_control"]["gas_spring_compensation"])
            plan["standing_odometry"].update(geometry_table_sha256=geometry.digest(directory/"encoder_geometry.json"),
                geometry_model_spec_sha256=geometry.digest(directory/"model_spec.json"))
            plan.update(mechanical_inner_knee_range_deg=list(geometry.KNEE_INNER_DEG),
                encoder_zero_inner_knee_deg=geometry.ENCODER_ZERO_INNER_DEG)
            geometry.save(plan_path, plan)
            from wheeled_tasks.chassis.full_vehicle_training import authenticate
            authenticate(mechanism, ROOT)
        except BaseException:
            if directory.exists():
                shutil.rmtree(directory)
            backup.rename(directory)
            plan_path.write_bytes(previous_plan)
            raise
    print(json.dumps(dict(stage="training_rebound_needs_native_entry_probe", task="WheelLeg-ProneMotion-v4",
        knee_inner_limits_deg=list(geometry.KNEE_INNER_DEG), hardware_zero_reference_inner_deg=geometry.ENCODER_ZERO_INNER_DEG)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--asset-only", action="store_true")
    mode.add_argument("--training-only", action="store_true")
    parser.add_argument("--component-validation", type=Path)
    args = parser.parse_args()
    if args.training_only:
        if args.component_validation is None:
            parser.error("--training-only requires --component-validation")
        sync_training(args.component_validation)
    else:
        sync_parent()
