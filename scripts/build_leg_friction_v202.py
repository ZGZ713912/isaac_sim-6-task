#!/usr/bin/env python3
"""Create an immutable friction derivative; preserve assets used by live learners."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202, digest
from wheeled_tasks.chassis.leg_motor_friction import JOINTS, MOTORS, declaration, load

PARENT = ROOT/"model/wheel_leg_v2.0.2"
TARGET = PARENT/"friction_v1"
MODEL = ROOT/"model/dm_leg_friction_20261031"


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def binding(path):
    return dict(file=str(path.relative_to(ROOT)), sha256=digest(path))


def build(source):
    if TARGET.exists() or MODEL.exists():
        raise ValueError("Preserve the existing friction derivative before rebuilding")
    asset = FullRobotV202(PARENT/"isaac")
    parameters = read(source/"parameters.json")
    MODEL.mkdir()
    for name in ("parameters.json", "README.md", "friction.png"):
        shutil.copy2(source/name, MODEL/name)
    save(MODEL/"model.json", dict(schema="dm-leg-moving-friction-v1", torque_units=parameters["torque_units"],
        joint_names=list(JOINTS), motor_names=list(MOTORS),
        coulomb_nm=[parameters["motors"][name]["coulomb_plateau_candidate_nm"] for name in MOTORS],
        viscous_coefficient_nm_s_rad=None, static_breakaway_nm=None, stribeck_velocity_rad_s=None,
        numerical_regularization=dict(kind="linear_velocity_band", velocity_rad_s=.05, identified=False),
        physical_static_model_enabled=False, bias_applied=False, controller_compensation=False,
        outside_identified_domain="constant_coulomb_prior_not_identified",
        low_speed_table_usage="evidence only; low-speed moving friction is not a static peak",
        files_sha256={name:digest(MODEL/name) for name in ("parameters.json", "README.md")},
        source_directory=str(source), source_parameter_sha256=digest(source/"parameters.json"),
        hardware_ready=False))
    model_cfg = declaration(ROOT)
    load(model_cfg, ROOT)
    TARGET.mkdir()
    plan_path = ROOT/"contracts/wheel_leg_prone_motion_v4.json"
    shutil.copy2(plan_path, TARGET/"previous_plan.json")
    # All bytes are independent copies: never hardlink mutable JSON or USD to
    # an existing run's frozen bundle. Git reuses identical mesh content blobs.
    shutil.copytree(PARENT/"isaac", TARGET/"isaac")
    shutil.copytree(PARENT/"training", TARGET/"training")
    bundle, training = TARGET/"isaac", TARGET/"training"
    runtime = read(bundle/"runtime.json")
    runtime["leg_motor_friction"] = model_cfg
    for field, filename, pathkey, shakey in (("identified_wheel_response", "wheel_model.json", "model_path", "model_sha256"),
            ("gas_spring_compensation", "gas_spring_model.json", "model_file", "model_sha256")):
        entry = runtime[field] if field in runtime else runtime["v5_control"][field]
        entry.update({pathkey:str((bundle/filename).relative_to(ROOT)), shakey:digest(bundle/filename)})
    save(bundle/"runtime.json", runtime)
    manifest = read(bundle/"manifest.json")
    manifest["asset_revision"] += "_dm_moving_friction_v1"
    manifest["runtime_leg_motor_friction_required"] = True
    manifest["files_sha256"]["runtime.json"] = digest(bundle/"runtime.json")
    save(bundle/"manifest.json", manifest)
    receipt = read(PARENT/"build_receipt.json")
    receipt.update(asset_manifest_sha256=digest(bundle/"manifest.json"),
        friction_derivative=dict(parent_manifest=binding(PARENT/"isaac/manifest.json"),
            model=model_cfg, native_validation="separate receipt required; inherited report covers geometry/spring/wheels"))
    save(TARGET/"build_receipt.json", receipt)
    recipe = read(training/"recipe.json")
    recipe.update(source_bundle=str(bundle.relative_to(ROOT)), source_manifest_sha256=digest(bundle/"manifest.json"),
        leg_motor_friction=model_cfg, allowed_changes=recipe["allowed_changes"]+"; add separately identified passive leg moving friction")
    manifest = read(training/"manifest.json")
    manifest["asset_revision"] += "_dm_moving_friction_v1"
    manifest["runtime_leg_motor_friction_required"] = True
    save(training/"manifest.json", manifest)
    recipe["asset_manifest_sha256"] = digest(training/"manifest.json")
    bank = read(training/"pose_bank.json")
    bank.update(asset_directory=str(training.relative_to(ROOT)), asset_manifest_sha256=recipe["asset_manifest_sha256"])
    save(training/"pose_bank.json", bank)
    recipe["files_sha256"] = {name:digest(training/name) for name in recipe["files_sha256"]}
    save(training/"recipe.json", recipe)
    cfg = read(training/"mechanism.json")
    cfg.update(asset_directory=str(training.relative_to(ROOT)), asset_manifest_sha256=recipe["asset_manifest_sha256"],
        full_vehicle=binding(training/"recipe.json"), leg_motor_friction=model_cfg)
    gas = cfg["v5_control"]["gas_spring_compensation"]
    gas.update(model_file=str((training/"gas_spring_model.json").relative_to(ROOT)))
    design = read(training/"design.json")
    row = design["stages"]["speed"]["roles"]["train"]
    for key in ("asset_directory", "asset_manifest_sha256", "full_vehicle", "leg_motor_friction", "v5_control"):
        row["contract"][key] = deepcopy(cfg[key])
    row["runtime"].update(leg_motor_friction=model_cfg, gas_spring_compensation=deepcopy(gas))
    row["runtime"]["modules"]["leg_motor_friction"] = True
    save(training/"design.json", design)
    cfg["design_preflight"] = binding(training/"design.json") | {
        k:v for k,v in cfg["design_preflight"].items() if k not in ("file", "sha256")}
    save(training/"mechanism.json", cfg)
    plan = read(plan_path)
    for field, filename in (("full_vehicle", "recipe.json"), ("mechanism", "mechanism.json"), ("pose_bank", "pose_bank.json")):
        plan[field] = binding(training/filename)
    plan["gas_spring_model"] = binding(training/"gas_spring_model.json")
    plan["wheel_model"]["model"] = binding(training/"wheel_model.json")
    plan["gas_spring_feedforward"] = deepcopy(gas)
    plan["standing_odometry"]["geometry_file"] = str((training/"encoder_geometry.json").relative_to(ROOT))
    plan["leg_motor_friction"] = model_cfg
    # Validate the complete new asset before publishing the default task.
    FullRobotV202(bundle)
    from wheeled_tasks.chassis.full_vehicle_training import authenticate
    authenticate(cfg, ROOT)
    save(plan_path, plan)
    save(TARGET/"migration.json", dict(schema="v202-leg-friction-migration-v1", generator=binding(Path(__file__)),
        previous_plan=binding(TARGET/"previous_plan.json"), current_plan=binding(plan_path),
        unchanged_parent_manifest=binding(PARENT/"isaac/manifest.json"),
        unchanged_training_manifest=binding(PARENT/"training/manifest.json"),
        model=model_cfg, new_PPO=0, checkpoint_reads=0, live_training_modified=False,
        hardware_controller_modified=False, raw_CAD_modified=False, behavior_qualified=False))
    print(json.dumps(dict(bundle=str(bundle), training=str(training), model=model_cfg), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent/"RMCS/reports/dm_leg_friction_20261031")
    build(parser.parse_args().source)
