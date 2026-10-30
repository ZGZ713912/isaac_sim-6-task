#!/usr/bin/env python3
"""Derive a moving-gimbal training fixture without rewriting the v2.0.2 CAD bundle."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202, digest
from wheeled_tasks.chassis.gimbal_motion import motion_contract

module_spec = importlib.util.spec_from_file_location("full_vehicle_builder_geometry", ROOT/"tools/full_robot_v202.py")
geometry = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(geometry)
DIRECTORY = ROOT/"model/wheel_leg_v2.0.2/training"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


class ResetGeometry:
    def __init__(self, spec, source):
        self.spec, self.source = spec, source
        self.joints = {j["name"]: j for j in spec["joints"]}
        self.meshes = {c["file"]: np.array([[float(x) for x in line.split()[1:4]]
            for line in (source/c["file"]).read_text().splitlines() if line.startswith("v ")])
            for b in spec["bodies"] for c in b["collisions"] if c["type"] == "mesh"}

    def solve(self, prescribed, initial):
        return geometry.solve_pose(self.spec, {**prescribed, "M_joint1": 0., "M_joint2": 0.}, initial)

    def knee(self, beta):
        q = self.spec["nominal_joint_pos"]
        zero = self.spec["passive_knee_zero_inner_rad"]
        start = math.degrees(q["L_joint2"]+zero["L_joint2"])
        for b in np.linspace(start, beta, max(2, math.ceil(abs(beta-start)/3.)+1))[1:]:
            q = self.solve(dict(L_joint1=q["L_joint1"], R_joint1=q["R_joint1"],
                L_joint3=0., R_joint3=0., L_joint2=math.radians(b)-zero["L_joint2"],
                R_jonit2=zero["R_jonit2"]-math.radians(b)), q)
        return q

    def rotate(self, q, left, right):
        fixed = {n: q[n] for n in geometry.ACTIVE}
        for side, angle in (("L", left), ("R", right)):
            for name in (side+"_joint1", side*2+"_joint1"):
                fixed[name] += angle
        return self.solve(fixed, q)

    def axis_sign(self, side):
        j = self.joints[side+"_joint1"]
        return float(np.sign((np.array(j["origin"])[:3, :3]@j["axis"])[1]))

    def bounds(self, q, rotation):
        frames = geometry.fk(self.spec, q)
        rows = []
        for b in self.spec["bodies"]:
            for c in b["collisions"]:
                t = frames[b["name"]]@np.array(c["origin"])
                rot, pos = rotation@t[:3, :3], rotation@t[:3, 3]
                if c["type"] == "mesh":
                    minimum = (self.meshes[c["file"]]@rot.T+pos).min(0)
                else:
                    a = rot[:, 2]
                    minimum = pos-c["radius"]*np.sqrt(np.maximum(0., 1.-a*a))-c["length"]*.5*np.abs(a)
                rows.append((b["name"], minimum))
        return rows

    def balanced(self, beta):
        initial = self.knee(beta)
        def pose(angle):
            return self.rotate(initial, angle/self.axis_sign("L"), angle/self.axis_sign("R"))
        def error(angle):
            frames = geometry.fk(self.spec, pose(angle))
            center = sum(b["mass"]*geometry.point(frames[b["name"]], b["com"])
                for b in self.spec["bodies"])/sum(b["mass"] for b in self.spec["bodies"])
            return center[0]-(frames["L_link3"][0, 3]+frames["R_link3"][0, 3])/2.
        q = pose(brentq(error, -.8, .8, xtol=1e-10))
        frames = geometry.fk(self.spec, q)
        delta = frames["L_link3"][:3, 3]-frames["R_link3"][:3, 3]
        rotation = Rotation.from_euler("x", math.atan2(-delta[2], delta[1])).as_matrix()
        bounds = self.bounds(q, rotation)
        height = -min(v[2] for name, v in bounds if name in ("L_link3", "R_link3"))
        return q, rotation, float(height), bounds

    def bank(self, manifest_sha):
        poses = []
        for name, height in (("stand_low", .23), ("stand", .305), ("stand_high", .42)):
            beta = brentq(lambda b: self.balanced(b)[2]-height, 40.001, 106.999, xtol=1e-7)
            q, rot, actual, bounds = self.balanced(beta)
            poses.append(self.pose(name, q, rot, actual, min(v[2]+actual for _, v in bounds)))
        # Same geometric construction as the chassis birth: aligned103deg legs,
        # then a1.6rad paired orbit. This solves the new linkage, not old q values.
        q = self.knee(103.)
        frames = geometry.fk(self.spec, q)
        angles = []
        for side in ("L", "R"):
            v = frames[side+"_link3"][:3, 3]-np.array(self.joints[side+"_joint1"]["origin"])[:3, 3]
            angles.append(math.atan2(v[0], -v[2])/self.axis_sign(side))
        q = self.rotate(q, *angles)
        q = self.rotate(q, 1.6, -1.6)
        minimum = min(v[2] for _, v in self.bounds(q, np.eye(3)))
        poses.append(self.pose("floor_knee103_103_orbit1.6_r0_p0", q, np.eye(3), -minimum+.008, .008))
        return dict(schema="wheel-leg-unified-reset-bank-v1", asset_directory=str(DIRECTORY.relative_to(ROOT)),
            asset_manifest_sha256=manifest_sha, model_spec_sha256=digest(DIRECTORY/"model_spec.json"),
            tree_joint_names=[j["name"] for j in self.spec["joints"]], poses=poses,
            state_writes="episode_reset_only", motor_effort_during_settle="zero_all_eight_axes",
            self_collision_scope="inherited disabled self-collision; no new self-collision qualification",
            diagnostic_height_starts=dict(mapping={"stand_low": 0, "stand": 1, "stand_high": 2}),
            geometry_scope="new full-vehicle closed-chain, COM-balanced wheel-supported births; no static PD targets or policy weights")

    def pose(self, name, q, rotation, height, clearance):
        gap = float(np.linalg.norm(geometry.closure_error(self.spec, q).reshape(-1, 3), axis=1).max())
        zero = self.spec["passive_knee_zero_inner_rad"]
        knees = [math.degrees(q["L_joint2"]+zero["L_joint2"]),
                 math.degrees(zero["R_jonit2"]-q["R_jonit2"])]
        if gap > 1e-7 or clearance < -1e-7 or any(not 40.-1e-5 <= b <= 107.+1e-5 for b in knees):
            raise ValueError(f"Invalid full-vehicle birth: {name}, gap={gap}, clearance={clearance}, knees={knees}")
        for n, binding in self.spec["spring_binding"].items():
            s = binding["compression_at_q_zero_m"]-q[n]
            if not -1e-8 <= s <= binding["stroke_m"]+1e-8:
                raise ValueError("Full-vehicle reset spring travel violated")
        return dict(name=name, family="upright", roll_deg=float(Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)[0]),
            pitch_deg=0., root_position_m=[0., 0., height], root_quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
            full_joint_positions=q, min_collision_clearance_m=clearance, knee_deg=knees, gap_m=gap)


def main():
    if DIRECTORY.exists():
        raise ValueError("Training derivative already exists; preserve it before an intentional rebuild")
    asset = FullRobotV202()
    DIRECTORY.mkdir()
    spec = deepcopy(asset.spec)
    for j in spec["joints"]:
        if j["name"] == "M_joint1":
            j.update(type="continuous", limit=dict(effort="2", velocity=str(math.pi)))
        if j["name"] == "M_joint2":
            j.update(type="revolute", limit=dict(lower=str(math.radians(-30.)), upper=str(math.radians(35.)), effort="10", velocity="100"))
    save(DIRECTORY/"model_spec.json", spec)
    spec_sha = digest(DIRECTORY/"model_spec.json")
    for name in ("encoder_geometry.json", "gas_spring_model.json", "wheel_model.json"):
        data = json.loads((asset.bundle/name).read_text())
        if name == "wheel_model.json":
            data["asset_rebinding"]["model_spec_sha256"] = spec_sha
        else:
            data["model_spec_sha256"] = spec_sha
        save(DIRECTORY/name, data)
    shutil.copytree(asset.bundle/"collisions", DIRECTORY/"collisions")
    shutil.copy2(ROOT/"model/纯底盘_v7_flat_base107/urdf/fit_10mpa.json", DIRECTORY/"fit_10mpa.json")
    # USD layer references preserve all native bodies, axes, closed-loop anchors,
    # collision hulls, visual meshes and the actual CAD gimbal-axis inclination.
    (DIRECTORY/"robot.usda").write_text('#usda 1.0\n(\n    defaultPrim = "Robot"\n    subLayers = [@../isaac/robot_gimbal_free.usda@]\n)\n')
    save(DIRECTORY/"gimbal_motion.json", motion_contract())
    manifest = deepcopy(asset.manifest)
    manifest.update(asset_revision="full_robot_v202_moving_gimbal_training_v1", usd="robot.usda",
        tree_joint_names=[j["name"] for j in spec["joints"]], fixed_joint_names=[],
        gimbal_mode="independent_servo", training_ready=False, hardware_ready=False)
    for k in ("gimbal_free_usd", "gimbal_limits", "urdf"):
        manifest.pop(k, None)
    manifest["files_sha256"] = {str(p.relative_to(DIRECTORY)): digest(p) for p in sorted(DIRECTORY.rglob("*")) if p.is_file()}
    save(DIRECTORY/"manifest.json", manifest)
    bank = ResetGeometry(spec, asset.bundle).bank(digest(DIRECTORY/"manifest.json"))
    save(DIRECTORY/"pose_bank.json", bank)
    reference = ROOT.parent/"example/wheeled-legged_RL/src/agent_world/assets/wheelbipe_V14_2.py"
    envcfg = ROOT.parent/"example/wheeled-legged_RL/src/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py"
    recipe = dict(schema="full-vehicle-training-derivative-v1", source_bundle=str(asset.bundle.relative_to(ROOT)),
        source_manifest_sha256=digest(asset.bundle/"manifest.json"), source_spec_sha256=asset.spec_sha256,
        model_spec_sha256=spec_sha, asset_manifest_sha256=digest(DIRECTORY/"manifest.json"),
        files_sha256={name: digest(DIRECTORY/name) for name in ("manifest.json", "pose_bank.json", "gimbal_motion.json")},
        parent_component_validation=dict(file="reports/full_robot_v202_20261030/validation_receipt.json",
            sha256=digest(ROOT/"reports/full_robot_v202_20261030/validation_receipt.json")),
        zero_friction_bodies=["base_link"], zero_friction_combine_mode="min",
        zero_friction_scope="all8 merged base collision hulls include belly guide rollers; driving wheels and other bodies retain their materials",
        references=[dict(file=str(p), sha256=digest(p)) for p in (reference, envcfg)],
        allowed_changes="release two gimbal joints; bind unchanged chassis component tables to this derived specification; new reset-only geometry",
        actor_dim=35, critic_dim=87, action_dim=6, policy_abi="v4 local wheel standing odometry; independent gimbal disturbance is not a policy output",
        hardware_ready=False)
    save(DIRECTORY/"recipe.json", recipe)
    cfg = json.loads((ROOT/"model/gas_spring_compensation_unified_20261029/mechanism.json").read_text())
    cfg.update(contract_id="full-robot-v202-gas-spring-training-v1", asset_directory=str(DIRECTORY.relative_to(ROOT)),
        asset_manifest_sha256=digest(DIRECTORY/"manifest.json"), critic_dim=87,
        full_vehicle=dict(file=str((DIRECTORY/"recipe.json").relative_to(ROOT)), sha256=digest(DIRECTORY/"recipe.json")),
        gimbal_motion=motion_contract(), zero_friction_bodies=["base_link"])
    cfg["critic_layout"] = [s.replace("q18", "q20").replace("dq18", "dq20") for s in cfg["critic_layout"]]
    cfg["v5_control"] = deepcopy(asset.runtime["v5_control"])
    cfg["v5_control"]["gas_spring_compensation"].update(model_file=str((DIRECTORY/"gas_spring_model.json").relative_to(ROOT)),
        model_sha256=digest(DIRECTORY/"gas_spring_model.json"))
    design = json.loads((ROOT/"model/gas_spring_compensation_unified_20261029/design.json").read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    for key in ("contract_id", "asset_directory", "asset_manifest_sha256", "critic_dim", "critic_layout", "v5_control",
                "full_vehicle", "gimbal_motion", "zero_friction_bodies"):
        row["contract"][key] = deepcopy(cfg[key])
    row["runtime"]["gas_spring_compensation"] = deepcopy(cfg["v5_control"]["gas_spring_compensation"])
    row["runtime"]["gimbal_motion"] = motion_contract()
    row["runtime"]["zero_friction_bodies"] = ["base_link"]
    design["scope"] = "full_vehicle_v202_moving_gimbal_fresh_training_mechanism"
    save(DIRECTORY/"design.json", design)
    cfg["design_preflight"].update(file=str((DIRECTORY/"design.json").relative_to(ROOT)), sha256=digest(DIRECTORY/"design.json"))
    save(DIRECTORY/"mechanism.json", cfg)
    print(json.dumps(dict(directory=str(DIRECTORY), asset_manifest_sha256=recipe["asset_manifest_sha256"],
        poses=len(bank["poses"]), mass_kg=manifest["total_mass_kg"], gimbal_limits_deg=[-30., 35.])))


if __name__ == "__main__":
    main()
