"""Authenticated full-vehicle assets and the inherited six-motor components.

This is an asset/component entry point. The old chassis pose bank and trained
policies are not admissions for this vehicle's different mass and collisions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / "model/wheel_leg_v2.0.2/friction_v1/isaac"
ACTIVE = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullRobotV202:
    def __init__(self, bundle=DEFAULT_BUNDLE):
        self.bundle = Path(bundle).resolve()
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())
        receipt = json.loads((self.bundle.parent / "build_receipt.json").read_text())
        if receipt.get("asset_manifest_sha256") != digest(self.bundle / "manifest.json"):
            raise ValueError("Full-vehicle manifest differs from its build receipt")
        for name, expected in self.manifest["files_sha256"].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle) or digest(path) != expected:
                raise ValueError(f"Full-vehicle asset dependency changed: {name}")
        self.spec = json.loads((self.bundle / "model_spec.json").read_text())
        self.runtime = json.loads((self.bundle / "runtime.json").read_text())
        self.limits = json.loads((self.bundle / "mechanical_limits.json").read_text())
        self.prior = json.loads((self.bundle / self.runtime["control_file"]).read_text())
        self.spec_sha256 = digest(self.bundle / "model_spec.json")
        if (self.manifest.get("model_kind") != "full_robot_v202_gas_spring_closedchain"
                or self.manifest.get("gimbal_mode") != "fixed_zero"
                or tuple(self.manifest["control_joint_names"]) != ACTIVE
                or self.runtime["physics_dt"] != .001 or self.runtime["policy_dt"] != .01
                or self.runtime["low_level_extra_delay_ms"] != 0.):
            raise ValueError("Unexpected v2.0.2 topology or control contract")
        for filename in ("encoder_geometry.json", "gas_spring_model.json"):
            if json.loads((self.bundle / filename).read_text())["model_spec_sha256"] != self.spec_sha256:
                raise ValueError(f"Full-vehicle geometry binding differs: {filename}")
        encoder = json.loads((self.bundle / "encoder_geometry.json").read_text())
        spring = json.loads((self.bundle / "gas_spring_model.json").read_text())
        inner = self.limits["knee_inner_limits_deg"]
        zero = self.limits["hardware_zero_reference_inner_deg"]
        if (self.manifest["knee_inner_limits_deg"] != inner
                or encoder["passive_knee_domain_deg"] != inner
                or spring["geometry_beta_domain_deg"] != inner
                or self.manifest["hardware_zero_reference_inner_deg"] != zero
                or encoder["hardware_zero_reference_inner_deg"] != zero
                or self.runtime["v5_control"]["gas_spring_compensation"]["relative_soft_limits"]
                    != self.limits["relative_soft_limits"]):
            raise ValueError("Full-vehicle knee stops, zero reference or soft limits differ")
        for name, sign in (("L_joint2", 1.), ("R_jonit2", -1.)):
            expected = sorted((math.radians(v)-self.spec["passive_knee_zero_inner_rad"][name])/sign for v in inner)
            joint = next(j for j in self.spec["joints"] if j["name"] == name)
            if any(not math.isclose(float(joint["limit"][key]), value, rel_tol=0., abs_tol=1e-10)
                   for key, value in zip(("lower", "upper"), expected)):
                raise ValueError("CAD passive knee limits differ from the physical included-angle stops")
        wheel = json.loads((self.bundle / "wheel_model.json").read_text())
        if wheel["asset_rebinding"]["model_spec_sha256"] != self.spec_sha256:
            raise ValueError("Full-vehicle wheel inertia binding differs")
        from .leg_motor_friction import load
        self.leg_friction_settings, _ = load(self.runtime.get("leg_motor_friction"), ROOT)
        if bool(self.manifest.get("runtime_leg_motor_friction_required")) != self.leg_friction_settings["enabled"]:
            raise ValueError("Full-vehicle leg friction is missing from its runtime")

    def leg_friction(self, count, device="cpu"):
        from .leg_motor_friction import LegMotorFriction
        return LegMotorFriction(count, device, self.runtime.get("leg_motor_friction"), root=ROOT)

    def encoder_geometry(self, device="cpu"):
        from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
        table = json.loads((self.bundle / self.runtime["encoder_geometry_file"]).read_text())
        return EncoderLegGeometry(table, expected_spec_sha256=self.spec_sha256).to(device)

    def components(self, count, device="cpu"):
        from .identified_wheel_response import IdentifiedWheelResponse
        from .v5_control import V5Control
        settings = deepcopy(self.runtime["v5_control"])
        settings["gas_spring_compensation"]["model_file"] = str(self.bundle / "gas_spring_model.json")
        wheels = deepcopy(self.runtime["identified_wheel_response"])
        wheels["model_path"] = str(self.bundle / "wheel_model.json")
        # The legacy polynomial is never evaluated when measured compensation
        # is active; retain the inherited curve as a constructor dependency.
        fit = json.loads((ROOT / "model/纯底盘_v7_flat_base107/urdf/fit_10mpa.json").read_text())
        control = V5Control(self.manifest, self.spec, fit, self.prior, settings, device, root=ROOT)
        response = IdentifiedWheelResponse(count, device, self.runtime["physics_dt"], wheels, root=ROOT)
        return control, response
