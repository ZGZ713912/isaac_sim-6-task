"""Authenticated full-vehicle assets and the inherited six-motor components.

This is an asset/component entry point. The old chassis pose bank and trained
policies are not admissions for this vehicle's different mass and collisions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / "model/wheel_leg_v2.0.2/isaac"
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
        wheel = json.loads((self.bundle / "wheel_model.json").read_text())
        if wheel["asset_rebinding"]["model_spec_sha256"] != self.spec_sha256:
            raise ValueError("Full-vehicle wheel inertia binding differs")

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
