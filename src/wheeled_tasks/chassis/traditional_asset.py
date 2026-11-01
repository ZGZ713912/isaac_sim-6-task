"""Authenticate the supplied v2.0.2 asset without rebuilding its frames."""
from pathlib import Path
import json

import numpy as np

from .full_robot_asset import ACTIVE, SPRINGS, MODEL_KIND, digest


class TraditionalAsset:
    def __init__(self, bundle):
        self.bundle = Path(bundle).resolve()
        self.manifest = json.loads((self.bundle/"manifest.json").read_text())
        receipt = json.loads((self.bundle.parent/"build_receipt.json").read_text())
        if receipt["asset_manifest_sha256"] != digest(self.bundle/"manifest.json"):
            raise ValueError("v2.0.2 build receipt differs from manifest")
        for name, checksum in self.manifest["files_sha256"].items():
            path = (self.bundle/name).resolve()
            if not path.is_relative_to(self.bundle) or digest(path) != checksum:
                raise ValueError("v2.0.2 asset dependency changed: "+name)
        self.spec = json.loads((self.bundle/"model_spec.json").read_text())
        self.limits = json.loads((self.bundle/"mechanical_limits.json").read_text())
        self.runtime = json.loads((self.bundle/"runtime.json").read_text())
        self.curve = json.loads((self.bundle/"gas_spring_model.json").read_text())
        self.wheel = json.loads((self.bundle/"wheel_model.json").read_text())
        from .leg_motor_friction import load
        self.leg_friction_settings, _ = load(self.runtime.get("leg_motor_friction"), self.bundle.parents[2])
        if bool(self.manifest.get("runtime_leg_motor_friction_required")) != self.leg_friction_settings["enabled"]:
            raise ValueError("Traditional plant leg moving friction is missing from its runtime")
        spec_hash = digest(self.bundle/"model_spec.json")
        if (self.manifest["model_kind"] != MODEL_KIND
            or self.manifest["control_frame"] != "Xforward_Yleft_Zup"
            or self.manifest["gimbal_mode"] != "fixed_zero"
            or self.manifest["control_joint_names"] != ACTIVE
            or self.manifest["spring_joint_names"] != SPRINGS
            or len(self.manifest["rigid_body_names"]) != 21
            or len(self.manifest["tree_joint_names"]) != 18
            or self.manifest["fixed_joint_names"] != ["M_joint1", "M_joint2"]):
            raise ValueError("Unexpected v2.0.2 frame/topology")
        for name in ("encoder_geometry.json", "gas_spring_model.json"):
            if json.loads((self.bundle/name).read_text())["model_spec_sha256"] != spec_hash:
                raise ValueError("v2.0.2 geometry binding differs: "+name)
        if self.wheel["asset_rebinding"]["model_spec_sha256"] != spec_hash:
            raise ValueError("v2.0.2 wheel inertia binding differs")
        if len(self.spec["constraints"]) != 6 or self.spec["control_frame"] != "Xforward_Yleft_Zup":
            raise ValueError("Expected six canonical loop closures")
        moving = [j["name"] for j in self.spec["joints"] if j["type"] != "fixed"]
        if moving != self.manifest["tree_joint_names"]:
            raise ValueError("v2.0.2 moving joint identity differs")
        for side in ("L", "R"):
            body = next(b for b in self.spec["bodies"] if b["name"] == side+"_link3")
            axis = np.asarray(next(j for j in self.spec["joints"] if j["name"] == side+"_joint3")["axis"])
            inertia = float(axis@np.asarray(body["inertia"])@axis)
            row = self.wheel["axes"][side+"_joint3"]
            if not np.isclose(inertia, row["cad_inertia_kg_m2"], atol=1e-12, rtol=0.):
                raise ValueError("v2.0.2 wheel CAD inertia differs")

    def spring_component(self, root, device="cpu"):
        from .gas_spring_compensation import GasSpringCompensation
        settings = dict(self.runtime["v5_control"]["gas_spring_compensation"])
        settings["model_file"] = str(self.bundle/"gas_spring_model.json")
        return GasSpringCompensation(settings, root=root, device=device)

    def leg_friction(self, root, device="cpu"):
        from .leg_motor_friction import LegMotorFriction
        return LegMotorFriction(1, device, self.leg_friction_settings, root=root)
