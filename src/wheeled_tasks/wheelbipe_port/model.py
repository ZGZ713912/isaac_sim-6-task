"""Name correspondence and reset geometry; native CAD coordinates stay intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "third_party/wheelbipe"
BUNDLE = ROOT / "model/wheel_leg_v2.0.2/friction_v1/isaac"
TABLE = ROOT / "model/wheelbipe_port_v202/reset_kinematics.json"
SOURCE_COMMIT = "0b62556b34977cc629fb5767e1cda9daa0d4bf36"
JOINT_ALIASES = {
    "L_joint1": "left_rear1_joint", "R_joint1": "right_rear1_joint",
    "LL_joint1": "left_front1_joint", "RR_joint1": "right_front1_joint",
    "L_joint2": "left_rear2_joint", "R_jonit2": "right_rear2_joint",
    "LL_joint2": "left_front2_joint", "RR_joint2": "right_front2_joint",
    "LL_joint3": "left_front3_joint", "RR_joint3": "right_front3_joint",
    "LL_link4": "left_front4_joint", "RR_joint4": "right_front4_joint",
    "L_joint3": "left_wheel_joint", "R_joint3": "right_wheel_joint",
    "LLL_joint2": "left_spring1_joint", "RRR_joint2": "right_spring1_joint",
    "L_spring_slide": "left_spring2_joint", "R_spring_slide": "right_spring2_joint",
}
BODY_ALIASES = {
    "base_link": "base_link", "M_link1": "gimbal_yaw_link", "M_link2": "gimbal_pitch_link",
    "L_link1": "left_rear1_link", "R_link1": "right_rear1_link",
    "L_link2": "left_rear2_link", "R_link2": "right_rear2_link",
    "LL_link1": "left_front1_link", "RR_link1": "right_front1_link",
    "LL_link2": "left_front2_link", "RR_link2": "right_front2_link",
    "LL_link3": "left_front3_link", "RR_link3": "right_front3_link",
    "LL_link4": "left_front4_link", "RR_link4": "right_front4_link",
    "L_link3": "left_wheel_link", "R_link3": "right_wheel_link",
    "LLL_link2": "left_spring1_link", "RRR_link2": "right_spring1_link",
    "LLL_link1": "left_spring2_link", "RRR_link1": "right_spring2_link",
}
ACTIVE = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticate_source():
    manifest = json.loads((VENDOR / "provenance.json").read_text())
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Unexpected Wheelbipe source revision")
    for relative, expected in manifest["files_sha256"].items():
        path = (VENDOR / relative).resolve()
        if not path.is_relative_to(VENDOR) or digest(path) != expected:
            raise ValueError("Wheelbipe source changed: " + relative)
    return manifest


def expose_source():
    authenticate_source()
    for path in (ROOT / "src", VENDOR, VENDOR / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class ResetLookup:
    """CAD branch lookup, with independent left/right length and common rotation.

    Only reset uses this table. It neither injects states during dynamics nor
    supplies actor observations. Its interpolation error is measured by the
    builder against all six original constraint equations.
    """
    def __init__(self, joint_names, device):
        import torch
        table = json.loads(TABLE.read_text())
        if table["model_spec_sha256"] != digest(BUNDLE / "model_spec.json"):
            raise ValueError("Reset lookup belongs to a different CAD model")
        self.torch = torch
        self.names = table["joint_names"]
        self.native_order = [self.names.index(n) for n in joint_names]
        self.rows = torch.tensor(table["positions"], device=device, dtype=torch.float32)
        self.sides = table["sides"]
        self.lengths, self.angles = [], []
        for side in self.sides:
            self.lengths.append(torch.tensor(side["lengths_m"], device=device))
            self.angles.append(torch.tensor(side["angles_rad"], device=device))

    def sample(self, length, angle):
        torch = self.torch
        result = self.rows[0].expand(len(length), -1).clone()
        for k, side in enumerate(self.sides):
            knots = self.lengths[k]
            requested = length[:, k].clamp(knots[0], knots[-1])
            upper = torch.searchsorted(knots, requested.contiguous()).clamp(1, len(knots)-1)
            lower = upper-1
            blend = (requested-knots[lower])/(knots[upper]-knots[lower])
            q = torch.lerp(self.rows[lower], self.rows[upper], blend[:, None])
            current_angle = torch.lerp(self.angles[k][lower], self.angles[k][upper], blend)
            rotation = (angle[:, k]-current_angle)/side["axis_y_sign"]
            for name in side["active_pair"]:
                q[:, self.names.index(name)] += rotation
            columns = side["joint_columns"]
            result[:, columns] = q[:, columns]
        return result[:, self.native_order]
