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


class PairedTargets:
    """Nearest shared winding with this asset's authenticated 40..110 branch."""
    def __init__(self, gas_model, device):
        import torch
        self.torch = torch
        self.bounds = torch.tensor([[gas_model["sides"][side]["delta_rad"][i] for i in (0,-1)]
            for side in ("left","right")],device=device,dtype=torch.float64)
        self.sign = torch.tensor([1.,-1.],device=device,dtype=torch.float64)

    def projection_cost(self, desired):
        """Squared discarded target distance in rad², per environment.

        Both axes share half of the relative-angle correction. Common rotation
        (including a shared 2*pi winding) is feasible and has zero cost. Keeping
        this independent of measured state prevents physical tracking lag from
        being mistaken for an unreachable policy request.
        """
        wanted = desired.double().unflatten(-1,(2,2))
        delta = self.sign*(wanted[...,1]-wanted[...,0])
        discarded = delta-delta.clamp(self.bounds[:,0],self.bounds[:,1])
        return (.5*discarded.square().sum(-1)).to(desired.dtype)

    def __call__(self, desired, current):
        torch = self.torch
        if desired.shape!=current.shape or desired.shape[-1]!=4:
            raise ValueError("Paired targets require matching four-joint tensors")
        wanted = desired.double().unflatten(-1,(2,2))
        measured = current.double().unflatten(-1,(2,2))
        mean = wanted.mean(-1)
        delta = (self.sign*(wanted[...,1]-wanted[...,0])).clamp(self.bounds[:,0],self.bounds[:,1])
        turns = torch.floor((measured.mean(-1)-mean)/(2.*torch.pi)+.5)
        hip = mean-self.sign*delta*.5+turns*(2.*torch.pi)
        auxiliary = mean+self.sign*delta*.5+turns*(2.*torch.pi)
        return torch.stack((hip,auxiliary),-1).flatten(-2).to(desired.dtype)


class CollisionSupport:
    """Reset-only native FK and conservative collision clearance above a plane."""
    def __init__(self,joint_names,device):
        import torch
        self.torch = torch
        spec = json.loads((BUNDLE/"model_spec.json").read_text())
        bounds = json.loads((TABLE.parent/"collision_support.json").read_text())
        if bounds["model_spec_sha256"]!=digest(BUNDLE/"model_spec.json"):
            raise ValueError("Collision support belongs to a different CAD model")
        tensor = lambda value:torch.tensor(value,device=device,dtype=torch.float32)
        self.bounds = {b["name"]:(tensor(b["lower_m"]),tensor(b["upper_m"])) for b in bounds["bodies"]}
        self.eye = torch.eye(3,device=device)
        self.joints = []
        seen = {"base_link"}
        pending = spec["joints"].copy()
        while pending:
            progressed = False
            for joint in pending[:]:
                if joint["parent"] not in seen:
                    continue
                origin = tensor(joint["origin"])
                axis = tensor(joint.get("axis",[0.,0.,1.]))
                skew = tensor([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
                index = None if joint["type"]=="fixed" else joint_names.index(joint["name"])
                self.joints.append((joint["parent"],joint["child"],joint["type"],index,
                    origin[:3,:3],origin[:3,3],axis,axis[:,None]*axis[None,:],skew))
                seen.add(joint["child"])
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("CAD joint graph is not a rooted tree")

    def minimum_z(self,positions,root_quaternion_wxyz):
        torch = self.torch
        count = len(positions)
        rotations = {"base_link":self.eye.expand(count,3,3)}
        translations = {"base_link":torch.zeros(count,3,device=positions.device)}
        for parent,child,kind,index,origin_r,origin_p,axis,outer,skew in self.joints:
            rotation = rotations[parent]@origin_r
            translation = translations[parent]+(rotations[parent]@origin_p[:,None]).squeeze(-1)
            if kind=="prismatic":
                translation = translation+(rotation@axis[:,None]).squeeze(-1)*positions[:,index,None]
            elif kind!="fixed":
                angle = positions[:,index,None,None]
                motion = angle.cos()*self.eye+(1.-angle.cos())*outer+angle.sin()*skew
                rotation = rotation@motion
            rotations[child],translations[child] = rotation,translation
        w,x,y,z = root_quaternion_wxyz.unbind(-1)
        vertical = torch.stack((2.*(x*z-w*y),2.*(y*z+w*x),1.-2.*(x*x+y*y)),-1)
        minima = []
        for name,(lower,upper) in self.bounds.items():
            direction = (rotations[name].transpose(-2,-1)@vertical[:,:,None]).squeeze(-1)
            center,half = (upper+lower)/2.,(upper-lower)/2.
            minima.append((translations[name]*vertical).sum(-1)+(direction*center).sum(-1)
                -(direction.abs()*half).sum(-1))
        return torch.stack(minima,-1).min(-1).values
