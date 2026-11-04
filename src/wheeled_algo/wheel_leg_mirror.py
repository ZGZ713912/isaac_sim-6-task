"""Independent model-geometry reconstruction for a read-only real robot mirror.

No physics step, controller, ROS publisher, or learned policy is involved.
Only the calibrated active coordinates are prescribed; all passive coordinates
come from the CAD joint tree and six loop-anchor constraints.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = ROOT / "contracts/wheel_leg_live_mirror_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_file(path, expected):
    path = Path(path).resolve()
    if digest(path) != expected:
        raise ValueError(f"asset/mapping identity mismatch: {path}")
    return path


def load_telemetry_reader(rmcs_root):
    """Reuse RMCS's protocol implementation without requiring a ROS install."""
    import importlib.util
    import sys
    path = Path(rmcs_root) / "rmcs_ws/src/rmcs_rl/tool/visualization/wheel_leg_telemetry.py"
    if not path.is_file():
        raise ValueError(f"RMCS telemetry reader missing: {path}; set --rmcs-root")
    name = "wheel_leg_telemetry_reader"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MirrorModel:
    def __init__(self, mapping=DEFAULT_MAPPING):
        self.mapping_path = Path(mapping).resolve()
        self.mapping = json.loads(self.mapping_path.read_text())
        if self.mapping.get("schema") != "wheel_leg_live_mirror_v1":
            raise ValueError("unsupported mirror mapping schema")
        asset = self.mapping["asset"]
        self.bundle = (ROOT / asset["bundle"]).resolve()
        manifest_path = checked_file(self.bundle / "manifest.json", asset["manifest_sha256"])
        self.manifest = json.loads(manifest_path.read_text())
        for name, sha in self.manifest["files_sha256"].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle):
                raise ValueError("asset manifest path escapes the bundle")
            checked_file(path, sha)
        spec_path = checked_file(self.bundle / "model_spec.json", asset["model_spec_sha256"])
        self.spec = json.loads(spec_path.read_text())
        self.visuals = checked_file(self.bundle / asset["visuals"], asset["visuals_sha256"])
        if self.spec["control_frame"] != "Xforward_Yleft_Zup":
            raise ValueError("unsupported model frame")
        self.metadata = dict(asset_manifest_sha256=digest(manifest_path),
                             model_spec_sha256=digest(spec_path),
                             mapping_sha256=digest(self.mapping_path),
                             visuals_sha256=digest(self.visuals))
        self.entries = self.mapping["joints"]
        sources = [e["source"] for e in self.entries]
        targets = [e["target"] for e in self.entries]
        if len(sources) != 6 or len(set(sources)) != 6 or len(set(targets)) != 6:
            raise ValueError("mapping requires six distinct source and target names")
        if set(targets) != set(self.manifest["control_joint_names"]):
            raise ValueError("mapped active joints differ from the model manifest")
        if not all(e["sign"] in (-1, 1) and math.isfinite(e["offset_rad"]) for e in self.entries):
            raise ValueError("invalid joint sign/offset")
        self.previous = dict(self.spec["nominal_joint_pos"])
        self.targets = set(targets)
        self.unmeasured = set(self.manifest["tree_joint_names"]) - self.targets
        self.unknown = [j["name"] for j in self.spec["joints"] if j["name"] not in self.targets
                        and j["name"] not in ("M_joint1", "M_joint2") and j["type"] != "fixed"]
        self.lower = np.array([float(self._joint(n)["limit"].get("lower", -np.inf)) for n in self.unknown])
        self.upper = np.array([float(self._joint(n)["limit"].get("upper", np.inf)) for n in self.unknown])
        self.joints = []
        available, pending = {"base_link"}, list(self.spec["joints"])
        while pending:
            ready = [j for j in pending if j["parent"] in available]
            if not ready:
                raise ValueError("disconnected model joint tree")
            for j in ready:
                axis = np.asarray(j["axis"], dtype=float)
                x, y, z = axis
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                self.joints.append((j, np.asarray(j["origin"]), axis, skew, skew @ skew))
                available.add(j["child"])
                pending.remove(j)
        self.body_names = set(self.manifest["rigid_body_names"])
        if available != self.body_names:
            raise ValueError("model body names differ from its joint tree")
        self.epoch = None

    def _joint(self, name):
        return next(j for j in self.spec["joints"] if j["name"] == name)

    def fk(self, q, root=None):
        frames = {"base_link": np.eye(4) if root is None else root}
        for j, origin, axis, skew, square in self.joints:
            motion = np.eye(4)
            value = q.get(j["name"], 0.)
            if j["type"] == "prismatic":
                motion[:3, 3] = axis * value
            elif j["type"] != "fixed":
                motion[:3, :3] += math.sin(value) * skew + (1. - math.cos(value)) * square
            frames[j["child"]] = frames[j["parent"]] @ origin @ motion
        return frames

    def closure(self, q):
        frames = self.fk(q)
        return np.concatenate([
            frames[c["body0"]][:3, :3] @ c["local_pos0_m"] + frames[c["body0"]][:3, 3]
            - frames[c["body1"]][:3, :3] @ c["local_pos1_m"] - frames[c["body1"]][:3, 3]
            for c in self.spec["constraints"]])

    def validate_source(self, snapshot):
        snapshot.validate()
        for name, count in (("velocities", 6), ("rmcs_knee", 2), ("rmcs_length", 2),
                            ("rmcs_model_positions", 6), ("model_offsets", 4)):
            if len(getattr(snapshot, name)) != count:
                raise ValueError(f"missing mirror reference: {name}")
        calibration = self.mapping["calibration"]
        for name in ("calibration_sha256", "calibration_model_spec_sha256"):
            if getattr(snapshot, name) != calibration[name]:
                raise ValueError(f"real robot {name} differs from mirror mapping")
        offsets = {e["source"]: e["offset_rad"] for e in self.entries if e["sign"] == -1}
        if snapshot.offsets_by_name != offsets:
            raise ValueError("real robot model offsets differ from mirror mapping")
        if len(snapshot.sensor_age_s) != 7 or any(age < 0 or age > .04 for age in snapshot.sensor_age_s):
            raise ValueError("real robot sensor feedback is missing or older than 40 ms")

    def reconstruct(self, snapshot, base_position=(0., 0., .65)):
        self.validate_source(snapshot)
        warnings = []
        if not snapshot.feedback_fresh:
            warnings.append("RMCS aggregate feedback_fresh is false; individual sensor ages checked")
        if not snapshot.state_valid:
            warnings.append("RMCS estimator invalid; its comparison values are unavailable")
        source = snapshot.positions_by_name
        if set(source) != {e["source"] for e in self.entries}:
            raise ValueError("telemetry joint names differ from the explicit mapping")
        measured = {e["target"]: e["sign"] * source[e["source"]] + e["offset_rad"] for e in self.entries}
        prescribed = {name: math.remainder(value, 2 * math.pi) for name, value in measured.items()}
        for side in self.mapping["legs"]:
            hip, auxiliary, sign = side["hip"], side["auxiliary"], side["relative_sign"]
            low, high = side["relative_limits_rad"]
            mid = .5 * (low + high)
            delta = mid + math.remainder(sign * (measured[auxiliary] - measured[hip]) - mid, 2 * math.pi)
            if not low - 1e-8 <= delta <= high + 1e-8:
                warnings.append(f"{side['name']} encoders outside nominal relative range: {delta:.6f} rad")
            prescribed[auxiliary] = prescribed[hip] + sign * delta
        # Gimbal feedback is absent. Its nominal pose is explicitly labeled in
        # the GUI; it is never presented as a real measurement.
        prescribed.update(M_joint1=0., M_joint2=0.)
        previous = self.previous if self.epoch == snapshot.reset_count else self.spec["nominal_joint_pos"]
        q = {**previous, **prescribed}
        def residual(values):
            return self.closure({**q, **dict(zip(self.unknown, values))})
        # This is a measurement viewer: nominal mechanical/soft limits are
        # diagnostics, never clamps or blockers. Only CAD loop closure is solved.
        initial = [q[n] for n in self.unknown]
        result = least_squares(residual, initial,
                               xtol=1e-11, ftol=1e-11, gtol=1e-13, max_nfev=80)
        error = float(np.max(np.abs(residual(result.x))))
        if not result.success or error > 2e-6:
            raise ValueError(f"closed-chain reconstruction failed ({error * 1000:.4f} mm)")
        q.update(zip(self.unknown, result.x))
        for i, name in enumerate(self.unknown):
            if not self.lower[i] - 1e-4 <= q[name] <= self.upper[i] + 1e-4:
                warnings.append(f"{name} outside nominal model limit: {q[name]:.6f}")
        frames = self.fk(q)
        knees, lengths, wheels = [], [], []
        for side in self.mapping["legs"]:
            hip, knee, wheel = (frames[n][:3, 3] for n in side["points"])
            first, second = hip - knee, wheel - knee
            beta = math.acos(float(np.clip(np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second), -1., 1.)))
            low, high = np.deg2rad(self.manifest["knee_inner_limits_deg"])
            if not low - 1e-4 <= beta <= high + 1e-4:
                warnings.append(f"{side['name']} passive knee outside nominal range: {math.degrees(beta):.3f} deg")
            knees.append(beta)
            lengths.append(float(np.linalg.norm(wheel - hip)))
            wheels.append(wheel.tolist())
        root = np.eye(4)
        w, x, y, z = snapshot.quaternion_wxyz
        root[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        root[:3, 3] = base_position
        gravity = root[:3, :3].T @ np.array([0., 0., -1.])
        gravity_error = float(np.linalg.norm(gravity - snapshot.gravity_rl)) if snapshot.state_valid else None
        coordinate_error = [math.remainder(measured[e["target"]] - snapshot.model_positions_by_name[e["source"]], 2 * math.pi)
                            for e in self.entries]
        self.previous, self.epoch = q, snapshot.reset_count
        return dict(joint_positions=measured,
                    body_frames={name: root @ pose for name, pose in frames.items()},
                    knee_rad=knees, leg_length_m=lengths, wheel_centers_body_m=wheels,
                    knee_error_rad=(np.asarray(knees) - snapshot.rmcs_knee).tolist() if snapshot.state_valid else None,
                    leg_length_error_m=(np.asarray(lengths) - snapshot.rmcs_length).tolist() if snapshot.state_valid else None,
                    coordinate_error_rad=coordinate_error if snapshot.state_valid else None, gravity_error=gravity_error,
                    reference_valid=snapshot.state_valid, diagnostic_warnings=warnings,
                    closure_error_m=error, stamp_ns=snapshot.stamp_ns,
                    fixed_base_position_m=list(base_position), unmeasured_gimbal="nominal zero")
