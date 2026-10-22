"""No-PPO check of encoder/IMU rolling kinematics against recorded body velocity.

Only the error analysis reads simulator velocity. The estimate itself uses the
six motor coordinates, IMU and sealed107 closed-chain geometry.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
TABLE = ROOT / "reports/flat_base_continuation_20261019/low_friction_candidate001/height_lookup.json"
ASSET = ROOT / "model/纯底盘_v7_flat_base107/urdf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RollingGeometry:
    def __init__(self):
        self.table = json.loads(TABLE.read_text())
        spec = json.loads((ASSET / "model_spec.json").read_text())
        assert self.table["model_spec_sha256"] == sha(ASSET / "model_spec.json") == "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"
        source = ASSET / "tools/v6_mechanism.py"
        assert sha(source) == self.table["fk_sha256"]
        module_spec = importlib.util.spec_from_file_location("rolling049_fk", source)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        frames = module.fk(spec, spec["nominal_joint_pos"])
        by_name = {j["name"]: j for j in spec["joints"]}
        self.axes = {}
        for name in ("L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3"):
            joint = by_name[name]
            self.axes[name] = (frames[joint["parent"]] @ np.asarray(joint["origin"]))[:3, :3] @ joint["axis"]
        self.nominal = np.array([spec["nominal_joint_pos"][n] for n in self.table["control_joint_names"]])

    def estimate(self, q, dq, omega, gravity):
        estimates, validities = [], []
        for name, side, i, knee in (("left", "L", 0, "L_joint2"), ("right", "R", 3, "R_jonit2")):
            t = self.table["sides"][name]
            knots, points = np.asarray(t["delta_rad"]), np.asarray(t["wheel_center_b_m"])
            delta = q[:, i + 1] - q[:, i]
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            valid = (delta >= knots[0]) & (delta <= knots[-1])
            upper = np.searchsorted(knots, delta).clip(1, len(knots) - 1)
            lower = upper - 1
            width = knots[upper] - knots[lower]
            blend = ((delta - knots[lower]) / width).clip(0., 1.)
            point = points[lower] + blend[:, None] * (points[upper] - points[lower])
            derivative = (points[upper] - points[lower]) / width[:, None]
            axis = np.asarray(t["hip_axis_b"])
            assert np.max(np.abs(axis - self.axes[side + "_joint1"])) < 1e-9
            assert np.linalg.norm(np.cross(axis, self.axes[knee])) < 1e-9
            angle = q[:, i] - t["hip_reference_rad"]
            c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]

            def rotate(v):
                return c * v + s * np.cross(axis, v) + (1. - c) * axis * np.sum(axis * v, axis=-1, keepdims=True)

            relative = rotate(point - t["hip_origin_b_m"])
            p = relative + t["hip_origin_b_m"]
            delta_velocity = dq[:, i + 1] - dq[:, i]
            dp = np.cross(axis, relative) * dq[:, i:i+1] + rotate(derivative) * delta_velocity[:, None]
            beta = np.deg2rad(t["passive_inner_deg"])
            beta_derivative = (beta[upper] - beta[lower]) / width
            knee_velocity = (1. if side == "L" else -1.) * beta_derivative * delta_velocity
            relative_angular = (axis * dq[:, i:i+1] + self.axes[knee] * knee_velocity[:, None]
                                + self.axes[side + "_joint3"] * dq[:, i+2:i+3])
            wheel_axis = np.asarray(t["wheel_axis_b"])
            downward = gravity - wheel_axis * np.sum(gravity * wheel_axis, axis=-1, keepdims=True)
            contact_offset = t["wheel_radius_m"] * downward / np.linalg.norm(downward, axis=-1, keepdims=True).clip(1e-9)
            # Contact-point no-slip equation expressed in the base-link frame.
            velocity = -np.cross(omega, p + contact_offset) - dp - np.cross(relative_angular, contact_offset)
            estimates.append(velocity[:, 0])
            validities.append(valid)
        return np.stack(estimates, -1).mean(-1), np.stack(validities, -1).all(-1)


def errors(actual, estimated, valid):
    e = estimated[valid] - actual[valid]
    return dict(samples=len(e), mae_m_s=float(np.abs(e).mean()), rms_error_m_s=float(np.square(e).mean()**.5),
                mean_error_m_s=float(e.mean()), p95_abs_error_m_s=float(np.quantile(np.abs(e), .95)))


def main():
    geometry = RollingGeometry()
    rows = []
    for name in ("quiet_velocity_baseline036", "command_feedback_positive048", "side_confirmation032", "rolling_command_feedback049"):
        directory = RUNS / name / "robust"
        report = json.loads((directory / "report.json").read_text())
        assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
        data = np.load(directory / "trajectory.npz")
        # Restrict this rolling/support assumption to ordinary upright standing.
        lanes = np.array([r["name"] in ("stand_low", "stand", "stand_high") for r in report["lanes"]])
        mask = data["alive"][1:] & (data["time"][1:] >= 3.) & lanes[None, :]
        q, dq = data["motor_position"][1:][mask], data["motor_velocity"][1:][mask]
        truth, true_valid = geometry.estimate(q, dq, data["omega"][1:][mask], data["gravity"][1:][mask])
        actual = data["velocity"][1:][mask][:, 0]
        obs = data["observations"][1:][mask]
        observed_q = np.zeros((len(obs), 6))
        observed_q[:, [0, 1, 3, 4]] = obs[:, 10:14]
        observed_q += geometry.nominal
        observed_dq = obs[:, 16:22][:, [0, 1, 4, 2, 3, 5]] * 10.
        estimated, observed_valid = geometry.estimate(observed_q, observed_dq, obs[:, 4:7] * 2., obs[:, 7:10])
        proxy = .3 * (obs[:, 20] - obs[:, 21])
        # Observations are read before a policy step; diagnostic truth after.
        actual_before = data["velocity"][:-1][mask][:, 0]
        valid = observed_valid & np.isfinite(estimated)
        row = dict(run=name, trajectory_sha256=report["trajectory_sha256"],
            valid_true_fraction=float(true_valid.mean()), valid_observed_fraction=float(observed_valid.mean()),
            true_sensor_kinematics=errors(actual, truth, true_valid),
            deployable_observation_kinematics=errors(actual_before, estimated, valid),
            original_wheel_proxy=errors(actual_before, proxy, valid))
        with SummaryWriter(str(directory.parent), filename_suffix=".rolling_velocity049") as writer:
            for label in ("true_sensor_kinematics", "deployable_observation_kinematics", "original_wheel_proxy"):
                for key in ("mae_m_s", "rms_error_m_s", "mean_error_m_s"):
                    writer.add_scalar(f"Diagnosis/rolling_velocity049/{label}/{key}", row[label][key], 441)
        rows.append(row)
    output = dict(schema="rolling049-kinematic-probe-v1", optimizer_updates=0,
        table=str(TABLE.relative_to(ROOT)), table_sha256=sha(TABLE),
        model_spec_sha256=sha(ASSET / "model_spec.json"), results=rows,
        limits="Rolling assumption and model geometry only. Truth measurements are diagnostic targets, never actor inputs. Sensor delays/noise retained; no controller or runtime contract changed.")
    (REPORTS / "rolling_velocity049_probe.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
