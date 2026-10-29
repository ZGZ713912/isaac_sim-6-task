"""Offline equilibria, constrained virtual work and LQR seeds for v2.0.1."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares
from scipy.signal import cont2discrete
from scipy.spatial.transform import Rotation

from .full_robot_asset import ACTIVE, SPRINGS, closure_residual, fk, point, solve_pose


def geometry(spec, q):
    frames = fk(spec, q)
    wheels = np.array([frames[name][:3, 3] for name in ("L_link3", "R_link3")])
    # Put both wheel axes over one flat support plane, allowing the infinitesimal
    # body roll required when differentiating left/right legs independently.
    d = wheels[0] - wheels[1]
    rotation = Rotation.from_euler("x", np.arctan2(-d[2], d[1])).as_matrix()
    wheel_z = (wheels @ rotation.T)[:, 2].mean()
    height = .06 - wheel_z
    masses = np.array([body["mass"] for body in spec["bodies"]])
    coms = np.array([point(frames[b["name"]], b["com"]) for b in spec["bodies"]]) @ rotation.T
    com = (masses[:, None] * coms).sum(0) / masses.sum()
    potential = float((masses * 9.81 * (coms[:, 2] + height)).sum())
    return frames, height, com, potential


def equilibrium_pose(spec, length, initial=None):
    if not .22 <= length <= .355:
        raise ValueError("Length outside the initial reviewed research interval")
    beta = np.arccos((.21**2 + .25**2 - length**2) / (2*.21*.25))
    knee = beta - (np.pi - 2.3573)
    seed = initial or spec["nominal_joint_pos"]
    fixed = dict(L_joint2=knee, R_jonit2=-knee, L_joint3=0., R_joint3=0., M_joint1=0., M_joint2=0.)

    def pose(values):
        hip, left_knee, right_knee = values
        return solve_pose(spec, dict(fixed, L_joint1=hip, R_joint1=-hip,
                                     L_joint2=left_knee, R_jonit2=right_knee), seed)

    def balance(values):
        frames, _, com, _ = geometry(spec, pose(values))
        axle_x = (frames["L_link3"][0, 3] + frames["R_link3"][0, 3]) / 2
        lengths = [np.linalg.norm((frames[s+"_link3"][:3, 3]
                   - frames[s+"_link1"][:3, 3])[[0, 2]]) for s in ("L", "R")]
        return np.array([com[0] - axle_x, lengths[0] - length, lengths[1] - length])

    result = least_squares(balance, [-.42, knee, -knee], xtol=1e-11, ftol=1e-11, gtol=1e-11,
                           bounds=([-1.4, -.08, -1.08], [.6, 1.08, .08]), diff_step=1e-4)
    if np.max(np.abs(balance(result.x))) > 1e-7:
        raise ValueError("No supported full-robot equilibrium at requested length")
    return pose(result.x)


def virtual_work(spec, q):
    names = [j["name"] for j in spec["joints"]]
    active = [names.index(n) for n in ACTIVE]
    passive = [i for i, n in enumerate(names) if n not in ACTIVE + ["M_joint1", "M_joint2"]]
    columns, gradient = [], []
    for name in names:
        plus, minus = dict(q), dict(q)
        plus[name] += 1e-6
        minus[name] -= 1e-6
        columns.append((closure_residual(spec, plus) - closure_residual(spec, minus)) / 2e-6)
        gradient.append((geometry(spec, plus)[3] - geometry(spec, minus)[3]) / 2e-6)
    jacobian = np.column_stack(columns)
    mapping = np.zeros((len(names), 6))
    mapping[active] = np.eye(6)
    mapping[passive] = np.linalg.lstsq(jacobian[:, passive], -jacobian[:, active], rcond=1e-8)[0]
    residual = float(np.max(np.abs(jacobian @ mapping)))
    if residual > 2e-7:
        raise ValueError(f"Virtual-work tangent does not close: {residual}")
    return np.array(gradient) @ mapping, mapping[[names.index(n) for n in SPRINGS]], residual


def linear_seed(spec, q, q_scale=1., r_scale=1.):
    """Engineering rolling-pendulum seed, not an identified full PhysX model."""
    frames, height, com, _ = geometry(spec, q)
    total = sum(b["mass"] for b in spec["bodies"])
    axle = (frames["L_link3"][:3, 3] + frames["R_link3"][:3, 3]) / 2
    h = float(com[2] - axle[2])
    inertia = np.zeros((3, 3))
    for body in spec["bodies"]:
        r = frames[body["name"]][:3, :3]
        delta = point(frames[body["name"]], body["com"]) - com
        inertia += r @ body["inertia"] @ r.T + body["mass"]*(delta @ delta*np.eye(3) - np.outer(delta, delta))
    wheel_j = sum(float(np.linalg.eigvalsh(b["inertia"])[-1]) for b in spec["bodies"] if b["name"] in ("L_link3", "R_link3"))
    mass_matrix = np.array([[total + wheel_j/.06**2, total*h], [total*h, inertia[1, 1] + total*h*h]])
    inverse = np.linalg.inv(mass_matrix)
    a = np.zeros((4, 4))
    a[0, 1], a[3, 2] = 1., 1.
    a[[2, 1], 0] = inverse[:, 1] * total*9.81*h
    b = np.zeros((4, 1))
    b[[2, 1], 0] = inverse @ np.array([2/.06, -2.])
    ad, bd, _, _, _ = cont2discrete((a, b, np.eye(4), np.zeros((4, 1))), .001)
    q_weight = np.diag([100., 3., 8., 1.])*q_scale
    r_weight = np.array([[r_scale]])
    p = solve_discrete_are(ad, bd, q_weight, r_weight)
    gain = -np.linalg.solve(r_weight + bd.T@p@bd, bd.T@p@ad)[0]
    radius = float(np.max(np.abs(np.linalg.eigvals(ad + bd @ gain[None]))))
    track = abs(frames["L_link3"][1, 3] - frames["R_link3"][1, 3])
    yaw_b = track / (.06*inertia[2, 2])
    ay, by, _, _, _ = cont2discrete((np.array([[0., 1.], [0., 0.]]), np.array([[0.], [yaw_b]]), np.eye(2), np.zeros((2, 1))), .001)
    py = solve_discrete_are(ay, by, np.diag([5., 1.]), [[1.]])
    yaw = -np.linalg.solve(np.array([[1.]]) + by.T@py@by, by.T@py@ay)[0]
    return dict(balance_gain=gain.tolist(), yaw_gain=yaw.tolist(),
        discrete_spectral_radius=radius, nominal_height_m=height, com_height_above_axle_m=h,
        model_scope="engineering_rolling_pendulum_seed_not_identified_full_model", dt_s=.001)


def posture_table(spec, lengths):
    rows = []
    q = None
    for length in lengths:
        q = equilibrium_pose(spec, float(length), q)
        gravity, spring, error = virtual_work(spec, q)
        seed = linear_seed(spec, q)
        frames = fk(spec, q)
        rows.append(dict(length_m=float(length), active_position=[q[n] for n in ACTIVE],
            all_joint_positions=q, gravity_torque_nm=gravity.tolist(),
            spring_extension_jacobian=spring.tolist(), tangent_residual_m=error,
            compression_m=[spec["spring_binding"][n]["compression_at_q_zero_m"] - q[n] for n in SPRINGS],
            axle_x_m=float((frames["L_link3"][0, 3]+frames["R_link3"][0, 3])/2), **seed))
    return dict(schema="full-robot-v201-traditional-postures/v1", active_order=ACTIVE,
        positions_are_cad_joint_coordinates=True, hardware_mapping_verified=False,
        lengths_m=[float(v) for v in lengths], knots=rows)
