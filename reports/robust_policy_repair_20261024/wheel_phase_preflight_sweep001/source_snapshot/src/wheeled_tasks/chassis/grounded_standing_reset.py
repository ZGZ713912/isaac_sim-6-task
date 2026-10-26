"""Explicit reset-only placement of a standing robot on both wheel cylinders."""
from __future__ import annotations

import torch

MODULE = 'src/wheeled_tasks/chassis/grounded_standing_reset.py'
VERSION = 'dual-wheel-grounded-reset-v1'
PRELOAD_M = .00005


def rotate_about_axis(value, axis, angle):
    axis = axis[:, None, :]
    c, s = angle[:, None, None].cos(), angle[:, None, None].sin()
    return value * c + torch.linalg.cross(axis.expand_as(value), value, dim=-1) * s + axis * (axis * value).sum(-1, keepdim=True) * (1. - c)


def cylinder_clearance(relative_centers, cylinder_axes, root_z, floor_z, roll_axis, angle):
    center = rotate_about_axis(relative_centers, roll_axis, angle)
    axis_z = rotate_about_axis(cylinder_axes, roll_axis, angle)[..., 2].clamp(-1., 1.)
    extent = .06 * (1. - axis_z.square()).clamp_min(0.).sqrt() + .012500000186264515 * axis_z.abs()
    return center[..., 2] + root_z[:, None] - floor_z[:, None] - extent


def grounded_pose(root, wheel_centers, wheel_quaternions, floor_z):
    """Rigid reset transform only; joint geometry and velocities are untouched."""
    x, y, z, w = root[:, 3:].unbind(-1)
    roll_axis = torch.stack((1. - 2. * (y*y + z*z), 2. * (x*y + w*z), 2. * (x*z - w*y)), -1)
    qx, qy, qz, qw = wheel_quaternions.unbind(-1)
    axes = torch.stack((2. * (qx*qz + qw*qy), 2. * (qy*qz - qw*qx), 1. - 2. * (qx*qx + qy*qy)), -1)
    centers = wheel_centers - root[:, None, :3]
    angle = root.new_zeros(len(root))
    for _ in range(4):
        gap = cylinder_clearance(centers, axes, root[:, 2], floor_z, roll_axis, angle)
        plus = cylinder_clearance(centers, axes, root[:, 2], floor_z, roll_axis, angle + .0001)
        minus = cylinder_clearance(centers, axes, root[:, 2], floor_z, roll_axis, angle - .0001)
        derivative = ((plus[:, 0] - plus[:, 1]) - (minus[:, 0] - minus[:, 1])) / .0002
        if (derivative.abs() < .1).any():
            raise ValueError('Standing wheels need a finite lateral support span')
        angle = (angle - (gap[:, 0] - gap[:, 1]) / derivative).clamp(-.02, .02)
    gap = cylinder_clearance(centers, axes, root[:, 2], floor_z, roll_axis, angle)
    if (gap.diff(dim=-1).abs() > .000002).any():
        raise ValueError('Cannot ground both wheels with the bounded reset roll')
    result = root.clone()
    result[:, 2] -= gap.mean(-1) + PRELOAD_M
    s, c = (angle / 2.).sin(), (angle / 2.).cos()
    # q_new = q_root * q_local_x(angle), scalar-last convention.
    result[:, 3:] = torch.stack((w*s + x*c, y*c + z*s, z*c - y*s, w*c - x*s), -1)
    if not torch.isfinite(result).all() or ((result[:, 2] - root[:, 2]).abs() > .025).any():
        raise ValueError('Grounding may correct only the small standing spawn clearance')
    return result, angle, gap


def install(env):
    """A labelled training/evaluation initialization; no writes inside step."""
    if hasattr(env, 'grounded_standing_reset'):
        raise ValueError('Standing placement may be installed once')
    from .root_kinematics import update_robot_data
    from .env import ChassisEnv
    native_reset = env.reset
    state = dict(version=VERSION, placement_calls=0, environments=0,
                 maximum_translation_m=0., maximum_roll_rad=0., preload_m=PRELOAD_M)

    def place(ids):
        selected = ids[(~env.recovery_lane[ids]) & (env.pending_command[ids, :2].abs() < .01).all(-1)]
        if not len(selected):
            return
        root = env.robot.data.root_link_pose_w.torch[selected].clone()
        wheels = env.robot.data.body_link_pose_w.torch[selected][:, env.wheel_ids]
        pose, roll, _ = grounded_pose(root, env.wheel_centers()[selected], wheels[:, :, 3:], env.origins[selected, 2])
        env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=selected)
        update_robot_data(env, env.dt)
        env.previous_potential[selected] = env.potential()[selected]
        # Retain already sampled gains, biases, delays and RNG. Only initialize
        # sensor history to the transformed reset gravity, as native reset does.
        frame = ChassisEnv.get_observations(env)['policy']
        sensors = frame[selected, 4:22].clone()
        if env.standing_observer is not None:
            sensors[:, 10:12] = env.robot.data.joint_pos.torch[selected][:, env.ids][:, [2, 5]]
            env.standing_observer.reset(selected)
            if env.standing_sensor_winding is not None:
                env.standing_sensor_winding.reset(selected)
        if env.domain.imu_mount is not None:
            sensors = env.domain.imu_mount.apply(sensors, env.tick * env.policy_dt, selected) + env.domain.bias[selected]
            env.domain.last_sensor_tick = -1
        env.domain.sensor_history[:, selected] = sensors[None]
        env.domain.observed[selected] = sensors
        state['placement_calls'] += 1
        state['environments'] += len(selected)
        state['maximum_translation_m'] = max(state['maximum_translation_m'], float((pose[:, 2] - root[:, 2]).abs().max()))
        state['maximum_roll_rad'] = max(state['maximum_roll_rad'], float(roll.abs().max()))

    def reset(ids):
        result = native_reset(ids)
        place(ids)
        return result

    env.reset = reset
    env.grounded_standing_reset = state
    # Initial construction has already completed its first episode reset.
    place(torch.arange(env.num_envs, device=env.device))
    return state
