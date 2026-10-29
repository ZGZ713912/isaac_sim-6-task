"""Standing costs measured at every existing 1 ms PhysX boundary.

This observer never changes an action, contact, simulator state or PD gain.
There is no speed deadband, upright gate or startup grace period.
"""
from __future__ import annotations

import torch

from .standing_contact_monitor import StandingContactMonitor
from .standing_contact_reward import support_density


def motion_density(velocity, torque_delta):
    if velocity.ndim != 3 or velocity.shape[-1] != 4 or torque_delta.shape != velocity.shape:
        raise ValueError("Expected [physical substep, environment, four leg motors]")
    if not torch.isfinite(velocity).all() or not torch.isfinite(torque_delta).all():
        raise ValueError("Nonfinite physical standing sample")
    # Smooth costs at zero; small oscillations remain visible, without allowing
    # a single contact impulse to dominate the entire advantage population.
    speed = torch.log1p((velocity / .2).square()).mean((0, 2))
    slew = torch.log1p((torque_delta / 2.).square()).mean((0, 2))
    return speed, slew


def install(env, *, speed_weight=2., contact_weight=8., torque_weight=.05, include_motion=False):
    if hasattr(env, "physical_standing_reward"):
        raise ValueError("Physical standing objective already installed")
    if not all(0. <= value <= 16. for value in (speed_weight, contact_weight, torque_weight)):
        raise ValueError("Invalid standing cost weights")
    monitor = StandingContactMonitor(env)
    native_step, native_gap = env.step, env.closure_gap
    native_torque = env.actuator_response.apply_torque
    velocities, torques = [], []
    inside = False
    actual_torque = None
    previous_torque = torch.zeros(env.num_envs, 4, device=env.device)

    def apply_torque(torque):
        nonlocal actual_torque
        actual_torque = native_torque(torque)
        # Native code clamps this same tensor before applying it to PhysX.
        return actual_torque

    def gap():
        result = native_gap()
        if inside:
            velocities.append(env.robot.data.joint_vel.torch[:, env.ids][:, list(env.v5.LEGS)].clone())
            torques.append(actual_torque[:, list(env.v5.LEGS)].clone())
        return result

    def step(actions):
        nonlocal inside
        requested = (~env.recovery_lane) & ~env.passive
        if not include_motion:
            requested &= (env.pending_command[:, :2].abs() < .01).all(-1)
        # A reset has zero actuator state; do not charge a cross-episode delta.
        previous_torque[env.episode_length_buf == 0] = 0.
        velocities.clear()
        torques.clear()
        inside = True
        try:
            obs, reward, done, extras = native_step(actions)
        finally:
            inside = False
        if len(velocities) != 5 or env.dt != .001 or env.policy_dt != .005:
            raise RuntimeError("Standing objective requires all five native 1 ms samples")
        dq, tau = torch.stack(velocities), torch.stack(torques)
        delta = tau - torch.cat((previous_torque[None], tau[:-1]))
        previous_torque.copy_(tau[-1])
        speed, slew = motion_density(dq, delta)
        support = support_density(monitor.last["force"], monitor.last["clearance"])
        eligible = requested & extras["policy_learning_mask"]
        paid_speed = env.policy_dt * speed_weight * speed * eligible
        paid_contact = env.policy_dt * contact_weight * support * eligible
        paid_slew = env.policy_dt * torque_weight * slew * eligible
        penalty = paid_speed + paid_contact + paid_slew
        sample = monitor.last
        metrics = {
            "dual_support_physical_fraction": (sample["force"].norm(dim=-1) > 2.).all(-1).float().mean(),
            "missing_ms_per_policy_step": sample["missing_ms"].float().mean(),
            "leg_speed_rms_rad_s": dq.square().mean().sqrt(),
            "leg_speed_peak_rad_s": dq.abs().max(),
            "leg_torque_delta_rms_nm_per_ms": delta.square().mean().sqrt(),
            "wheel_clearance_peak_m": sample["clearance"].max(),
            "speed_cost_density_per_second": (speed * speed_weight * eligible).mean(),
            "support_cost_density_per_second": (support * contact_weight * eligible).mean(),
            "torque_cost_density_per_second": (slew * torque_weight * eligible).mean(),
            "paid_speed_cost_per_policy_step": paid_speed.mean(),
            "paid_support_cost_per_policy_step": paid_contact.mean(),
            "native_reward_per_policy_step": reward.mean(),
        }
        extras["log"].update({"ScratchStanding/" + key: value for key, value in metrics.items()})
        env.physical_standing_last = dict(dq=dq, torque=tau, metrics=metrics,
            paid_speed=paid_speed, paid_contact=paid_contact, paid_slew=paid_slew)
        return obs, reward - penalty, done, extras

    env.step, env.closure_gap = step, gap
    env.actuator_response.apply_torque = apply_torque
    env.physical_standing_reward = dict(version="physical-standing-cost-v1", physical_dt_s=.001,
        speed_scale_rad_s=.2, torque_delta_scale_nm=2., speed_weight=speed_weight,
        contact_weight=contact_weight, torque_weight=torque_weight, deadband_rad_s=0., startup_grace_s=0.)
    if include_motion:
        env.physical_standing_reward["scope"] = "all upright tracking requests; original standing weights and samples"
    return env.physical_standing_reward
