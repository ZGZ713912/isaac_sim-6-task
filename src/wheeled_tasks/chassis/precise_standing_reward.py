"""Fine standing objective, with native mechanical termination preserved.

The recovery reward was deliberately broad over height, tilt and velocity.
This opt-in foundation profile replaces it only on zero-request standing
lanes. All physical-rate costs are still paid from the original 1 ms samples.
"""
import math
import torch


def scores(height, gravity, velocity, omega, target_height):
    if gravity.shape != velocity.shape or omega.shape != gravity.shape or gravity.shape != (height.numel(), 3):
        raise ValueError("Need simultaneous actual height, gravity, velocity and gyro")
    if target_height.shape != height.shape or any(not torch.isfinite(value).all() for value in
            (height, gravity, velocity, omega, target_height)):
        raise ValueError("Invalid precise standing state")
    upright = ((1. - gravity[:, 2]) * .5).clamp(0., 1.)
    attitude = upright * torch.exp(-gravity[:, :2].square().sum(-1) / math.sin(math.radians(5.))**2)
    height_score = torch.exp(-((height - target_height) / .015).square())
    stationary = torch.exp(-velocity[:, :2].square().sum(-1) / .03**2 - (omega[:, 2] / .2).square())
    angular_cost = .1 * torch.log1p((omega[:, :2] / .5).square()).sum(-1)
    # A quiet tilted pose must not earn a separate velocity reward. All three
    # conditions contribute jointly, with a smooth dense signal near standing.
    return 12. * attitude * height_score * stationary - angular_cost


def install(env):
    if not hasattr(env, "physical_standing_reward") or hasattr(env, "precise_standing_reward"):
        raise ValueError("Attach once after the physical standing observer")
    # Native diagnostics are copied before an autoreset. Reading env.state()
    # after step would incorrectly reward the next episode's reset posture.
    env.cfg["record_diagnostics"] = True
    native_step = env.step

    def step(actions):
        observation, reward, done, extras = native_step(actions)
        state = extras["diagnostics"]
        density = scores(state["height"], state["gravity"], state["velocity"], state["omega"], state["commands"][:, 2])
        eligible = extras["policy_learning_mask"] & (state["requested_command"][:, :2].abs() < .01).all(-1)
        physical = env.physical_standing_last
        total_cost = physical["paid_speed"] + physical["paid_contact"] + physical["paid_slew"]
        exact = density * env.policy_dt - total_cost - 3. * state["terminated"]
        result = torch.where(eligible, exact, reward)
        extras["log"].update({
            "PrecisionStanding/height_mae_m": (state["height"] - state["commands"][:, 2]).abs().mean(),
            "PrecisionStanding/tilt_mean_deg": torch.acos((-state["gravity"][:, 2]).clamp(-1., 1.)).mean() * 180. / math.pi,
            "PrecisionStanding/planar_speed_rms_m_s": state["velocity"][:, :2].square().sum(-1).mean().sqrt(),
            "PrecisionStanding/score_density_per_second": density.mean(),
            "PrecisionStanding/reward_per_policy_step": result.mean()})
        return observation, result, done, extras

    env.step = step
    env.precise_standing_reward = dict(version="precise-standing-reward-v2-coupled", target_height_width_m=.015,
        tilt_width_deg=5., stationary_velocity_width_m_s=.03, yaw_width_rad_s=.2,
        density_maximum_per_second=12., native_termination_cost=3.,
        composition="12*attitude_score*height_score*stationary_score-angular_cost",
        physical_costs=env.physical_standing_reward, startup_grace_s=0.,
        native_recovery_potential_retained_on_standing=False)
    return env.precise_standing_reward
