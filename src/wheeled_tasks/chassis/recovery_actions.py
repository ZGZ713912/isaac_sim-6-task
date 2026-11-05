"""Encoder-relative recovery targets; no accumulated target or privileged input.

Each side commands common output-shaft rotation and signed auxiliary-minus-hip
angle. The latter is a motor coordinate, not the passive mechanical knee angle.
"""
from copy import deepcopy
import math

import torch


CONTRACT = dict(schema="wheel-leg-encoder-increment-actions-v1",
    policy_order=["left_common", "left_relative", "right_common", "right_relative",
                  "left_wheel", "right_wheel"],
    leg_raw_clip=1., common_rate_rad_s=2., relative_rate_rad_s=2.,
    wheel_semantics="original_absolute_velocity_target",
    reference="last_sample_of_existing_PD_encoder_feedback",
    feedback_clock="actual_fresh_PD_period", target_clock="once_per_policy_interval",
    target_accumulator=False, shared_turn="hip_encoder_chart_with_wrapped_pair_difference",
    boundary="authenticated_relative_soft_limits_bounded_inward_return",
    previous_action="clipped_raw_coordinates_zero_at_reset_and_while_disabled",
    action_rate_cost="mean_squared_raw_coordinate_change_with_original_reference_dt",
    privileged_input=False, all_six_axes_trainable=True)

RANGE_CONTRACT = dict(deepcopy(CONTRACT), schema="wheel-leg-encoder-increment-actions-v2",
    common_rate_rad_s=8., relative_rate_rad_s=4.)


def contract():
    return deepcopy(CONTRACT)


def validate(config):
    if config not in (CONTRACT, RANGE_CONTRACT):
        raise ValueError("Use the complete encoder increment action contract")
    return deepcopy(config)


def targets(actions, measured, *, policy_dt, relative_limits, config=None):
    """Return four leg targets, on the measured hip's shared winding.

    Inside the soft domain, targets stay inside it. Outside it, the projection
    returns inward by at most one declared increment, including for zero input;
    it never inserts an arbitrarily large boundary correction. This bounds
    *target displacement from the encoder reference*, not actual joint speed.
    """
    if actions.shape != measured.shape or actions.shape[-1] != 4:
        raise ValueError("Recovery leg actions and encoders need matching [...,4] shapes")
    if not math.isfinite(policy_dt) or policy_dt <= 0:
        raise ValueError("Recovery actions require a positive policy period")
    lo, hi = relative_limits
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("Recovery actions require authenticated relative bounds")
    settings = CONTRACT if config is None else validate(config)
    raw = actions.clamp(-settings["leg_raw_clip"], settings["leg_raw_clip"])
    q = measured.double().unflatten(-1, (2, 2))
    action = raw.double().unflatten(-1, (2, 2))
    sign = q.new_tensor([1., -1.])
    difference = q[..., 1]-q[..., 0]
    relative = sign*torch.atan2(difference.sin(), difference.cos())
    common = q[..., 0]+.5*sign*relative
    common_step = action[..., 0]*settings["common_rate_rad_s"]*policy_dt
    relative_step = settings["relative_rate_rad_s"]*policy_dt
    wanted = (relative+action[..., 1]*relative_step).clamp(lo, hi)
    delta = (wanted-relative).clamp(-relative_step, relative_step)
    common = common+common_step
    relative = relative+delta
    result = torch.stack((common-.5*sign*relative, common+.5*sign*relative), -1)
    return result.flatten(-2).to(measured.dtype), raw


class EncoderIncrementActions:
    def __init__(self, env, config):
        from .getup_feedback import GetupFeedback
        self.config = validate(config)
        if env.tick != 0 or env.getup_feedback.tick != 0:
            raise ValueError("Install recovery actions before any physics or feedback history")
        gas = env.v5.gas_compensation
        if gas is None or not env.getup_cfg.get("paired_feedback_winding"):
            raise ValueError("Recovery increments require authenticated paired encoder feedback")
        limits = gas.cfg["relative_soft_limits"]
        self.relative_limits = (limits["minimum"]+limits["margin"],
                                limits["maximum"]-limits["margin"])
        self.env = env
        # observe() is called only on fresh PD evaluations, not solver substeps.
        # Construct at the actual PD period before the final environment reset.
        env.getup_feedback = GetupFeedback(env.num_envs, env.device, env.domain.enabled,
            env.getup_cfg, env.domain.rng, dt=env.policy_dt/env.decimation*
            env.control_audit.control_stride)
        self.decode_count = 0
        self.last_reference = self.last_targets = self.last_raw = None
        self.reference_feedback_tick = 0
        bounds = env.v5.action_bounds.clone()
        bounds[list(env.v5.LEGS)] = config["leg_raw_clip"]
        env.v5.action_bounds = bounds
        env.v5.decode = self.decode
        env._measured_action_history = self.initialize_history
        quiet_update = env.quiet_standing.update
        def update_quiet(active, xy, unused_absolute_action_target, velocity):
            return quiet_update(active, xy, self.physical_targets(), velocity)
        env.quiet_standing.update = update_quiet
        env.getup_cfg["cyclic_leg_targets"] = False
        env.startup_report["encoder_increment_actions"] = dict(config,
            feedback_dt=env.getup_feedback.dt, policy_dt=env.policy_dt,
            relative_soft_bounds_rad=list(self.relative_limits),
            reference_age_nominal_at_policy_boundary_s=env.getup_feedback.dt,
            mechanical_knee_is_not_motor_relative_angle=True,
            physical_speed_is_not_limited_by_target_increment=True)

    def initialize_history(self, ids):
        # Zero now means hold the encoder pose, with original wheel target0.
        # This callback is also used at the disabled/enable boundary.
        self.env.actions[ids] = 0.

    def decode(self, actions, unused_current_simulator_q):
        env = self.env
        clipped = actions.clamp(-env.v5.action_bounds, env.v5.action_bounds)
        reference = env.getup_feedback.q[:, list(env.v5.LEGS)].clone()
        legs, _ = targets(clipped[:, list(env.v5.LEGS)], reference,
            policy_dt=env.policy_dt, relative_limits=self.relative_limits, config=self.config)
        wheels = clipped[:, list(env.v5.WHEELS)]*env.v5.settings["wheel_velocity_scale"]
        self.last_reference, self.last_targets, self.last_raw = reference, legs, clipped.clone()
        self.reference_feedback_tick = env.getup_feedback.tick
        self.decode_count += 1
        return legs, wheels, clipped

    def physical_targets(self):
        """Actual shaft-radian / wheel-rad/s targets for the quiet filter."""
        result = torch.empty_like(self.last_raw)
        result[:, list(self.env.v5.LEGS)] = self.last_targets
        result[:, list(self.env.v5.WHEELS)] = (self.last_raw[:, list(self.env.v5.WHEELS)]*
            self.env.v5.settings["wheel_velocity_scale"])
        return result

    def metrics(self):
        if self.last_targets is None:
            return {}
        from .paired_feedback import paired_position_error
        step = paired_position_error(self.last_targets, self.last_reference)
        return {"RecoveryAction/target_encoder_displacement_peak_rad": step.abs().max(),
            "RecoveryAction/raw_leg_clip_fraction":
                (self.last_raw[:, list(self.env.v5.LEGS)].abs() >= self.config["leg_raw_clip"]).float().mean(),
            "RecoveryAction/PD_feedback_period_s": step.new_tensor(self.env.getup_feedback.dt),
            "RecoveryAction/target_accumulator": step.new_tensor(0.)}
