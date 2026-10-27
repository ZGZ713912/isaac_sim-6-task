"""State-based guidance for a learned in-place rise from standard prone."""
import math

import torch


def initialization_motor_target(base_control_positions, common_rotation):
    """A fixed, trainable mechanical bias; no sensed state or action clock."""
    rotation = base_control_positions.new_tensor(common_rotation)
    if (base_control_positions.shape != (6,) or rotation.shape != (2,)
            or not torch.isfinite(base_control_positions).all() or not torch.isfinite(rotation).all()
            or (rotation.abs() > math.pi).any()):
        raise ValueError("Need six finite model positions and two bounded common rotations")
    target = base_control_positions.clone()
    target[:2] += rotation[0]
    target[3:5] += rotation[1]
    return target


def physical_knee_degrees(knee_positions, left_lower_limit):
    zero = math.radians(40.) - left_lower_limit
    return torch.rad2deg(knee_positions * knee_positions.new_tensor([1., -1.]) + zero)


def support_error(wheel_midpoint, whole_com, minimum_below):
    horizontal = (wheel_midpoint[:, :2] - whole_com[:, :2]).norm(dim=-1)
    above = (wheel_midpoint[:, 2] - whole_com[:, 2] + minimum_below).clamp_min(0.)
    return torch.sqrt(horizontal.square() + above.square())


def valid_disabled_start(gravity, height, belly_force, spec):
    lo, hi = spec["enable_height_range_m"]
    return ((gravity[:, 2] < -math.cos(math.radians(spec["enable_max_tilt_deg"])))
            & (height >= lo) & (height <= hi) & (belly_force > 2.))


def route_potential(gravity, knee_deg, support_error_m, height, spec, geometry_valid=None):
    up = ((1. - gravity[:, 2]) * .5).clamp(0., 1.)
    # A band next to the stop has no incentive to drive into/past the stop.
    fold_error = ((knee_deg - spec["fold_target_deg"] - spec["fold_soft_band_deg"]).clamp_min(0.)
                  + (spec["fold_target_deg"] - knee_deg).clamp_min(0.))
    folded = (1. - fold_error / spec["fold_guidance_span_deg"]).clamp_min(0.).mean(-1)
    aligned = 1. / (1. + (support_error_m / spec["support_alignment_width_m"]).square())
    if geometry_valid is not None:
        folded = folded * geometry_valid
        aligned = aligned * geometry_valid
    lift = ((height - spec["belly_height_m"]) / (spec["target_height_m"] - spec["belly_height_m"])).clamp(0., 1.)
    # Once the wheels reach the support region, extension must no longer lose
    # a large held-fold bonus before the body can physically respond upward.
    folding_weight = (1. - lift) * (1. - aligned).pow(spec["fold_capture_release_power"])
    standing_height = torch.exp(-((height - spec["target_height_m"]) / spec["target_height_width_m"]).square())
    # A belly-grounded body already has upright IMU gravity. Use actual lift,
    # rather than attitude alone, to fade the folding guidance during rising.
    potential = 4.*up + folding_weight*folded + 2.*up*aligned + 8.*up*lift + up*standing_height
    return potential, dict(fold_score=folded, support_alignment=aligned, upright_score=up,
                           folding_guidance_weight=folding_weight, lift_progress=lift,
                           support_guidance_weight=1.-lift,
                           standing_height_score=standing_height,
                           normalized_height_error=(height-spec["target_height_m"])
                               / (spec["target_height_m"]-spec["belly_height_m"]))


def stable(gravity, height, omega, velocity, wheel_contact, leg_velocity, spec):
    return ((gravity[:, 2] < -math.cos(math.radians(spec["stable_tilt_deg"])))
        & ((height - spec["target_height_m"]).abs() < spec["stable_height_error_m"])
        & (omega.norm(dim=-1) < spec["stable_body_rate_rad_s"])
        & (velocity[:, :2].norm(dim=-1) < spec["stable_planar_speed_m_s"])
        & wheel_contact.all(-1) & (leg_velocity.abs().amax(-1) < spec["stable_leg_speed_rad_s"]))


def density(gravity, velocity, omega, torque, leg_speed_square, excursion, stable_now, spec,
            route_scores=None, recovered=None):
    result = dict(
        upright=.5 * ((1.-gravity[:, 2])*.5).clamp(0., 1.),
        stable_standing=8.*stable_now.float(),
        in_place_cost=-((excursion-spec["position_cost_free_radius_m"]).clamp_min(0.)/spec["position_cost_scale_m"]).square(),
        travel_speed_cost=-.1*(velocity[:, :2]/.3).square().sum(-1),
        angular_cost=-.02*omega.square().sum(-1),
        leg_speed_cost=-.001*leg_speed_square,
        effort_cost=-.0001*torque.square().mean(-1))
    if route_scores is not None:
        weight = route_scores["folding_guidance_weight"]
        result.update(folding_progress=2.*weight*route_scores["fold_score"],
                      support_alignment=route_scores["support_guidance_weight"]*route_scores["support_alignment"],
                      rise_progress=spec["rise_progress_reward_per_second"]*route_scores["upright_score"]*route_scores["lift_progress"],
                      target_height=spec["target_height_reward_per_second"]*route_scores["upright_score"]*route_scores["standing_height_score"],
                      height_error_cost=-spec["height_error_cost_per_second"]*route_scores["normalized_height_error"].square())
        level_weight = spec["level_cost_min_lift_weight"] + (
            1.-spec["level_cost_min_lift_weight"])*route_scores["lift_progress"]
        # Squared distance to level gravity penalizes small roll/pitch errors
        # explicitly and does not make a fully inverted body look level.
        level_error = (gravity-gravity.new_tensor([0., 0., -1.])).square().sum(-1)
        result["level_error_cost"] = (-spec["level_error_cost_per_second"]*level_weight*level_error
                                      / math.sin(math.radians(spec["level_error_scale_deg"]))**2)
    if recovered is not None:
        result["unrecovered_time_cost"] = -spec["unrecovered_time_cost_per_second"]*(~recovered).float()
    return result
