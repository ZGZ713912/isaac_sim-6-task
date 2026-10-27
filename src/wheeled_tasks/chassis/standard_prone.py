"""State-based guidance for a learned in-place rise from standard prone."""
import math

import torch


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


def route_potential(gravity, knee_deg, support_error_m, height, spec):
    up = ((1. - gravity[:, 2]) * .5).clamp(0., 1.)
    # A band next to the stop has no incentive to drive into/past the stop.
    fold_error = ((knee_deg - spec["fold_target_deg"] - spec["fold_soft_band_deg"]).clamp_min(0.)
                  + (spec["fold_target_deg"] - knee_deg).clamp_min(0.))
    folded = (1. / (1. + (fold_error / spec["fold_width_deg"]).square())).mean(-1)
    aligned = 1. / (1. + (support_error_m / spec["support_alignment_width_m"]).square())
    lift = ((height - spec["belly_height_m"]) / (spec["target_height_m"] - spec["belly_height_m"])).clamp(0., 1.)
    folding_weight = 1. - lift
    standing_height = torch.exp(-((height - spec["target_height_m"]) / .06).square())
    # A belly-grounded body already has upright IMU gravity. Use actual lift,
    # rather than attitude alone, to fade the folding guidance during rising.
    potential = 4.*up + folding_weight*folded*(1.+aligned) + 2.*up*aligned + 8.*up*lift + up*standing_height
    return potential, dict(fold_score=folded, support_alignment=aligned, upright_score=up,
                           folding_guidance_weight=folding_weight, lift_progress=lift)


def stable(gravity, height, omega, velocity, wheel_contact, leg_velocity, spec):
    return ((gravity[:, 2] < -math.cos(math.radians(spec["stable_tilt_deg"])))
        & ((height - spec["target_height_m"]).abs() < spec["stable_height_error_m"])
        & (omega.norm(dim=-1) < spec["stable_body_rate_rad_s"])
        & (velocity[:, :2].norm(dim=-1) < spec["stable_planar_speed_m_s"])
        & wheel_contact.all(-1) & (leg_velocity.abs().amax(-1) < spec["stable_leg_speed_rad_s"]))


def density(gravity, velocity, omega, torque, leg_speed_square, excursion, stable_now, spec):
    return dict(
        upright=.5 * ((1.-gravity[:, 2])*.5).clamp(0., 1.),
        stable_standing=8.*stable_now.float(),
        in_place_cost=-(excursion/spec["position_cost_scale_m"]).square(),
        travel_speed_cost=-.1*(velocity[:, :2]/.3).square().sum(-1),
        angular_cost=-.02*omega.square().sum(-1),
        leg_speed_cost=-.001*leg_speed_square,
        effort_cost=-.0001*torque.square().mean(-1))
