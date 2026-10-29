"""State-based guidance for a learned in-place rise from standard prone."""
import math

import torch


def physical_knee_degrees(knee_positions, left_lower_limit):
    zero = math.radians(40.) - left_lower_limit
    return torch.rad2deg(knee_positions * knee_positions.new_tensor([1., -1.]) + zero)


def route_potential(gravity, knee_deg, support_error_m, height, spec):
    up = ((1. - gravity[:, 2]) * .5).clamp(0., 1.)
    # A band next to the stop has no incentive to drive into/past the stop.
    fold_error = ((knee_deg - spec["fold_target_deg"] - spec["fold_soft_band_deg"]).clamp_min(0.)
                  + (spec["fold_target_deg"] - knee_deg).clamp_min(0.))
    folded = (1. / (1. + (fold_error / spec["fold_width_deg"]).square())).mean(-1)
    aligned = 1. / (1. + (support_error_m / spec["support_alignment_width_m"]).square())
    lying = ((gravity[:, 2] + math.cos(math.radians(15.))) / math.cos(math.radians(15.))).clamp(0., 1.)
    standing_height = torch.exp(-((height - spec["target_height_m"]) / .06).square())
    # Folding guidance fades with actual upright attitude, so it cannot force
    # the knees to remain tucked after the robot needs to extend and balance.
    potential = 4.*up + lying*folded*(1.+aligned) + 2.*up*aligned + up*standing_height
    return potential, dict(fold_score=folded, support_alignment=aligned, upright_score=up,
                           folding_guidance_weight=lying)


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
