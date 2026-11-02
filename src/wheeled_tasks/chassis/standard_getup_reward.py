"""Continuous state guidance for standard-prone exploration, with no trajectory."""
from copy import deepcopy
import math

import torch


def reward_contract():
    return dict(schema="wheel-leg-standard-getup-dense-v1", target_height_m=.305,
        height_weight=2., height_width_m=.06, height_error_weight=2., height_error_scale_m=.08,
        upright_error_weight=2., support_weight=1., support_scale_m=.12, body_rate_weight=.02,
        units="reward_per_second", activation="original_standard_prone_before_completion")


def ordered_reward_contract():
    """Geometry-based preparation; these scales are engineering candidates."""
    return dict(schema="wheel-leg-standard-getup-ordered-v2", target_height_m=.305,
        height_weight=2., height_width_m=.06, height_error_weight=2., height_error_scale_m=.08,
        upright_error_weight=6., support_weight=1., support_scale_m=.12, body_rate_weight=.1,
        fold_error_weight=2., fold_target_max_deg=46., fold_error_span_deg=57.,
        capture_inner_m=.04, capture_outer_m=.12, capture_exponent=2.,
        height_gate_full_tilt_deg=8., height_gate_zero_tilt_deg=25.,
        preparation_height_ceiling_m=.22, premature_height_weight=2., premature_height_scale_m=.08,
        potential_fold_weight=2., potential_capture_weight=3., potential_lift_weight=3.,
        belly_height_m=.173, units="reward_per_second",
        activation="original_standard_prone_before_completion")


def contact_reward_contract():
    """Require wheel load and arrested body rotation before rewarding lift."""
    return dict(ordered_reward_contract(), schema="wheel-leg-standard-getup-contact-v3",
        wheel_load_full_fraction=.15, wheel_clearance_full_m=.001,
        wheel_clearance_zero_m=.003, height_gate_full_rate_rad_s=.3,
        height_gate_zero_rate_rad_s=1.5, capture_rate_scale_rad_s=.6,
        capture_stability_weight=2., potential_capture_stability_weight=1.)


def contact_guided(config):
    return config.get("schema") in ("wheel-leg-standard-getup-contact-v3",
                                    "wheel-leg-standard-getup-standing-v4")


def standing_reward_contract():
    """Finish the rise as quiet standing, without paying to hold preparation."""
    return dict(contact_reward_contract(), schema="wheel-leg-standard-getup-standing-v4",
        support_weight=0., capture_stability_weight=0.,
        preparation_full_knee_deg=50., preparation_zero_knee_deg=65.,
        chassis_contact_weight=12., quiet_standing_weight=4.,
        quiet_nonwheel_load_scale=.02,
        fold_upright_weight=12., fold_body_rate_weight=.1,
        fold_wheel_clearance_weight=2., fold_wheel_clearance_scale_m=.02,
        fold_wheel_light_load_fraction=.1, fold_wheel_load_weight=2.,
        fold_wheel_load_scale_fraction=.15)


def standing_guided(config):
    return config.get("schema") == "wheel-leg-standard-getup-standing-v4"


def ordered(config):
    return config.get("schema") == "wheel-leg-standard-getup-ordered-v2" or contact_guided(config)


def validate(config):
    if config not in (reward_contract(), ordered_reward_contract(), contact_reward_contract(),
                      standing_reward_contract()):
        raise ValueError("Standard getup needs its complete versioned dense reward contract")
    return deepcopy(config)


def geometry_inputs(route, root):
    """Require both wheels near the COM in the horizontal forward direction.

    A midpoint alone can hide one wheel ahead and the other behind the COM.
    This is privileged reward geometry; it never changes actor observations.
    """
    x, y, z, w = root[:, 3:].unbind(-1)
    forward = torch.stack((1.-2.*(y.square()+z.square()), 2.*(x*y+w*z)), -1)
    forward = forward/forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta = route["encoder_wheel_world_m"][:, :, :2]-route["com_world_m"][:, None, :2]
    return dict(recovery_knee_deg=route["knee_deg"],
        recovery_fore_aft_error_m=(delta*forward[:, None]).sum(-1).abs().amax(-1),
        recovery_geometry_valid=route["encoder_valid"].all(-1))


def _smoothstep(value):
    value = value.clamp(0., 1.)
    return value.square()*(3.-2.*value)


def guidance(state, config):
    """Continuous geometry gates, without a clock, latch or motor sequence."""
    c = config
    knees = state["recovery_knee_deg"]
    error = torch.maximum(state["recovery_support_error_m"], state["recovery_fore_aft_error_m"])
    valid = state["recovery_geometry_valid"] & torch.isfinite(knees).all(-1) & torch.isfinite(error)
    knees = torch.where(valid[:, None], knees, torch.full_like(knees, 103.))
    error = torch.where(valid, error, torch.full_like(error, c["capture_outer_m"]))
    fold_error = (knees-c["fold_target_max_deg"]).clamp_min(0.)/c["fold_error_span_deg"]
    folded = (1.-fold_error.amax(-1)).clamp(0., 1.)*valid
    support = _smoothstep((c["capture_outer_m"]-error)/(c["capture_outer_m"]-c["capture_inner_m"]))
    full = math.cos(math.radians(c["height_gate_full_tilt_deg"]))
    zero = math.cos(math.radians(c["height_gate_zero_tilt_deg"]))
    attitude = _smoothstep((-state["gravity"][:, 2]-zero)/(full-zero))
    capture = support*attitude*valid
    result = dict(fold_score=folded, fold_error_square=fold_error.square().mean(-1)*valid,
        support_capture=capture, height_gate=capture.pow(c["capture_exponent"]),
        support_error_m=error, geometry_valid=valid)
    if contact_guided(c):
        load = state["recovery_wheel_load_fraction"]
        clearance = state["recovery_wheel_clearance_m"]
        bearing_valid = torch.isfinite(load).all(-1) & torch.isfinite(clearance).all(-1)
        bearing_valid &= torch.isfinite(state["omega"]).all(-1)
        load = torch.where(bearing_valid[:, None], load, 0.)
        clearance = torch.where(bearing_valid[:, None], clearance, c["wheel_clearance_zero_m"])
        rate = torch.where(bearing_valid, state["omega"][:, :2].norm(dim=-1), 0.)
        # Vertical load on BOTH wheels, together with collider clearance.
        # This is reward guidance, not a change to contact acceptance rules.
        bearing = _smoothstep(load/c["wheel_load_full_fraction"]).amin(-1)
        bearing *= _smoothstep((c["wheel_clearance_zero_m"]-clearance)/(
            c["wheel_clearance_zero_m"]-c["wheel_clearance_full_m"])).amin(-1)
        bearing *= bearing_valid
        rate_ready = _smoothstep((c["height_gate_zero_rate_rad_s"]-rate)/(
            c["height_gate_zero_rate_rad_s"]-c["height_gate_full_rate_rad_s"]))
        rate_score = 1./(1.+(rate/c["capture_rate_scale_rad_s"]).square())
        loaded_capture = capture*bearing
        result.update(geometric_capture=capture, support_capture=loaded_capture,
            dual_wheel_bearing=bearing, body_horizontal_rate_rad_s=rate,
            rate_readiness=rate_ready, capture_stability=loaded_capture*rate_score,
            height_gate=(loaded_capture*rate_ready).pow(c["capture_exponent"]),
            fold_release=loaded_capture, bearing_valid=bearing_valid)
        if standing_guided(c):
            tucked = _smoothstep((c["preparation_zero_knee_deg"]-knees.amax(-1))/(
                c["preparation_zero_knee_deg"]-c["preparation_full_knee_deg"]))
            # Loaded capture keeps the cost active while the legs extend. No
            # timer/latch, new actor feature or motor trajectory is introduced.
            preparation = torch.maximum(tucked*valid, loaded_capture)
            chassis = state["recovery_chassis_load_fraction"]
            nonwheel = state["recovery_nonwheel_load_fraction"]
            wheel_peak = state["recovery_peak_wheel_load_fraction"]
            force_valid = torch.isfinite(chassis) & torch.isfinite(nonwheel) & torch.isfinite(wheel_peak)
            chassis = torch.where(force_valid, chassis.clamp_min(0.), 1.)
            nonwheel = torch.where(force_valid, nonwheel.clamp_min(0.), 1.)
            wheel_peak = torch.where(force_valid, wheel_peak.clamp_min(0.), 1.)
            body_free = torch.exp(-(nonwheel/c["quiet_nonwheel_load_scale"]).square())*force_valid
            result.update(preparation_score=preparation, chassis_load_fraction=chassis,
                nonwheel_load_fraction=nonwheel, quiet_body_free_score=body_free,
                fold_phase=1.-loaded_capture, peak_wheel_load_fraction=wheel_peak,
                fold_wheel_clearance_error=torch.log1p(((clearance-c["wheel_clearance_zero_m"])
                    .clamp_min(0.)/c["fold_wheel_clearance_scale_m"]).square()).mean(-1),
                fold_wheel_load_error=torch.log1p(((wheel_peak-c["fold_wheel_light_load_fraction"])
                    .clamp_min(0.)/c["fold_wheel_load_scale_fraction"]).square()))
    return result


def ordered_potential(state, config, *, scores=None):
    """Replace the old ungated height potential as well as the dense reward."""
    c = config
    g = guidance(state, c) if scores is None else scores
    up = (-state["gravity"][:, 2]).clamp(0., 1.)
    lift = ((state["height"]-c["belly_height_m"])/(c["target_height_m"]-c["belly_height_m"])).clamp(0., 1.)
    capture = g.get("geometric_capture", g["support_capture"])
    value = up*(c["potential_fold_weight"]*g["fold_score"]*(1.-capture)
        + c["potential_capture_weight"]*capture + c["potential_lift_weight"]*g["height_gate"]*lift)
    if contact_guided(c):
        value += up*c["potential_capture_stability_weight"]*g["capture_stability"]
    return value


def dense_terms(state, config, *, scores=None):
    """Height and posture gradients exist before the first successful getup.

    Like wheelbipe's height and orientation terms, these are state rewards on
    every active step. A flipped body cannot earn positive lift/support credit.
    Support geometry is reward-only privileged state, never an actor feature.
    """
    c = config
    height_error = state["height"]-c["target_height_m"]
    gravity = state["gravity"]
    up = (-gravity[:, 2]).clamp(0., 1.)
    level = (gravity-gravity.new_tensor([0., 0., -1.])).square().sum(-1)
    aligned = 1./(1.+(state["recovery_support_error_m"]/c["support_scale_m"]).square())
    terms = dict(recovery_height=c["height_weight"]*up*torch.exp(-(height_error/c["height_width_m"]).square()),
        recovery_height_error=-c["height_error_weight"]*torch.log1p((height_error/c["height_error_scale_m"]).square()),
        recovery_upright=-c["upright_error_weight"]*level,
        recovery_support=c["support_weight"]*up*aligned,
        recovery_body_rate=-c["body_rate_weight"]*state["omega"].square().sum(-1))
    if ordered(c):
        g = guidance(state, c) if scores is None else scores
        # Defer BOTH height terms. Otherwise the low preparation posture still
        # pays an immediate height-error cost and is encouraged to push upward.
        # Center the height-error density at the belly reference. Activating
        # support must not introduce a negative cost cliff before extension.
        reference = (c["belly_height_m"]-c["target_height_m"])/c["height_error_scale_m"]
        terms["recovery_height_error"] += c["height_error_weight"]*math.log1p(reference**2)
        terms["recovery_height"] *= g["height_gate"]
        terms["recovery_height_error"] *= g["height_gate"]
        safe_aligned = 1./(1.+(g["support_error_m"]/c["support_scale_m"]).square())
        terms["recovery_support"] = c["support_weight"]*up*safe_aligned*torch.maximum(
            g["fold_score"], g["support_capture"])
        terms.update(
            recovery_fold_error=-c["fold_error_weight"]*(1.-g.get("fold_release", g["height_gate"]))*g["fold_error_square"],
            recovery_premature_height=-c["premature_height_weight"]*(1.-g["height_gate"])
                *((state["height"]-c["preparation_height_ceiling_m"]).clamp_min(0.)
                    /c["premature_height_scale_m"]).square())
        if contact_guided(c):
            terms["recovery_capture_stability"] = c["capture_stability_weight"]*up*g["capture_stability"]
        if standing_guided(c):
            from .prone_motion import quiet_standing_quality
            velocity = state["velocity"]
            quality = quiet_standing_quality(state["gravity"], state["height"],
                velocity, state["omega"], c["target_height_m"])
            terms.update(
                recovery_chassis_contact=-c["chassis_contact_weight"]*g["preparation_score"]
                    *g["chassis_load_fraction"],
                recovery_quiet_standing=c["quiet_standing_weight"]*g["preparation_score"]
                    *g["dual_wheel_bearing"]*g["quiet_body_free_score"]*quality,
                recovery_motion_error=-g["preparation_score"]*(
                    torch.log1p((velocity[:, 0]/.5).square())
                    +torch.log1p((state["omega"][:, 2]/.8).square())
                    +.5*torch.log1p((velocity[:, 1]/.2).square())+.1*velocity[:, 2].square()),
                recovery_fold_upright=-c["fold_upright_weight"]*g["fold_phase"]*level,
                recovery_fold_body_rate=-c["fold_body_rate_weight"]*g["fold_phase"]
                    *state["omega"][:, :2].square().sum(-1),
                recovery_ground_skim=-c["fold_wheel_clearance_weight"]*g["fold_phase"]
                    *g["fold_wheel_clearance_error"],
                recovery_fold_wheel_load=-c["fold_wheel_load_weight"]*g["fold_phase"]
                    *g["fold_wheel_load_error"])
    return terms
