"""State-based orient, capture and lift guidance for arbitrary fallen poses.

The phases follow quadruped/humanoid fall-recovery curricula, but do not
prescribe a trajectory or expose contacts, family labels or phase to the actor.
"""
from copy import deepcopy
import math

import torch


def contract():
    return dict(schema="wheel-leg-phase-recovery-reward-v1", units="reward_per_second",
        target_height_m=.305, height_width_m=.06, support_scale_m=.12,
        orient_weight=2., capture_weight=2., lift_weight=4., height_error_weight=2.,
        roll_rate_weight=.01, capture_rate_weight=.12, horizontal_speed_weight=.2,
        body_contact_weight=4., wheel_load_full_fraction=.15,
        observation_additions=False, motor_targets=False)


def validate(config):
    if config != contract():
        raise ValueError("Recovery shaping requires the complete phase-reward contract")
    return deepcopy(config)


def dense_terms(state, config, *, scores=None):
    c = config
    up = (-state["gravity"][:, 2]).clamp(0., 1.)
    alignment = 1./(1.+(state["recovery_support_error_m"]/c["support_scale_m"]).square())
    load = (state["recovery_wheel_load_fraction"]/c["wheel_load_full_fraction"]).clamp(0., 1.).amin(-1)
    capture = up.square()*alignment*load
    body_rate = state["omega"][:, :2].square().sum(-1)
    error = state["height"]-c["target_height_m"]
    # Shift the lift error to zero at belly height, avoiding a negative
    # discontinuity when the wheels first begin to support the body.
    reference = math.log1p(((.173-c["target_height_m"])/.08)**2)
    return dict(
        recovery_orientation=c["orient_weight"]*(-state["gravity"][:, 2]).clamp(-1., 1.),
        recovery_support_capture=c["capture_weight"]*capture,
        recovery_lift=c["lift_weight"]*capture*torch.exp(-(error/c["height_width_m"]).square()),
        recovery_lift_error=c["height_error_weight"]*capture*(reference-torch.log1p((error/.08).square())),
        recovery_phase_body_rate=-(c["roll_rate_weight"]+c["capture_rate_weight"]*capture)*body_rate,
        recovery_capture_speed=-c["horizontal_speed_weight"]*capture*state["velocity"][:, :2].square().sum(-1),
        recovery_capture_body_contact=-c["body_contact_weight"]*capture*state["recovery_chassis_load_fraction"])
