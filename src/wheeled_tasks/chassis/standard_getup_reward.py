"""Continuous state guidance for standard-prone exploration, with no trajectory."""
from copy import deepcopy

import torch


def reward_contract():
    return dict(schema="wheel-leg-standard-getup-dense-v1", target_height_m=.305,
        height_weight=2., height_width_m=.06, height_error_weight=2., height_error_scale_m=.08,
        upright_error_weight=2., support_weight=1., support_scale_m=.12, body_rate_weight=.02,
        units="reward_per_second", activation="original_standard_prone_before_completion")


def validate(config):
    if config != reward_contract():
        raise ValueError("Standard getup needs its complete versioned dense reward contract")
    return deepcopy(config)


def dense_terms(state, config):
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
    return dict(recovery_height=c["height_weight"]*up*torch.exp(-(height_error/c["height_width_m"]).square()),
        recovery_height_error=-c["height_error_weight"]*torch.log1p((height_error/c["height_error_scale_m"]).square()),
        recovery_upright=-c["upright_error_weight"]*level,
        recovery_support=c["support_weight"]*up*aligned,
        recovery_body_rate=-c["body_rate_weight"]*state["omega"].square().sum(-1))
