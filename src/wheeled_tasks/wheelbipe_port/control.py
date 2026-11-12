"""Encoder-based, intent-gated spring cancellation; no simulator state inputs."""
from __future__ import annotations

import math
import torch

DEFAULT_TUCK_COMPENSATION = dict(mode="off", gain=.75, cap_nm=16.,
    error_on_rad=.02, error_full_rad=.10, inward_pd_full_nm=2.)


def validate_tuck_compensation(settings):
    if settings["mode"] not in ("off", "model"):
        raise ValueError("Tuck spring compensation must be off or model")
    for name, low, high in (("gain", 0., 1.), ("cap_nm", 0., 40.),
                            ("error_on_rad", 0., .35), ("error_full_rad", 0., 1.),
                            ("inward_pd_full_nm", 0., 40.)):
        value = float(settings[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid tuck spring compensation "+name)
    if (settings["error_full_rad"] <= settings["error_on_rad"]
            or settings["inward_pd_full_nm"] <= 0.):
        raise ValueError("Tuck compensation needs nonzero error and PD ramps")
    return settings


def smooth_ramp(value):
    value = value.clamp(0., 1.)
    return value.square()*(3.-2.*value)


def tuck_spring_effort(position, target, pd_effort, eligible, holding_effort, settings):
    """Four active axes in L hip/aux, R hip/aux order, motor-output units.

    Eligibility comes from the external jump request clock. Actual contact,
    root state, apex and the reward observer must never be passed here.
    Cancel only spring extension opposing a requested inward PD effort;
    common leg rotation and extension/braking commands receive zero help.
    """
    if (position.shape != target.shape or position.shape != pd_effort.shape
            or position.shape != holding_effort.shape or position.shape[-1] != 4
            or eligible.shape != position.shape[:-1]):
        raise ValueError("Tuck compensation requires aligned four-axis inputs and eligibility")
    coefficients = position.new_tensor([[-1.,1.],[1.,-1.]])
    relative = lambda value: (value.unflatten(-1,(2,2))*coefficients).sum(-1)
    error = relative(position)-relative(target)
    intent = smooth_ramp((error-settings["error_on_rad"])/
                         (settings["error_full_rad"]-settings["error_on_rad"]))
    inward_pd = -.5*relative(pd_effort)
    braking_gate = smooth_ramp(inward_pd/settings["inward_pd_full_nm"])
    gate = intent*braking_gate*eligible[...,None]
    if settings["mode"] == "off":
        gate = torch.zeros_like(gate)
    # The authenticated holding map is already zero outside its valid
    # mechanical branch. It is never evaluated at the virtual target.
    inward = (-.5*relative(holding_effort)).clamp_min(0.)
    magnitude = (inward*settings["gain"]).clamp_max(settings["cap_nm"])*gate
    effort = (-coefficients*magnitude[...,None]).flatten(-2)
    return effort, gate
