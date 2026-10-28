"""Explicit bilateral use of measured force, with labelled endpoint extension."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import torch

from .identified_gas_spring import IdentifiedGasSpringCurve


def resolve(root, declaration, spec):
    """Pin the measured input and verify installed compression coordinates."""
    path = root / declaration["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != declaration["sha256"]:
        raise ValueError("Identified gas-spring model SHA differs from the declared input")
    if (declaration["outside_identified_domain"] != "endpoint_hold_unvalidated"
            or declaration["right_leg_symmetry_assumed"] is not True):
        raise ValueError("Bilateral use requires explicit symmetry and bounded endpoint extension")
    model = json.loads(raw)
    if (model["kind"] != "effective_monotone_table" or model["side"] != "left"
            or model["application_sides"] != ["left", "right"]
            or model["right_leg_symmetry_assumed"] is not True
            or model["direction_friction_enabled"] is not False):
        raise ValueError("Need the exported symmetric mean-load model, without nuisance friction")
    curve = IdentifiedGasSpringCurve(tuple(model["compression_knots_m"]), tuple(model["force_knots_n"]))
    curve.for_side("right", assume_symmetry=True)
    if set(spec["spring_binding"]) != {"L_spring_slide", "R_spring_slide"}:
        raise ValueError("Need the declared left/right spring sliders")
    for binding in spec["spring_binding"].values():
        for actual, expected in ((binding["compression_at_q_zero_m"], model["spring_slide_compression_at_q_zero_m"]),
                                 (binding["stroke_m"], model["stroke_m"]),
                                 (binding["full_extension_pin_distance_m"], model["extended_pin_distance_m"]),
                                 (binding["effort_sign"], 1.)):
            if not math.isclose(actual, expected, abs_tol=1e-9, rel_tol=0.):
                raise ValueError("Runtime spring installation differs from measured-model coordinates")
    return dict(deepcopy(declaration), model=model,
                measured_compression_domain_m=[curve.compression_knots_m[0], curve.compression_knots_m[-1]],
                physical_pressure_identified=False, right_leg_measured=False,
                installed_stroke_m=model["stroke_m"], enabled=True)


class TrainingGasSpring:
    """The strict measured curve stays unchanged; extension is explicit here."""

    def __init__(self, settings, device):
        if (settings["outside_identified_domain"] != "endpoint_hold_unvalidated"
                or not settings["right_leg_symmetry_assumed"]):
            raise ValueError("Missing explicit gas-spring training assumptions")
        model = settings["model"]
        self.curve = IdentifiedGasSpringCurve(tuple(model["compression_knots_m"]), tuple(model["force_knots_n"]))
        self.right_curve = self.curve.for_side("right", assume_symmetry=True)
        self.settings = deepcopy(settings)
        self.knots = torch.tensor(self.curve.compression_knots_m, device=device)
        self.values = torch.tensor(self.curve.force_knots_n, device=device)
        self.slopes = self.values.diff() / self.knots.diff()

    def force(self, compression):
        if not bool(torch.isfinite(compression).all()):
            raise ValueError("Nonfinite spring compression")
        bounded = compression.clamp(self.knots[0], self.knots[-1])
        index = torch.searchsorted(self.knots, bounded.contiguous(), right=True).sub(1).clamp(0, len(self.knots)-2)
        force = self.values[index] + self.slopes[index] * (bounded-self.knots[index])
        self.last_compression = compression
        self.last_force = force
        self.last_outside = (compression < self.knots[0]) | (compression > self.knots[-1])
        return force

    def install(self, env):
        # Replace the old polynomial at its actual 1kHz physical force call.
        # Preserve the existing domain multiplier (fixed1 in current v2).
        env.v5.spring_efforts = lambda position: self.force(env.v5.s0-position) * env.domain.spring
        env.startup_report["identified_gas_spring"] = deepcopy(self.settings)
        env.startup_report["identified_gas_spring"].update(
            applied_to=["L_spring_slide", "R_spring_slide"], force_units="N, positive outward",
            source_calibration_side="left", same_axial_curve_both_sides=True,
            domain_multiplier_preserved=True, physical_call_hz=round(1./env.dt),
            default_catalogue_force_replaced_for_this_task=True,
            unmeasured_extension_is_not_calibrated=True)
