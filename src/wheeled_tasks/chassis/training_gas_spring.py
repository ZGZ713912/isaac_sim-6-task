"""Bilateral measured force with explicit, authenticated unmeasured extensions."""
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
    policy = declaration["outside_identified_domain"]
    if (policy not in ("endpoint_hold_unvalidated", "positive_c1_endpoint_trend_v1")
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
    result = dict(deepcopy(declaration), model=model,
                measured_compression_domain_m=[curve.compression_knots_m[0], curve.compression_knots_m[-1]],
                physical_pressure_identified=False, right_leg_measured=False,
                installed_stroke_m=model["stroke_m"], enabled=True)
    if policy == "positive_c1_endpoint_trend_v1":
        frozen = declaration["trend_extension"]
        raw_extension = (root/frozen["file"]).read_bytes()
        if hashlib.sha256(raw_extension).hexdigest() != frozen["sha256"]:
            raise ValueError("Spring trend extension contract SHA differs")
        extension = json.loads(raw_extension)
        slopes = [(curve.force_knots_n[1]-curve.force_knots_n[0])/
                  (curve.compression_knots_m[1]-curve.compression_knots_m[0])/curve.force_knots_n[0],
                  (curve.force_knots_n[-1]-curve.force_knots_n[-2])/
                  (curve.compression_knots_m[-1]-curve.compression_knots_m[-2])/curve.force_knots_n[-1]]
        if (extension["policy"] != policy or extension["source_model_sha256"] != declaration["sha256"]
                or extension["soft_knee_domain_deg"] != [40., 107.]
                or extension["physical_compression_domain_m"] != [0., model["stroke_m"]]
                or extension["measured_knots_unchanged"] is not True
                or extension["extrapolation_validated_by_measurement"] is not False
                or len(extension["endpoint_log_slopes_per_m"]) != 2
                or any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                       for a,b in zip(slopes, extension["endpoint_log_slopes_per_m"]))):
            raise ValueError("Spring trend extension differs from measured endpoint slopes")
        result["resolved_trend_extension"] = extension
    return result


def force_from_knots(compression, knots, values, slopes, policy, stroke):
    """Shared physical/encoder force law; force unitsN, compression unitsm."""
    bounded = compression.clamp(knots[0], knots[-1])
    index = torch.searchsorted(knots, bounded.contiguous(), right=True).sub(1).clamp(0, len(knots)-2)
    measured = values[index]+slopes[index]*(bounded-knots[index])
    if policy == "endpoint_hold_unvalidated":
        return measured
    if policy != "positive_c1_endpoint_trend_v1":
        raise ValueError("Unknown gas-spring force extension")
    # Positive exponential tails match both the endpoint force and the SAME
    # one-sided derivative as the measured piecewise-linear curve. Clamp only
    # at physical stroke ends during numerical penetration, not at fit edges.
    physical = compression.clamp(0., stroke)
    low = values[0]*torch.exp((slopes[0]/values[0])*(physical.clamp_max(knots[0])-knots[0]))
    high = values[-1]*torch.exp((slopes[-1]/values[-1])*(physical.clamp_min(knots[-1])-knots[-1]))
    return torch.where(compression < knots[0], low, torch.where(compression > knots[-1], high, measured))


class TrainingGasSpring:
    """The strict measured curve stays unchanged; extension is explicit here."""

    def __init__(self, settings, device):
        if (settings["outside_identified_domain"] not in ("endpoint_hold_unvalidated", "positive_c1_endpoint_trend_v1")
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
        force = force_from_knots(compression, self.knots, self.values, self.slopes,
            self.settings["outside_identified_domain"], self.settings["installed_stroke_m"])
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
