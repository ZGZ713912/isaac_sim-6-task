"""Explicit chassis motor priors, distinct from the legacy V4 policy ABI."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .preflight import control_decimation


SCHEMA = "chassis-explicit-motor-control-v1"
WHEEL_GEAR_RATIO = 15.62
LEGACY_WHEEL_CONTROL_IDS = frozenset((
    "v6-asset-capacity-control-v1", "v6-nominal-research-control-v1",
    "v6-wheel-response-control-v1", "v6-wheel-response-control-200hz-v1",
    "v6-wheel-response-control-policy200hz-v1", "flat-base-policy-200hz-control-v1",
))
ACTIVE_ORDER = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
COARSE_TIMING = {"physics_dt": .005, "policy_dt": .02, "pc_control_dt": .005, "decimation": 4}
COARSE_SCOPE = {"model": "200hz_feedback_pd_with_1ms_transport_impulse",
                "hardware_pd_dt": .001, "raw_bag_dt": .001,
                "fresh_feedback_per_policy_step": 4, "hardware_equivalence_verified": False}
POLICY_200_TIMING = {"physics_dt": .005, "policy_dt": .005, "pc_control_dt": .005, "decimation": 1}
POLICY_200_SCOPE = {**COARSE_SCOPE, "model": "200hz_policy_and_feedback_pd_with_1ms_transport_impulse",
                    "fresh_feedback_per_policy_step": 1}
FINE_POLICY_200_TIMING = {"physics_dt": .001, "policy_dt": .005, "pc_control_dt": .001, "decimation": 5}


def declared_control_timing(control):
    if "simulation_timing_approximation" not in control:
        if control.get("control_id") in ("flat-base-policy-200hz-control-v1", "flat-base-policy-200hz-control-v2"):
            return dict(FINE_POLICY_200_TIMING)
        return {"physics_dt": .001, "policy_dt": .02, "pc_control_dt": .001, "decimation": 20}
    scope = control["simulation_timing_approximation"]
    if scope == COARSE_SCOPE:
        return dict(COARSE_TIMING)
    if scope == POLICY_200_SCOPE:
        return dict(POLICY_200_TIMING)
    raise ValueError("Unknown simulation motor-control timing approximation")


def load_chassis_control(path):
    def reject(value):
        raise ValueError(f"Nonfinite motor-control value: {value}")
    control = json.loads(Path(path).read_text(), parse_constant=reject)
    if control.get("schema") != SCHEMA:
        from wheeled_tasks.v40.contract import load_contract
        return load_contract(path)
    return validate_chassis_control(control)


def validate_chassis_control(control):
    """Validate explicit output-side priors without claiming hardware identification."""
    capacity = control.get("engineering_probe_only") is True and not control.get("research_training", False)
    research = (control.get("engineering_probe_only") is False and control.get("research_training") is True
                and control.get("hardware_deployment_ready") is False)
    if control.get("schema") != SCHEMA or not (capacity or research) or control.get("training_release") is not False:
        raise ValueError("Explicit motor prior must declare capacity or simulation-research scope, without hardware release")
    if control.get("control_frame") != "Xforward_Yleft_Zup" or control.get("active_order") != ACTIVE_ORDER:
        raise ValueError("Explicit motor prior requires canonical six active-output axes")
    timing = control["timing"]
    coarse = "simulation_timing_approximation" in control
    expected = declared_control_timing(control)
    if (control_decimation(timing["physics_dt"], timing["policy_dt"]) != expected["decimation"]
            or timing != expected or (coarse and not research)):
        raise ValueError("Motor prior requires its declared physical, policy and feedback-PD timing")
    leg, wheel = control["actuators"]["leg"], control["actuators"]["wheel"]
    for key, expected in (("kp", 160.), ("kd", 2.5), ("effort_limit", 40.)):
        if leg.get(key) != expected:
            raise ValueError(f"Engineering leg prior {key} differs from selected PD")
    required = {"model", "gear_ratio", "gearbox_efficiency", "kd", "armature", "effort_limit",
                "curve_side", "curve_kind", "torque_reference", "scope", "source_note"}
    if set(wheel) != required:
        raise ValueError("Explicit wheel-output cap must not include an ambiguous motor-side curve or conversion")
    # Only the named frozen controls retain their historical 15.8 identity.
    # New controls use the full rotor-to-wheel ratio, matching the RMCS driver.
    # A wheel-output cap must still never be multiplied by this ratio again.
    ratio = 15.8 if control.get("control_id") in LEGACY_WHEEL_CONTROL_IDS else WHEEL_GEAR_RATIO
    for key, expected in (("model", "M3508"), ("gear_ratio", ratio), ("gearbox_efficiency", 1.),
                          ("kd", .6), ("effort_limit", 4.5), ("curve_side", "wheel_output"),
                          ("torque_reference", "wheel_output"),
                          ("curve_kind", "software_cap_only_not_identified"),
                          ("scope", "engineering_capacity_only" if capacity else "simulation_research_prior")):
        if wheel[key] != expected:
            raise ValueError(f"Engineering wheel prior {key} differs from the explicit output-side cap")
    for module in (leg, wheel):
        value = module["armature"]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Armature must be finite and nonnegative")
    return control
