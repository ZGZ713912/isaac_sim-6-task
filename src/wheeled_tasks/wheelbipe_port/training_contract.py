"""Explicit physical settings and optional historical closed-chain admission."""
import math

CONSTRAINT_DIAGNOSTIC_SCOPE = "policy observation boundaries only; not a per-physics-step maximum"
DEFAULT_LEG_ARMATURE_KG_M2 = .015795


def projection_penalty(value):
    value = float(value)
    if not math.isfinite(value) or value<0.:
        raise ValueError("Target-projection penalty must be finite and nonnegative")
    return value


def leg_pd_gains(kp,kd):
    kp,kd = float(kp),float(kd)
    if not math.isfinite(kp) or kp<=0. or not math.isfinite(kd) or kd<0.:
        raise ValueError("Leg PD requires finite Kp>0 and Kd>=0")
    return kp,kd


def target_overtravel_rad(value):
    value = float(value)
    if not math.isfinite(value) or not 0. <= value <= .35:
        raise ValueError("Virtual pair-target overtravel must be within 0..0.35 rad")
    return value


def leg_armature_kg_m2(value):
    value = float(value)
    if not math.isfinite(value) or value<0.:
        raise ValueError("Leg armature must be finite and nonnegative in kg*m^2")
    return value


def constraint_limit_m(millimetres):
    if millimetres is None:
        return None
    value = float(millimetres)
    if not math.isfinite(value) or value<=0.:
        raise ValueError("Closed-chain limit must be finite and positive in millimetres")
    return value/1000.


def check_constraint_gap(gap_m, limit_m, context):
    if not math.isfinite(gap_m):
        raise ValueError(f"{context} native loop diagnostic is nonfinite")
    if limit_m is not None and gap_m>limit_m:
        raise ValueError(f"{context} native loop gap {gap_m}m exceeds {limit_m}m or is nonfinite")
