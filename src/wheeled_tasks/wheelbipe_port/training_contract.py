"""Units and finite closed-chain admission for the isolated Wheelbipe port."""
import math


def constraint_limit_m(millimetres):
    value = float(millimetres)
    if not math.isfinite(value) or value<=0.:
        raise ValueError("Closed-chain limit must be finite and positive in millimetres")
    return value/1000.


def check_constraint_gap(gap_m, limit_m, context):
    if not math.isfinite(gap_m) or gap_m>limit_m:
        raise ValueError(f"{context} native loop gap {gap_m}m exceeds {limit_m}m or is nonfinite")
