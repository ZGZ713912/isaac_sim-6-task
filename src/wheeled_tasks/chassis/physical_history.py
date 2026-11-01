"""Physical milliseconds to causal history samples, including sub-tick delays."""
import math

import torch


def split_delay_ms(milliseconds, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Physical history requires a positive finite dt")
    ticks = milliseconds / (dt * 1000.)
    whole = torch.floor(ticks).to(torch.long)
    return whole, ticks - whole


def delayed_sample(history, cursor, whole, fraction, rows):
    recent = history[(cursor - whole) % len(history), rows]
    older = history[(cursor - whole - 1) % len(history), rows]
    return recent * (1. - fraction[:, None]) + older * fraction[:, None]
