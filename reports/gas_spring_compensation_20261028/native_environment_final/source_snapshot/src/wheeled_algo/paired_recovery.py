"""Sensor-only circular leg targets, preserving each closed-chain knee angle."""
import torch


NOMINAL_DELTA = (.5574228259539536, -.5574155762580851)
DELTA_BOUNDS = ((-.14062820701680934, 1.3692123767724917),
                (-.1406283965709381, 1.3692125038958158))


def wrap(angle):
    return torch.atan2(angle.sin(), angle.cos())


def paired_offsets(offsets, measured_wrapped=None, max_lead=.35):
    """Wrap the common angle once for both motors; never wrap knees separately.

    Encoder observations wrap each motor independently. Their small difference
    recovers the common measured phase even when the pair straddles +/-pi.
    The lead bound prevents an accumulated target from lapping the mechanism.
    """
    pair = offsets.unflatten(-1, (2, 2))
    nominal = offsets.new_tensor(NOMINAL_DELTA)
    sign = offsets.new_tensor([1., -1.])
    bounds = offsets.new_tensor(DELTA_BOUNDS)
    delta = sign * (nominal + pair[..., 1] - pair[..., 0])
    offset_delta = sign * delta.clamp(bounds[:, 0], bounds[:, 1]) - nominal
    common = pair.mean(-1)
    if measured_wrapped is not None:
        measured = measured_wrapped.unflatten(-1, (2, 2))
        measured_common = measured[..., 0] + .5 * wrap(measured[..., 1] - measured[..., 0])
        common = measured_common + wrap(common - measured_common).clamp(-max_lead, max_lead)
    common = wrap(common)
    return torch.stack((common - .5 * offset_delta, common + .5 * offset_delta), -1).flatten(-2)
