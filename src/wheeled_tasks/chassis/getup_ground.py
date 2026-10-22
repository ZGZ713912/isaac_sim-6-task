"""Flat support for arbitrary-heading getup and 12 s motion rehearsals.

Eight-metre cells retain the existing local triangle scale. Each environment
receives a separate 96 m square, so high-speed commands and arbitrary reset yaw
cannot turn a narrow test corridor into accidental cliff training.
"""
import math


def world_layout_bounds(count, half_length=48., width=96.):
    """Bounds of the centered corridor layout used by ChassisEnv."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("A positive native environment count is required")
    spacing = 2 * half_length + 12.
    columns = max(1, math.ceil(math.sqrt(count * (width + 1.) / spacing)))
    rows = math.ceil(count / columns)
    origin = [(columns - 1) * spacing / 2, (rows - 1) * (width + 1.) / 2]
    return dict(columns=columns, rows=rows, max_abs_origin_xy_m=origin,
                max_abs_support_xy_m=[origin[0] + half_length, origin[1] + width / 2])


def require_bounded_world(count):
    layout = world_layout_bounds(count)
    if max(layout["max_abs_support_xy_m"]) > 1024.:
        raise ValueError("Full-RL ground exceeds the checked world-coordinate range; "
                         "reduce native batch size or use the explicit large-world diagnostic override")
    return layout


def flat_patch(half_length=48., width=96., segment_length=8.):
    if not (0 < half_length <= 64 and 0 < width <= 128 and 0 < segment_length <= 8):
        raise ValueError("Getup patch dimensions exceed bounded native scene")
    nx = math.ceil(2 * half_length / segment_length)
    ny = math.ceil(width / segment_length)
    points = [(float(-half_length + 2 * half_length * i / nx), float(-width / 2 + width * j / ny), 0.)
              for i in range(nx + 1) for j in range(ny + 1)]
    indices = []
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            b = (i + 1) * (ny + 1) + j
            indices.extend((a, b, a + 1, a + 1, b, b + 1))
    return points, indices
