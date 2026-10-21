"""Flat support for arbitrary-heading getup and 12 s motion rehearsals.

Eight-metre cells retain the existing local triangle scale. Each environment
receives a separate 96 m square, so high-speed commands and arbitrary reset yaw
cannot turn a narrow test corridor into accidental cliff training.
"""
import math


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
