"""A tilted finite-width wheel must not be mistaken for a sphere or force sensor."""
import math

import torch

from wheeled_tasks.chassis.standing_support import wheel_bottom_clearance


def test_vertical_and_horizontal_wheel_cylinders_have_different_extents():
    center = torch.tensor([[[0., 0., .06], [0., 0., .0125]]])
    quat = torch.tensor([[[math.sqrt(.5), 0., 0., math.sqrt(.5)], [0., 0., 0., 1.]]])
    torch.testing.assert_close(wheel_bottom_clearance(center, quat, torch.zeros(1, 1)),
                               torch.zeros(1, 2), atol=1e-8, rtol=0)


def test_tilted_width_and_ground_translation_are_measured_in_world_z():
    angle = math.radians(60.)
    extent = .06 * math.sin(angle) + .0125 * math.cos(angle)
    center = torch.tensor([[[108., -97., 4. + extent + .007]]], dtype=torch.float64)
    quat = torch.tensor([[[math.sin(angle/2), 0., 0., math.cos(angle/2)]]], dtype=torch.float64)
    actual = wheel_bottom_clearance(center, quat, torch.tensor([[4.]]), length=.025)
    torch.testing.assert_close(actual, torch.tensor([[.007]], dtype=torch.float64), atol=1e-12, rtol=0)


def test_rotation_about_wheel_axis_preserves_cylinder_clearance():
    half = math.sqrt(.5)
    # R_x(90 deg) followed by any rotation around local cylinder Z.
    spin = 1.7
    quat = torch.tensor([[[half, 0., 0., half],
                          [half*math.cos(spin/2), -half*math.sin(spin/2),
                           half*math.sin(spin/2), half*math.cos(spin/2)]]])
    center = torch.tensor([[[0., 0., .07], [0., 0., .07]]])
    actual = wheel_bottom_clearance(center, quat, torch.zeros(1, 1))
    torch.testing.assert_close(actual, torch.full((1, 2), .01), atol=1e-7, rtol=0)
