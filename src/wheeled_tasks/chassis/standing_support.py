"""Geometric wheel clearance for flat-floor standing, independent of force thresholds."""
import torch


def wheel_bottom_clearance(centers_w, quaternions_xyzw, floor_z, radius=.06, length=.02500000037252903):
    """Lowest point of a circular cylinder whose axis is the link's local Z.

    The authenticated 107-degree asset has identity wheel-collider rotations.
    This includes wheel width when tilted; subtracting only the radius would
    incorrectly report the geometry of a sphere. It is an analytic collider
    measure, not a claim of mesh/contact-patch resolution.
    """
    axial_z = (1. - 2. * quaternions_xyzw[..., :2].square().sum(-1)).clamp(-1., 1.)
    extent_z = radius * (1. - axial_z.square()).clamp_min(0.).sqrt() + .5 * length * axial_z.abs()
    return centers_w[..., 2] - extent_z - floor_z
