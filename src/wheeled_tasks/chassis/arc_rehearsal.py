"""Explicit moderate simultaneous requests for a bounded motion-only pilot."""
import torch


def moderate_arc_commands(draws):
    """Four independent uniform draws: vx magnitude/sign and yaw magnitude/sign."""
    vx = (.4 + .85 * draws[:, 0]) * torch.where(draws[:, 1] < .5, -1., 1.)
    yaw = (.4 + draws[:, 2]) * torch.where(draws[:, 3] < .5, -1., 1.)
    return torch.stack((vx, yaw), -1)
