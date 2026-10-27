"""Bilateral sensor/action reflection for the paired-107 wheel-leg mechanism.

Body linear vectors are polar; IMU angular velocity is axial. All motor axes
reverse under the L/R reflection, including the opposite-sign wheel axes.
No simulator state, privileged phase, elapsed time or scripted angle is used.
"""
import torch


def reflect_actions(value):
    return -value[:, [2, 3, 0, 1, 5, 4]]


def reflect_observation(obs):
    out = obs.clone()
    out[:, [1, 2, 4, 6, 8]] *= -1.
    out[:, 10:16] = reflect_actions(obs[:, 10:16])
    out[:, 16:22] = reflect_actions(obs[:, 16:22])
    out[:, 22:28] = reflect_actions(obs[:, 22:28])
    return out


def canonical_recovery_logits(head, obs, feature_scale):
    """Share a learned recovery across body sides, with a smooth central seam.

Positive body-frame gravity Y is reflected into the negative-Y half-space.
The 0.15 g blend avoids a switching discontinuity near upright/inverted pitch
poses; the resulting recovery logits obey exact bilateral equivariance.
"""
    weight = ((obs[:, 8:9] / .15 + 1.) * .5).clamp(0., 1.)
    direct = head(torch.tanh(obs * feature_scale))
    reflected = reflect_actions(head(torch.tanh(reflect_observation(obs) * feature_scale)))
    return direct + weight * (reflected - direct)
