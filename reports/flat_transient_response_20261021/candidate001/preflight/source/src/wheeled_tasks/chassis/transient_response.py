"""Transient costs allow useful pitch without rewarding leg motion itself."""
from copy import deepcopy
import math

import torch


REWARD_SETTINGS = dict(version='leg-assisted-response-cost-v1',
    vx_scale_m_s=.5, yaw_scale_rad_s=.5, pitch_allowance_rad=math.radians(8.),
    leg_smoothness_scale=.5, response_error_weight=1.5)


def validate_settings(settings):
    if settings != REWARD_SETTINGS:
        raise ValueError('Transient response rewards differ from the declared candidate')
    return deepcopy(settings)


class TransientResponse:
    def __init__(self, settings, device):
        self.settings = validate_settings(settings)
        self.sums = torch.zeros(6, dtype=torch.float64, device=device)
        self.frames = 0

    def apply(self, terms, velocity, omega, gravity, commands, ordinary, supported, dq):
        s = self.settings
        vx = velocity[:, 0] * (1. - gravity[:, 0].square()).clamp_min(0.).sqrt()
        ex, ey = (vx - commands[:, 0]).abs(), (omega[:, 2] - commands[:, 1]).abs()
        weight = torch.maximum(ex / s['vx_scale_m_s'], ey / s['yaw_scale_rad_s']).clamp(0., 1.)
        weight = weight * ordinary.to(weight.dtype)
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2]).abs()
        excess = (pitch - s['pitch_allowance_rad']).clamp_min(0.)
        relax = weight * supported.to(weight.dtype)
        # Relax pitch only while tracking is wrong and both wheels support the
        # body. Roll, height, effort, contact and mechanical guards are retained.
        ratio = (excess / pitch.clamp_min(1e-6)).square()
        for name in ('pitch_velocity', 'locomotion_pitch_floor'):
            if name in terms:
                terms[name] = terms[name] * (1. - relax + relax * ratio)
        if 'pitch_exp' in terms:
            allowed = torch.exp(-excess.sin().square() / .02)
            terms['pitch_exp'] = terms['pitch_exp'] + relax * (allowed - terms['pitch_exp'])
        if 'leg_action_smoothness' in terms:
            terms['leg_action_smoothness'] *= 1. - relax * (1. - s['leg_smoothness_scale'])
        # This is a density; the environment integrates it exactly once by dt.
        terms['transient_response_error'] = -s['response_error_weight'] * weight * (
            (ex / s['vx_scale_m_s']).clamp_max(6.) + (ey / s['yaw_scale_rad_s']).clamp_max(8.))
        values = torch.stack((weight, ex, ey, pitch * (180. / math.pi),
                              dq[:, [0, 1, 3, 4]].abs().mean(-1), relax), -1)
        self.sums += values.detach().double().sum(0)
        self.frames += len(velocity)

    def drain(self):
        if not self.frames:
            return {}
        names = ('active_weight_mean', 'vx_abs_error_m_s', 'yaw_abs_error_rad_s',
                 'pitch_abs_deg', 'leg_speed_mean_rad_s', 'supported_relaxation_mean')
        result = {'FlatTransient/' + key: value for key, value in
                  zip(names, (self.sums / self.frames).cpu().tolist(), strict=True)}
        result['FlatTransient/sample_frames'] = self.frames
        self.sums.zero_()
        self.frames = 0
        return result
