"""Training cost on applied leg-target changes; never a runtime action filter."""
import hashlib
import math
from pathlib import Path

import torch

MODULE = 'src/wheeled_tasks/chassis/standing_leg_target_rate.py'
ROOT = Path(__file__).resolve().parents[3]


def contract(weight):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or not 0. <= weight <= 16.):
        raise ValueError('Leg-target density weight must be finite within0..16')
    return dict(schema='standing-leg-target-rate-training-v1', weight_density_per_second=float(weight),
        target_delta_scale_rad=.001, leg_position_scale_rad=.25, policy_dt_s=.005,
        formula='weight*policy_dt*mean_four_legs(log1p((0.25*(applied_action-current_held_action)/0.001)^2))',
        before_step_reference='current actual held action, including the native reset initialization',
        after_step_reference='native diagnostics before any autoreset', startup_grace_s=0.,
        module_sha256={MODULE: hashlib.sha256((ROOT / MODULE).read_bytes()).hexdigest()},
        scope='Zero-request pure-RL standing training only. No action, PD, sensor, reset or acceptance changes; '
              'all six feedback outputs remain trainable. Fixed evaluation is unchanged.')


def validate_contract(value, sources):
    if value != contract(value.get('weight_density_per_second')):
        raise ValueError('Leg-target variation contract changed')
    if any(sources.get(name) != digest for name, digest in value['module_sha256'].items()):
        raise ValueError('Leg-target variation source changed')


def density(delta_rad):
    if delta_rad.ndim != 2 or delta_rad.shape[1] != 4 or not torch.isfinite(delta_rad).all():
        raise ValueError('Need finite simultaneous applied target deltas for four legs')
    return torch.log1p((delta_rad / .001).square()).mean(-1)


def install(env, weight):
    spec = contract(weight)
    if (not hasattr(env, 'precise_standing_reward')
            or hasattr(env, 'standing_leg_target_rate_contract') or env.policy_dt != .005):
        raise ValueError('Attach once to the precise200Hz standing foundation')
    native_step = env.step

    def step(actions):
        # Read the actually held reference at this decision boundary. A reset
        # already initializes it; never carry an old episode's cached target.
        before = env.actions[:, list(env.v5.LEGS)].clone()
        observation, reward, done, extras = native_step(actions)
        if env.fixed_cases is not None:
            return observation, reward, done, extras
        state = extras['diagnostics']
        delta = .25 * (state['actions'][:, list(env.v5.LEGS)] - before)
        value = density(delta)
        eligible = extras['policy_learning_mask'] & (state['requested_command'][:, :2].abs() < .01).all(-1)
        paid = env.policy_dt * weight * value * eligible
        env.standing_leg_target_rate_last = dict(delta_rad=delta, density=value, eligible=eligible, paid=paid)
        extras['log'].update({
            'LegTargetRate/target_delta_rms_rad': delta.square().mean().sqrt(),
            'LegTargetRate/target_delta_peak_rad': delta.abs().max(),
            'LegTargetRate/target_rate_rms_rad_s': delta.square().mean().sqrt() / env.policy_dt,
            'LegTargetRate/cost_density_per_second': (weight * value * eligible).mean(),
            'LegTargetRate/paid_cost_per_policy_step': paid.mean(),
            'LegTargetRate/weight_density_per_second': reward.new_tensor(weight)})
        return observation, reward if weight == 0. else reward - paid, done, extras

    env.step = step
    env.standing_leg_target_rate_contract = spec
    return spec
