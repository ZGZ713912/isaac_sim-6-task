"""Training-only dense cost for uninterrupted, loaded wheel support."""
import hashlib
import math
from pathlib import Path

import torch

from .standing_contact_monitor import StandingContactMonitor

MODULE = 'src/wheeled_tasks/chassis/standing_contact_reward.py'
MONITOR = 'src/wheeled_tasks/chassis/standing_contact_monitor.py'
RESET = 'src/wheeled_tasks/chassis/grounded_standing_reset.py'
SCHEMA = 'strict-standing-contact-training-v1'
ROOT = Path(__file__).resolve().parents[3]


def contract(weight, grounded_reset=False):
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0. <= weight <= 16.:
        raise ValueError('Contact density weight must be finite within0..16')
    if not isinstance(grounded_reset, bool):
        raise ValueError('Standing reset protocol must be an explicit boolean')
    paths = (MODULE, MONITOR, RESET) if grounded_reset else (MODULE, MONITOR)
    return dict(schema=SCHEMA, weight_density_per_second=float(weight), grounded_reset=bool(grounded_reset),
        module_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        physical_dt_s=.001, policy_dt_s=.005, force_margin_n=20., clearance_scale_m=.0002,
        formula='weight*policy_dt*mean_5substeps(max_wheel((1-clamp(Fz/20,0,1))^2+log1p((max(clearance,0)/0.0002)^2)))',
        scope='zero-request non-recovery pure-RL standing only; actor ABI and runtime function unchanged')


def validate_contract(value, sources):
    if not isinstance(value, dict) or value != contract(value.get('weight_density_per_second'), value.get('grounded_reset')):
        raise ValueError('Standing contact objective specification changed')
    if any(sources.get(k) != v for k, v in value['module_sha256'].items()):
        raise ValueError('Standing contact training dependencies changed')


def support_density(force, clearance):
    if force.shape[:-1] != clearance.shape or force.shape[-2:] != (2, 3):
        raise ValueError('Need simultaneous native[substep,environment,wheel,xyz] contacts')
    if not torch.isfinite(force).all() or not torch.isfinite(clearance).all():
        raise ValueError('Nonfinite contact reward input')
    load_deficit = (1. - (force[..., 2] / 20.).clamp(0., 1.)).square()
    separation = torch.log1p((clearance.clamp_min(0.) / .0002).square())
    return (load_deficit + separation).amax(-1).mean(0)


def install(env, weight, *, grounded_reset=False):
    if env.fixed_cases is not None or hasattr(env, 'standing_contact_training_contract'):
        raise ValueError('Attach once to a native training environment')
    spec = contract(weight, grounded_reset)
    if grounded_reset:
        from .grounded_standing_reset import install as install_reset
        install_reset(env)
    monitor = getattr(env, 'standing_contact_monitor', None) or StandingContactMonitor(env)
    native_step = env.step

    def step(actions):
        standing = (~env.recovery_lane) & (env.pending_command[:, :2].abs() < .01).all(-1) & ~env.passive
        obs, reward, done, extras = native_step(actions)
        sample = monitor.last
        density = support_density(sample['force'], sample['clearance'])
        mask = standing & extras['policy_learning_mask']
        penalty = weight * env.policy_dt * density * mask
        env.standing_contact_reward_last = dict(native_reward=reward.detach().clone(), penalty=penalty.detach().clone(),
            density=density.detach().clone(), eligible=mask.clone())
        extras['log'].update({
            'StrictStanding/support_every_1ms_fraction': sample['contact_every_ms'][mask].float().mean() if mask.any() else reward.new_zeros(()),
            'StrictStanding/missing_physical_ms_per_policy_step': sample['missing_ms'][mask].float().mean() if mask.any() else reward.new_zeros(()),
            'StrictStanding/maximum_wheel_separation_m': sample['clearance_peak_m'][mask].max() if mask.any() else reward.new_zeros(()),
            'StrictStanding/minimum_wheel_vertical_load_n': sample['normal_load_min_n'][mask].min() if mask.any() else reward.new_zeros(()),
            'StrictStanding/support_cost_density_per_second': (density * mask).mean(),
            'StrictStanding/paid_penalty_per_policy_step': penalty.mean(),
            'StrictStanding/weight_density_per_second': reward.new_tensor(weight)})
        return obs, reward if weight == 0. else reward - penalty, done, extras

    env.step = step
    env.standing_contact_training_contract = spec
    return spec
