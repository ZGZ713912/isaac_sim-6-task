"""Training-only cost for each actual interruption of dual-wheel support."""
import hashlib
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE = 'src/wheeled_tasks/chassis/standing_contact_continuity.py'
MONITOR = 'src/wheeled_tasks/chassis/standing_contact_monitor.py'
SCHEMA = 'standing-contact-continuity-training-v2'


def contract(cost):
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or not 0. <= cost <= .2:
        raise ValueError('Contact break cost must be finite within0..0.2 reward per event')
    return dict(schema=SCHEMA, reward_cost_per_break=float(cost), force_threshold_n=2.,
        physical_dt_s=.001, policy_dt_s=.005, startup_grace_s=0.,
        module_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (MODULE, MONITOR)},
        formula='cost*count_1ms(supported_previous_ms and not supported_current_ms)',
        support='both wheel contact force norms strictly exceed2N; previous support initialized true at explicit grounded reset',
        scope='zero-request non-recovery pure-RL standing training only; dense load and separation cost retained; fixed evaluation unchanged; no added termination')


def validate_contract(value, sources):
    if not isinstance(value, dict) or value != contract(value.get('reward_cost_per_break')):
        raise ValueError('Standing contact continuity specification changed')
    if any(sources.get(k) != v for k, v in value['module_sha256'].items()):
        raise ValueError('Standing contact continuity dependencies changed')


def break_counts(previous, present):
    if present.ndim != 2 or previous.shape != present.shape[1:] or present.dtype != torch.bool or previous.dtype != torch.bool:
        raise ValueError('Need native bool[substep,environment] and previous bool[environment]')
    predecessor = torch.cat((previous[None], present[:-1]), dim=0)
    return (predecessor & ~present).sum(dim=0)


def install(env, cost):
    contact_objective = (hasattr(env, 'standing_contact_training_contract')
                         or hasattr(env, 'physical_standing_reward'))
    if (env.fixed_cases is not None or not contact_objective
            or not hasattr(env, 'grounded_standing_reset') or hasattr(env, 'standing_contact_continuity_contract')):
        raise ValueError('Continuity training requires one native contact-objective and grounded-reset environment')
    spec = contract(cost)
    previous = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    generation = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    native_reset, native_step = env.reset, env.step

    def reset(ids):
        result = native_reset(ids)
        previous[ids] = True
        generation[ids] += 1
        return result

    def step(actions):
        if env.fixed_cases is not None:
            return native_step(actions)
        standing = (~env.recovery_lane) & (env.pending_command[:, :2].abs() < .01).all(-1) & ~env.passive
        start, before = previous.clone(), generation.clone()
        obs, reward, done, extras = native_step(actions)
        present = (env.standing_contact_monitor.last['force'].norm(dim=-1) > 2.).all(-1)
        counts = break_counts(start, present)
        same_episode = generation == before
        previous[same_episode] = present[-1, same_episode]
        mask = standing & extras['policy_learning_mask']
        penalty = counts * cost * mask
        env.standing_contact_continuity_last = dict(break_count=counts.clone(), penalty=penalty.detach().clone(),
            eligible=mask.clone(), same_episode=same_episode)
        extras['log'].update({
            'StandingContinuity/support_break_events_per_policy_step': counts[mask].float().mean() if mask.any() else reward.new_zeros(()),
            'StandingContinuity/paid_penalty_per_policy_step': penalty.mean(),
            'StandingContinuity/reward_cost_per_break': reward.new_tensor(cost)})
        return obs, reward if cost == 0. else reward - penalty, done, extras

    env.reset, env.step = reset, step
    env.standing_contact_continuity_contract = spec
    return spec
