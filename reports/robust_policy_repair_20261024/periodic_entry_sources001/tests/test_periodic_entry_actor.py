from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.gap_entry_preservation import EntryPreservingGapActor
from wheeled_algo.getup_symmetry import reflect_actions, reflect_observation
from wheeled_algo.periodic_entry_actor import (
    PeriodicEntryGapActor, periodic_entry_logits, periodic_canonical_entry_logits,
    validate_periodic_entry_checkpoint,
)


def head():
    torch.manual_seed(2131)
    return nn.Sequential(nn.Linear(35, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 6))


def test_nonseam_logits_are_bit_exact_and_gradients_remain_finite():
    h = head()
    obs = torch.randn(80, 35)
    obs[:, 10:14] = obs[:, 10:14].clamp(-2.7, 2.7)
    scale = torch.ones(35)
    expected = h(torch.tanh(obs*scale))
    actual = periodic_entry_logits(h, obs, scale)
    assert torch.equal(expected, actual)
    obs[:, 10:14] = 3.05
    obs.requires_grad_()
    periodic_entry_logits(h, obs, scale).square().sum().backward()
    assert torch.isfinite(obs.grad).all() and obs.grad[:, 10:14].abs().sum() > 0


@pytest.mark.parametrize('encoder', [10, 11, 12, 13])
def test_one_encoder_cut_is_continuous(encoder):
    h = head()
    obs = torch.randn(1, 35).repeat(2, 1)
    obs[:, 10:14] = .2
    obs[0, encoder] = math.pi-1e-5
    obs[1, encoder] = -math.pi+1e-5
    out = periodic_entry_logits(h, obs, torch.ones(35))
    torch.testing.assert_close(out[0], out[1], rtol=0., atol=2e-5)


def test_all_sixteen_sides_of_simultaneous_cuts_meet():
    h = head()
    obs = torch.randn(1, 35).repeat(16, 1)
    for mask in range(16):
        for j in range(4):
            obs[mask, 10+j] = (math.pi-1e-6)*(1 if mask & (1 << j) else -1)
    out = periodic_entry_logits(h, obs, torch.ones(35))
    torch.testing.assert_close(out, out[:1].expand_as(out), rtol=0., atol=2e-5)


def test_periodic_canonical_head_retains_bilateral_equivariance():
    h = head()
    obs = torch.randn(30, 35)
    obs[:, 10:14] = torch.empty(30, 4).uniform_(-math.pi, math.pi)
    scale = torch.ones(35)
    a = periodic_canonical_entry_logits(h, obs, scale)
    b = reflect_actions(periodic_canonical_entry_logits(h, reflect_observation(obs), scale))
    torch.testing.assert_close(a, b, rtol=0., atol=1e-6)


def test_full_actor_state_and_nonseam_or_inactive_entry_path_stay_exact():
    torch.manual_seed(2132)
    kw = dict(upright_entry_recovery=True, canonical_recovery=True, smooth_knee_targets=True,
              recovery_leg_step=.3, final_target_winding_version='paired-v1',
              cyclic_leg_targets=True, leg_action_clip=16.)
    old = EntryPreservingGapActor(nn.Linear(35, 6), **kw)
    new = PeriodicEntryGapActor(deepcopy(old.parent), **kw)
    new.load_state_dict(old.state_dict(), strict=True)
    obs = torch.randn(80, 35)
    obs[:, 3] = .305*5.
    obs[:, 9] = -1.
    obs[:, 10:14] = obs[:, 10:14].clamp(-2.7, 2.7)
    assert torch.equal(old(obs), new(obs))
    obs[:, 10:14] = 3.05
    obs[:, 9] = 1.
    assert not handoff_entry_gate(obs).any()
    assert torch.equal(old(obs), new(obs))
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


def migration_states():
    prior = dict(infos=dict(completed_updates=1501, settings=dict(gap_entry_recovery_split_version='frozen-v1'),
                           source_sha256={}, rng=torch.arange(4)), actor_state_dict={'weight': torch.ones(2)},
                 critic_state_dict={'weight': torch.ones(1)}, optimizer_state_dict={'moments': torch.ones(2)})
    state = deepcopy(prior)
    state['infos']['settings']['gap_entry_encoder_version'] = 'periodic-chart-v1'
    state['infos']['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = 'bound-by-factory'
    state['infos']['entry_encoder_migration'] = dict(schema='periodic-entry-migration-v1',
        new_PPO=0, learning_clock=1501)
    state['infos']['hardware_ready'] = False
    return state, prior


def test_zero_update_contract_preserves_the_full_checkpoint():
    state, prior = migration_states()
    validate_periodic_entry_checkpoint(state, prior)


@pytest.mark.parametrize('tamper', ['actor', 'critic', 'optimizer', 'rng', 'clock', 'settings'])
def test_zero_update_contract_rejects_unrelated_changes(tamper):
    state, prior = migration_states()
    if tamper in ('actor', 'critic', 'optimizer'):
        key = tamper+'_state_dict'
        next(iter(state[key].values())).add_(1.)
    elif tamper == 'rng':
        state['infos']['rng'][0] += 1
    elif tamper == 'clock':
        state['infos']['completed_updates'] += 1
    else:
        state['infos']['settings']['leg_action_clip'] = 100
    with pytest.raises(ValueError):
        validate_periodic_entry_checkpoint(state, prior)
