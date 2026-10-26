from copy import deepcopy
import ast
import inspect
import textwrap

import pytest
import torch
from torch import nn

from wheeled_algo import robust_getup, quiet_output_actor
from wheeled_algo.motion_target_damping import MotionTargetDampedGapActor
from wheeled_algo.periodic_entry_actor import PeriodicEntryGapActor, exact_state_equal
from wheeled_algo.quiet_stance_authority import (
    MODULE, SCHEMA, VERSION, QuietStanceAuthorityActor, attach_stance_authority,
    stance_quiet_gate, validate_checkpoint,
)


def actor():
    return MotionTargetDampedGapActor(nn.Linear(35, 6), motion_leg_alpha=.8,
        upright_entry_recovery=True, canonical_recovery=True, smooth_knee_targets=True,
        recovery_leg_step=.3, final_target_winding_version='paired-v1',
        cyclic_leg_targets=True, leg_action_clip=16., stance_aware_recovery=True,
        recovery_shape_aware=True, quiet_output_wheel_alpha=.25)


def samples():
    torch.manual_seed(1229)
    obs = torch.randn(16, 35)*.05
    obs[:, 3] = 1.525
    obs[:, 9] = -1.
    obs[:, 0:3] = 0.
    obs[8:, 4] = .8
    return obs


def test_zero_strength_preserves_full_actor_and_state_exactly():
    source = actor()
    before = deepcopy(source.state_dict())
    candidate = attach_stance_authority(deepcopy(source), 0.)
    assert torch.equal(candidate(samples()), source(samples()))
    assert exact_state_equal(before, candidate.state_dict())
    assert [n for n, _ in candidate.named_parameters()] == [n for n, _ in source.named_parameters()]


def test_active_gate_retains_half_authority_and_shared_gate_is_unchanged():
    obs = samples()
    original = robust_getup.quiet_gate
    gated = stance_quiet_gate(obs, .5)
    assert torch.equal(gated[:8], original(obs[:8]))
    torch.testing.assert_close(gated[8:], torch.full((8, 1), .5), rtol=0., atol=0.)
    candidate = attach_stance_authority(actor(), .5)
    candidate(obs)
    assert robust_getup.quiet_gate is original
    assert quiet_output_actor.quiet_gate is original


@pytest.mark.parametrize('mode', ['request', 'tilt', 'entry', 'fallen'])
def test_no_additional_authority_in_excluded_states(mode):
    source = actor()
    candidate = attach_stance_authority(deepcopy(source), .5)
    obs = samples()
    if mode == 'request':
        obs[:, 0] = .5
    elif mode == 'tilt':
        obs[:, 9] = -.9
    elif mode == 'entry':
        obs[:, 10:14] = torch.tensor([-2.5, -2.5, 2.5, 2.5])
    else:
        obs[:, 9] = 1.
    assert torch.equal(stance_quiet_gate(obs, .5), robust_getup.quiet_gate(obs))
    assert torch.equal(source(obs), candidate(obs))


@pytest.mark.parametrize('floor', [True, -.1, 1.1, float('nan'), float('inf')])
def test_invalid_authority_is_rejected(floor):
    with pytest.raises(ValueError, match='authority'):
        attach_stance_authority(actor(), floor)


def test_actual_active_actor_has_finite_feedback_gradients():
    candidate = attach_stance_authority(actor(), .5)
    obs = samples().requires_grad_()
    candidate(obs).square().sum().backward()
    assert torch.isfinite(obs.grad).all()
    assert obs.grad.abs().sum() > 0.


@pytest.mark.parametrize('old, name', [
    (robust_getup.RobustGetupActor.forward, '_robust_forward'),
    (quiet_output_actor.QuietOutputGetupActor.forward, '_quiet_forward'),
    (PeriodicEntryGapActor.forward, '_periodic_forward'),
    (MotionTargetDampedGapActor.forward, 'forward'),
])
def test_static_forward_changes_are_limited_to_gate_and_local_calls(old, name):
    source = textwrap.dedent(inspect.getsource(old))
    source = source.replace('def forward(', 'def '+name+'(')
    if name in ('_robust_forward', '_quiet_forward'):
        source = source.replace('quiet_gate(obs)', 'self.quiet_stance_gate(obs)')
    if name == '_quiet_forward':
        source = source.replace('super().forward(obs)', 'self._robust_forward(obs)')
    elif name == '_periodic_forward':
        source = source.replace('QuietOutputGetupActor.forward(self, obs)', 'self._quiet_forward(obs)')
    elif name == 'forward':
        source = source.replace('super().forward(obs)', 'self._periodic_forward(obs)')
    implemented = textwrap.dedent(inspect.getsource(getattr(QuietStanceAuthorityActor, name)))
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(implemented))


def states():
    source = dict(infos=dict(schema='robust-getup-checkpoint-v2', completed_updates=1501,
        settings={}, source_sha256={}, rng=torch.arange(4)),
        actor_state_dict={'weight': torch.ones(2)}, critic_state_dict={'weight': torch.ones(1)},
        optimizer_state_dict={'moments': torch.ones(2)})
    candidate = deepcopy(source)
    info = candidate['infos']
    info['schema'] = SCHEMA
    info['settings'].update(quiet_stance_authority_version=VERSION,
        quiet_stance_retained_authority=.5, quiet_stance_recovery_threshold=.05)
    info['source_sha256'][MODULE] = 'factory-authenticated'
    info['quiet_stance_authority_migration'] = dict(schema=VERSION, new_PPO=0, learning_clock=1501)
    info['hardware_ready'] = False
    return candidate, source


def test_zero_update_contract_preserves_all_saved_state():
    validate_checkpoint(*states())


@pytest.mark.parametrize('tamper', ['actor', 'critic', 'optimizer', 'rng', 'clock', 'settings', 'threshold', 'schema'])
def test_contract_rejects_unrelated_or_training_changes(tamper):
    candidate, source = states()
    info = candidate['infos']
    if tamper in ('actor', 'critic', 'optimizer'):
        next(iter(candidate[tamper+'_state_dict'].values())).add_(1.)
    elif tamper == 'rng':
        info['rng'][0] += 1
    elif tamper == 'clock':
        info['completed_updates'] += 1
    elif tamper == 'settings':
        info['settings']['unrelated'] = True
    elif tamper == 'threshold':
        info['settings']['quiet_stance_recovery_threshold'] = .1
    else:
        info['schema'] = 'robust-getup-checkpoint-v2'
    with pytest.raises(ValueError):
        validate_checkpoint(candidate, source)
