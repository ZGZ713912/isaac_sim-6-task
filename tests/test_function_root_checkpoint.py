"""Learned checkpoint provenance keeps old neutral migrations strict."""
from copy import deepcopy

import pytest
import torch

from wheeled_algo import function_root_checkpoint as contract


@pytest.fixture
def root(tmp_path):
    actor = {'mlp.log_std': torch.zeros(6), 'mlp.mean.recovery.0.weight': torch.ones(3, 2)}
    for layer in (0, 2, 4):
        actor[contract.PREFIX + str(layer) + '.weight'] = torch.ones(3, 2)
        actor[contract.PREFIX + str(layer) + '.bias'] = torch.zeros(3)
    optimizer = dict(state={0: dict(step=torch.tensor(3.), exp_avg=torch.zeros(6), exp_avg_sq=torch.ones(6)),
        2: dict(step=torch.tensor(3.), exp_avg=torch.zeros(2), exp_avg_sq=torch.ones(2))},
        param_groups=[dict(name='actor', params=[0, 1], lr=.0001), dict(name='critic', params=[2], lr=.001)])
    state = dict(actor_state_dict=actor, critic_state_dict={'value.weight': torch.zeros(2)},
        optimizer_state_dict=optimizer, iter=100, infos=dict(schema='robust-getup-checkpoint-v2',
        completed_updates=100, base_checkpoint='parent.pt', base_checkpoint_sha256='parent',
        bank_sha256='bank', actor_variant='robust', hardware_ready=False,
        source_sha256={'src/wheeled_algo/periodic_entry_actor.py': 'periodic-source'},
        settings=dict(gap_entry_encoder_version='periodic-chart-v1', gap_entry_recovery_split_version='frozen-v1',
                      motion_leg_damping_version='motion-leg-damping-v1', motion_leg_damping_alpha=.8),
        torch_rng=torch.arange(8, dtype=torch.uint8)))
    path = tmp_path / 'root.pt'
    torch.save(state, path)
    initial = contract.initialize_state(path, state, 'module-source')
    return path, state, initial


def trained(root, tmp_path, count=1):
    _, _, initial = root
    path = tmp_path / 'initial.pt'
    torch.save(initial, path)
    result = deepcopy(initial)
    info = result['infos']
    info.update(function_root_stage='native-training', completed_updates=100 + count)
    if count:
        result['actor_state_dict'][contract.PREFIX + '4.bias'].add_(.01)
        result['critic_state_dict']['value.weight'].add_(.01)
        keys = [contract.PREFIX + str(layer) + suffix for layer in (0, 2, 4) for suffix in ('.weight', '.bias')]
        for i, key in enumerate(keys, 2):
            result['optimizer_state_dict']['state'][i] = dict(step=torch.tensor(12.),
                exp_avg=torch.zeros_like(result['actor_state_dict'][key]),
                exp_avg_sq=torch.zeros_like(result['actor_state_dict'][key]))
    info['function_root_learning'] = contract.make_progress(path, initial, result['actor_state_dict'], count)
    return result


def test_neutral_initialization_preserves_policy_rng_and_learning_clock(root):
    _, original, initial = root
    contract.validate_state(initial, original, original)
    assert contract.exact_state_equal(original['actor_state_dict'], initial['actor_state_dict'])
    assert torch.equal(original['infos']['torch_rng'], initial['infos']['torch_rng'])
    assert initial['infos']['completed_updates'] == 100
    assert not set(range(2, 8)) & initial['optimizer_state_dict']['state'].keys()
    assert torch.equal(initial['optimizer_state_dict']['state'][8]['exp_avg'],
                       original['optimizer_state_dict']['state'][2]['exp_avg'])


def test_learned_entry_loads_without_relaxing_neutral_contract(root, tmp_path):
    _, original, initial = root
    learned = trained(root, tmp_path)
    contract.validate_state(learned, original, initial)
    learned['infos']['function_root_stage'] = 'optimizer-initialization'
    with pytest.raises(ValueError):
        contract.validate_state(learned, original, initial)


@pytest.mark.parametrize('kind', ['std', 'fallen', 'entry_digest', 'settings', 'missing_Adam',
    'Adam_shape', 'Adam_nonfinite', 'critic_nonfinite', 'order', 'clock', 'function_source', 'root'])
def test_learning_rejects_mutated_contract(root, tmp_path, kind):
    _, original, initial = root
    learned = trained(root, tmp_path)
    if kind == 'std':
        learned['actor_state_dict']['mlp.log_std'].add_(.01)
    elif kind == 'fallen':
        learned['actor_state_dict']['mlp.mean.recovery.0.weight'].add_(.01)
    elif kind == 'entry_digest':
        learned['actor_state_dict'][contract.PREFIX + '0.bias'].add_(.01)
    elif kind == 'settings':
        learned['infos']['settings']['motion_leg_damping_alpha'] = .7
    elif kind == 'missing_Adam':
        learned['optimizer_state_dict']['state'].pop(2)
    elif kind == 'Adam_shape':
        learned['optimizer_state_dict']['state'][2]['exp_avg'] = torch.zeros(1)
    elif kind == 'Adam_nonfinite':
        learned['optimizer_state_dict']['state'][2]['exp_avg'].fill_(float('nan'))
    elif kind == 'critic_nonfinite':
        learned['critic_state_dict']['value.weight'].fill_(float('nan'))
    elif kind == 'order':
        learned['optimizer_state_dict']['param_groups'][0]['params'].reverse()
    elif kind == 'clock':
        learned['infos']['function_root_learning']['new_optimizer_updates'] = True
    elif kind == 'function_source':
        learned['infos']['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = 'wrong'
    else:
        learned['infos']['training_function_root']['sha256'] = 'wrong'
    with pytest.raises(ValueError):
        contract.validate_state(learned, original, initial)


def test_zero_native_updates_allow_only_reset_rng_and_bookkeeping(root, tmp_path):
    _, original, initial = root
    state = trained(root, tmp_path, count=0)
    state['infos']['torch_rng'][0] ^= 1
    contract.validate_state(state, original, initial)
    state['critic_state_dict']['value.weight'].add_(.01)
    with pytest.raises(ValueError):
        contract.validate_state(state, original, initial)


def test_exact_reviewed_zero_reward_environment_revision_is_explicit(root, tmp_path):
    _, original, initial = root
    before, after = contract.NEUTRAL_ENVIRONMENT_REVISION
    for state in (original, initial):
        state['infos']['source_sha256'][contract.ENVIRONMENT_MODULE] = before
    state = trained(root, tmp_path)
    state['infos']['source_sha256'][contract.ENVIRONMENT_MODULE] = after
    with pytest.raises(ValueError):
        contract.validate_state(state, original, initial)
    state['infos']['function_root_environment_revision'] = contract.environment_revision(
        original['infos']['source_sha256'], state['infos']['source_sha256'], state['infos']['settings'])
    contract.validate_state(state, original, initial)


@pytest.mark.parametrize('kind', ['unknown_environment', 'nonzero_reward', 'different_algorithm'])
def test_environment_revision_exception_does_not_allow_other_changes(root, tmp_path, kind):
    _, original, initial = root
    before, after = contract.NEUTRAL_ENVIRONMENT_REVISION
    for state in (original, initial):
        state['infos']['source_sha256'][contract.ENVIRONMENT_MODULE] = before
    state = trained(root, tmp_path)
    state['infos']['source_sha256'][contract.ENVIRONMENT_MODULE] = after
    state['infos']['function_root_environment_revision'] = contract.environment_revision(
        original['infos']['source_sha256'], state['infos']['source_sha256'], state['infos']['settings'])
    if kind == 'unknown_environment':
        state['infos']['source_sha256'][contract.ENVIRONMENT_MODULE] = 'unreviewed'
    elif kind == 'nonzero_reward':
        state['infos']['settings']['recovery_support_weight'] = 1.
    else:
        state['infos']['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = 'changed'
    with pytest.raises(ValueError):
        contract.validate_state(state, original, initial)
