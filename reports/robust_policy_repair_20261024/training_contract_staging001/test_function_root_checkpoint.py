"""Real checkpoint contracts must reject silent function/state corruption."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)
PATH = Path(__file__).with_name('function_root_checkpoint.py')
spec = importlib.util.spec_from_file_location('wheeled_algo._staged_function_root_checkpoint', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/robust_policy_repair_20261024/motion_damping080002/model_probe.pt'


@pytest.fixture(scope='module')
def original():
    return torch.load(SOURCE, map_location='cpu', weights_only=False)


def initial(original):
    return module.initialize_state(SOURCE, original, module.digest(PATH))


def test_initialization_is_exact_except_explicit_optimizer_append(original):
    state = initial(original)
    module.validate_state(state, original, original)
    for key in ('actor_state_dict', 'critic_state_dict'):
        assert module.exact_state_equal(state[key], original[key])
    for key in ('environment_rng', 'domain_rng', 'torch_rng', 'cuda_rng', 'completed_updates'):
        assert module.exact_state_equal(state['infos'][key], original['infos'][key])
    assert len(state['optimizer_state_dict']['param_groups'][0]['params']) == 34
    assert not set(range(28, 34)) & state['optimizer_state_dict']['state'].keys()


@pytest.mark.parametrize('mutation', ('actor', 'critic', 'optimizer', 'rng', 'clock',
    'alpha', 'encoder_version', 'observation', 'bounds', 'unknown_setting', 'module_hash', 'hardware'))
def test_initialization_rejects_unrelated_changes(original, mutation):
    state = initial(original)
    info = state['infos']
    if mutation == 'actor':
        state['actor_state_dict']['mlp.mean.entry_recovery.4.bias'][0] += .01
    elif mutation == 'critic':
        next(iter(state['critic_state_dict'].values())).add_(.01)
    elif mutation == 'optimizer':
        next(iter(state['optimizer_state_dict']['state'].values()))['exp_avg'].add_(.01)
    elif mutation == 'rng':
        info['torch_rng'][0] ^= 1
    elif mutation == 'clock':
        info['completed_updates'] += 1
    elif mutation == 'alpha':
        info['settings']['motion_leg_damping_alpha'] = .7
    elif mutation == 'encoder_version':
        info['settings']['gap_entry_encoder_version'] = 'another-function'
    elif mutation == 'observation':
        info['settings']['standing_position_observation']['scale_per_m'] = 19.
    elif mutation == 'bounds':
        info['settings']['leg_action_clip'] = 18.
    elif mutation == 'unknown_setting':
        info['settings']['unreviewed_flag'] = True
    elif mutation == 'module_hash':
        info['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = 'bad'
    else:
        info['hardware_ready'] = True
    with pytest.raises(ValueError):
        module.validate_state(state, original, original)


def learned(original, tmp_path, count=1):
    source = initial(original)
    path = tmp_path / 'initial.pt'
    torch.save(source, path)
    result = deepcopy(source)
    info = result['infos']
    info['function_root_stage'] = 'native-training'
    info['completed_updates'] += count
    if count:
        result['actor_state_dict']['mlp.mean.entry_recovery.4.bias'].add_(.0001)
        next(iter(result['critic_state_dict'].values())).add_(.0001)
        entry_keys = [module.PREFIX + str(layer) + suffix for layer in (0, 2, 4)
                      for suffix in ('.weight', '.bias')]
        for index, key in enumerate(entry_keys, 28):
            value = result['actor_state_dict'][key]
            result['optimizer_state_dict']['state'][index] = dict(step=torch.tensor(12.),
                exp_avg=torch.zeros_like(value), exp_avg_sq=torch.zeros_like(value))
    info['function_root_learning'] = module.make_progress(path, source, result['actor_state_dict'], count)
    return source, result


def test_learning_allows_only_existing_entry_head_and_critic(original, tmp_path):
    source, result = learned(original, tmp_path)
    module.validate_state(result, original, source)


@pytest.mark.parametrize('mutation', ('frozen_actor', 'std', 'digest', 'lost_history',
    'bad_entry_shape', 'nonfinite_Adam', 'nonfinite_critic', 'optimizer_order', 'clock', 'ancestry'))
def test_learned_contract_rejects_corruption(original, tmp_path, mutation):
    source, state = learned(original, tmp_path)
    info = state['infos']
    if mutation == 'frozen_actor':
        state['actor_state_dict']['mlp.mean.recovery.4.bias'].add_(.01)
    elif mutation == 'std':
        state['actor_state_dict']['mlp.log_std'].add_(.01)
    elif mutation == 'digest':
        state['actor_state_dict']['mlp.mean.entry_recovery.4.bias'].add_(.01)
    elif mutation == 'lost_history':
        state['optimizer_state_dict']['state'].pop(28)
    elif mutation == 'bad_entry_shape':
        state['optimizer_state_dict']['state'][28]['exp_avg'] = torch.zeros(1)
    elif mutation == 'nonfinite_Adam':
        state['optimizer_state_dict']['state'][28]['exp_avg'].fill_(float('nan'))
    elif mutation == 'nonfinite_critic':
        next(iter(state['critic_state_dict'].values())).fill_(float('inf'))
    elif mutation == 'optimizer_order':
        state['optimizer_state_dict']['param_groups'][0]['params'].reverse()
    elif mutation == 'clock':
        info['function_root_learning']['new_optimizer_updates'] = True
    else:
        info['training_function_root']['sha256'] = 'bad'
    with pytest.raises(ValueError):
        module.validate_state(state, original, source)


def test_zero_native_updates_allow_reset_rng_but_no_weight_updates(original, tmp_path):
    source, state = learned(original, tmp_path, count=0)
    state['infos']['environment_rng'][0] ^= 1
    module.validate_state(state, original, source)
    state['critic_state_dict'][next(iter(state['critic_state_dict']))].add_(.01)
    with pytest.raises(ValueError):
        module.validate_state(state, original, source)
