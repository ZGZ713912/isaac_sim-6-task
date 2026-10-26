"""V8 retains exact initialization and rejects changes outside quiet MLPs."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.quiet_full_checkpoint import LEARNED_KEYS, validate_state

BASE = Path(__file__).resolve().parents[1] / 'runs/robust_policy_repair_20261024'


@pytest.fixture(scope='module')
def snapshots():
    initial = BASE / 'quiet_full_root001/model_initial.pt'
    root = BASE / 'quiet_stance050003/model_probe.pt'
    if not initial.is_file() or not root.is_file():
        pytest.skip('Local authenticated policy fixture is unavailable')
    a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in (initial, root)]
    return a, b, a['infos']['policy_function_root']['optimizer_actor_parameter_names']


def test_initialization_retains_actor_critic_optimizer_and_rng(snapshots):
    initial, root, names = snapshots
    validate_state(initial, root, root, names)
    assert exact_state_equal({k: v for k, v in initial.items() if k != 'infos'},
                             {k: v for k, v in root.items() if k != 'infos'})
    assert len(LEARNED_KEYS) == 12
    assert sum(initial['actor_state_dict'][k].numel() for k in LEARNED_KEYS) == 8082


def test_source_and_initial_mean_are_bit_exact(snapshots):
    from export_robust_getup import load_mean
    before, _ = load_mean(BASE / 'quiet_stance050003/model_probe.pt')
    after, _ = load_mean(BASE / 'quiet_full_root001/model_initial.pt')
    with torch.inference_mode(), torch.random.fork_rng():
        torch.manual_seed(241519)
        obs = torch.randn(256, 35)
        assert torch.equal(before(obs), after(obs))


@pytest.mark.parametrize('kind', ['exploration', 'frozen_Adam', 'runtime_settings', 'nonfinite_quiet'])
def test_invalid_training_scope_is_rejected(snapshots, kind):
    initial, root, names = snapshots
    bad = deepcopy(initial)
    if kind == 'exploration':
        bad['actor_state_dict']['mlp.log_std'] += .001
    elif kind == 'frozen_Adam':
        key = bad['optimizer_state_dict']['param_groups'][0]['params'][names.index('mlp.log_std')]
        bad['optimizer_state_dict']['state'][key]['exp_avg'] += .001
    elif kind == 'runtime_settings':
        bad['infos']['settings']['quiet_leg_alpha'] = .99
    else:
        bad['actor_state_dict'][LEARNED_KEYS[0]].flatten()[0] = float('nan')
    with pytest.raises(ValueError):
        validate_state(bad, root, root, names)
