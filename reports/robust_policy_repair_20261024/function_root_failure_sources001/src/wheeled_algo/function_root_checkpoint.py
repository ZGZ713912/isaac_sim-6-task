"""Entry-only learning on an authenticated, unchanged policy function root.

V2 migration validators stay zero-update validators. V3 learns the existing
entry tensors while rebuilding the runtime function from its validated V2 root.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from .entry_checkpoint import PREFIX, entry_digest
from .entry_optimizer import append_entry_optimizer_state
from .periodic_entry_actor import exact_state_equal

SCHEMA = 'robust-getup-checkpoint-v3'
VERSION = 'entry-function-root-v1'
MODULE = 'src/wheeled_algo/function_root_checkpoint.py'
# These fields own exposure/reward/learning schedules, never actor coordinates,
# observation meanings, gains, output bounds, or the runtime target function.
TRAINING_FIELDS = frozenset(('version', 'motion_push_fraction', 'motion_fraction',
    'recovery_family_weights', 'side_recovery_birth_focus', 'curriculum_updates',
    'origin_updates', 'recovery_support_weight', 'gap_entry_optimizer_layout'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_training_settings(settings, root_settings):
    if (settings.get('gap_entry_optimizer_layout') != 'append-v1'
            or settings.get('gap_entry_recovery_split_version') != 'frozen-v1'
            or {k: v for k, v in settings.items() if k not in TRAINING_FIELDS}
            != {k: v for k, v in root_settings.items() if k not in TRAINING_FIELDS}):
        raise ValueError('Entry training changed the authenticated runtime function or observation settings')


def _same_tensor_layout(actual, expected):
    return (actual.keys() == expected.keys()
            and all(isinstance(actual[k], torch.Tensor) and actual[k].shape == v.shape
                    and actual[k].dtype == v.dtype and torch.isfinite(actual[k]).all()
                    for k, v in expected.items()))


def make_progress(source_path, source, actor, count):
    if not _integer(count):
        raise ValueError('Require a nonnegative completed PPO increment')
    start = source['infos']['completed_updates']
    return dict(schema=VERSION, source_checkpoint=str(Path(source_path).resolve()),
        source_sha256=digest(source_path), source_learning_clock=start,
        new_optimizer_updates=count, completed_learning_clock=start + count,
        entry_state_sha256=entry_digest(actor), scope='six-existing-entry-tensors-only')


def initialize_state(source_path, original, module_sha256):
    """Zero-update optimizer append; all policy, critic and RNG bytes retained."""
    if (original['infos']['schema'] != 'robust-getup-checkpoint-v2'
            or original['infos']['settings'].get('gap_entry_encoder_version') != 'periodic-chart-v1'
            or original['infos']['settings'].get('gap_entry_optimizer_layout') is not None):
        raise ValueError('Initialize exactly once from a neutral periodic V2 checkpoint')
    result = deepcopy(original)
    result['optimizer_state_dict'] = append_entry_optimizer_state(original['optimizer_state_dict'], 6)
    info = result['infos']
    info.update(schema=SCHEMA, entry_recovery_only_tuning=True, hardware_ready=False)
    info['settings']['gap_entry_optimizer_layout'] = 'append-v1'
    info['source_sha256'][MODULE] = module_sha256
    info['training_function_root'] = dict(schema=VERSION,
        checkpoint=str(Path(source_path).resolve()), sha256=digest(source_path))
    info['function_root_learning'] = make_progress(source_path, original, result['actor_state_dict'], 0)
    info['function_root_stage'] = 'optimizer-initialization'
    return result


def validate_state(state, root, source):
    """The caller separately authenticates and loads root and immediate source."""
    info, prior = state['infos'], source['infos']
    root_info = root['infos']
    record = info.get('function_root_learning', {})
    origin = info.get('training_function_root', {})
    if (info.get('schema') != SCHEMA or origin.get('schema') != VERSION
            or record.get('schema') != VERSION or record.get('scope') != 'six-existing-entry-tensors-only'
            or info.get('entry_recovery_only_tuning') is not True
            or info.get('hardware_ready') is not False
            or root_info.get('schema') != 'robust-getup-checkpoint-v2'):
        raise ValueError('Require the explicit entry-only function-root contract')
    if prior.get('schema') == SCHEMA and not exact_state_equal(origin, prior.get('training_function_root')):
        raise ValueError('A learned checkpoint changed its function ancestry')
    count = record.get('new_optimizer_updates')
    start = prior['completed_updates']
    if (not _integer(start) or not _integer(count)
            or record.get('source_learning_clock') != start
            or record.get('completed_learning_clock') != start + count
            or info.get('completed_updates') != start + count):
        raise ValueError('Entry training clocks are inconsistent')
    validate_training_settings(info['settings'], root_info['settings'])
    for name, sha in root_info['source_sha256'].items():
        if name.startswith('src/') and info['source_sha256'].get(name) != sha:
            raise ValueError('Entry training changed a function or observation source')
    for key in ('base_checkpoint', 'base_checkpoint_sha256', 'bank_sha256', 'actor_variant'):
        if info.get(key) != root_info.get(key):
            raise ValueError('Entry training changed base, bank or actor identity')
    actor = state['actor_state_dict']
    if not _same_tensor_layout(actor, root['actor_state_dict']):
        raise ValueError('Actor tensor layout or finiteness differs')
    if any(not torch.equal(v, actor[k]) for k, v in root['actor_state_dict'].items() if not k.startswith(PREFIX)):
        raise ValueError('Entry training changed a frozen actor tensor or std')
    if entry_digest(actor) != record.get('entry_state_sha256'):
        raise ValueError('Learned entry tensors differ from the recorded digest')
    if not _same_tensor_layout(state['critic_state_dict'], root['critic_state_dict']):
        raise ValueError('Critic tensor layout or finiteness differs')
    expected_optimizer = append_entry_optimizer_state(root['optimizer_state_dict'], 6)
    optimizer = state['optimizer_state_dict']
    if optimizer.keys() != expected_optimizer.keys() or len(optimizer['param_groups']) != 2:
        raise ValueError('Optimizer structure changed')
    for current, expected in zip(optimizer['param_groups'], expected_optimizer['param_groups'], strict=True):
        if {k: v for k, v in current.items() if k != 'lr'} != {k: v for k, v in expected.items() if k != 'lr'}:
            raise ValueError('Optimizer parameter order or options changed')
        if not isinstance(current.get('lr'), (int, float)) or not 0. < current['lr'] <= .001:
            raise ValueError('Optimizer learning rate is outside the bounded contract')
    frozen_count = len(root['optimizer_state_dict']['param_groups'][0]['params'])
    if not exact_state_equal({k: v for k, v in optimizer['state'].items() if k < frozen_count},
                             {k: v for k, v in expected_optimizer['state'].items() if k < frozen_count}):
        raise ValueError('Frozen actor Adam moments changed')
    ids = {p for group in optimizer['param_groups'] for p in group['params']}
    if not set(optimizer['state']) <= ids:
        raise ValueError('Optimizer contains undeclared parameter slots')
    for slot in optimizer['state'].values():
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in slot.values()):
            raise ValueError('Nonfinite optimizer state')
    if not expected_optimizer['state'].keys() <= optimizer['state'].keys():
        raise ValueError('Optimizer lost existing parameter history')
    entry_keys = [PREFIX + str(layer) + suffix for layer in (0, 2, 4) for suffix in ('.weight', '.bias')]
    for index, key in enumerate(entry_keys, frozen_count):
        slot = optimizer['state'].get(index)
        if slot is None:
            if count:
                raise ValueError('A completed entry PPO checkpoint lacks its Adam history')
            continue
        if (set(slot) != {'step', 'exp_avg', 'exp_avg_sq'}
                or slot['step'].numel() != 1 or slot['step'].item() <= 0
                or any(slot[name].shape != actor[key].shape or slot[name].dtype != actor[key].dtype
                       for name in ('exp_avg', 'exp_avg_sq'))):
            raise ValueError('Entry Adam history has an invalid tensor layout')
    for key in expected_optimizer['state']:
        if key >= frozen_count + 6:
            current, expected = optimizer['state'][key], expected_optimizer['state'][key]
            if (current.keys() != expected.keys()
                    or any(isinstance(v, torch.Tensor) and
                           (not isinstance(current[name], torch.Tensor) or current[name].shape != v.shape
                            or current[name].dtype != v.dtype) for name, v in expected.items())):
                raise ValueError('Critic Adam history changed tensor layout')
    if info.get('function_root_stage') == 'optimizer-initialization':
        expected = initialize_state(origin['checkpoint'], root, info['source_sha256'][MODULE])
        if not exact_state_equal(state, expected):
            raise ValueError('Optimizer initialization changed unrelated weights, state or metadata')
    elif info.get('function_root_stage') == 'native-training':
        if prior.get('schema') != SCHEMA:
            raise ValueError('Native training must start from an authenticated optimizer initialization')
        if count == 0:
            if (not exact_state_equal(actor, source['actor_state_dict'])
                    or not exact_state_equal(state['critic_state_dict'], source['critic_state_dict'])):
                raise ValueError('Zero completed PPO updates changed actor or critic')
            before = deepcopy(source['optimizer_state_dict'])
            for a, b in zip(before['param_groups'], optimizer['param_groups'], strict=True):
                a['lr'] = b['lr']
            if not exact_state_equal(before, optimizer):
                raise ValueError('Zero completed PPO updates changed Adam state')
    else:
        raise ValueError('Unknown function-root checkpoint stage')
