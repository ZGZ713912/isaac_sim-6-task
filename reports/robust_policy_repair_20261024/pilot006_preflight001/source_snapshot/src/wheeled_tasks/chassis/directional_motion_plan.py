"""Authenticated 200 Hz motion pilot; retain the previous time/material contract."""
from copy import deepcopy
import json
from pathlib import Path

KIND = 'original-static-grip-with-directional-motion-v12'
SCHEMA = 'flat-base-finetune-plan-v19'


def make_directional_plan(root, base, parent_file, parent_sha256, base_file, base_sha256):
    if base['schema'] != 'flat-base-finetune-plan-v18' or base['policy_dt'] != .005:
        raise ValueError('Directional motion starts from the verified 200 Hz plan')
    result = deepcopy(base)
    result.update(schema=SCHEMA, actor_adaptation=KIND, learning_rate=1e-4,
        checkpoint_interval=1, additional_critic_warmup_updates=2, first_block_updates=8,
        training_scope='200Hz direction-specific bounded high-speed and moving-stop banks; all old heads/std frozen',
        directional_motion=dict(checkpoint=parent_file, sha256=parent_sha256,
            base_plan_file=base_file, base_plan_sha256=base_sha256),
        motion_trust_region=dict(max_kl=.02, max_backtracks=12))
    catalogue = {g['name']: g for g in base['train_groups']}
    fractions = dict(stand_mid=.03, stand_low=.02, stand_high=.02,
        forward_low=.015, backward_low=.015, forward_high=.10, backward_high=.15,
        start_stop_3_0=.05, turn_left=.075, turn_right=.075)
    groups = []
    for name, fraction in fractions.items():
        group = deepcopy(catalogue[name])
        group['fraction'] = fraction
        if group['skill']['kind'] == 'start_stop':
            group['skill']['segment_seconds'] = 3.
        groups.append(group)
    for name, source, command, fraction in (
            ('forward_edge3', 'forward_high', [3., 0., .305], .10),
            ('backward_edge3', 'backward_high', [-3., 0., .305], .10),
            ('backward_start_stop3', 'start_stop_3_0', [-3., 0., .305], .10),
            ('yaw_start_stop_positive', 'turn_left', [0., 4., .305], .075),
            ('yaw_start_stop_negative', 'turn_right', [0., -4., .305], .075)):
        group = deepcopy(catalogue[source])
        group.update(name=name, fraction=fraction)
        group['skill'].update(command=command, sample_amplitude=False)
        group['skill'].pop('amplitude_range', None)
        if group['skill']['kind'] == 'start_stop':
            group['skill']['segment_seconds'] = 3.
        elif group['skill']['kind'] == 'rotate':
            group['skill']['yaw_start_stop_segment_seconds'] = 3.
        groups.append(group)
    result['train_groups'] = groups
    result['group_episode_seconds'] = {g['name']: 12. for g in groups
        if g['skill']['kind'] == 'start_stop' or 'yaw_start_stop_segment_seconds' in g['skill']}
    return result


def validate_directional_plan(root, plan):
    from .flat_base_continuation import read_plan, digest
    root = Path(root)
    spec = plan.get('directional_motion', {})
    required = {'checkpoint', 'sha256', 'base_plan_file', 'base_plan_sha256'}
    if set(spec) != required or digest(root / spec['checkpoint']) != spec['sha256']:
        raise ValueError('Directional motion parent is not authenticated')
    base = read_plan(root, spec['base_plan_file'])
    if digest(root / spec['base_plan_file']) != spec['base_plan_sha256']:
        raise ValueError('Directional motion base plan changed')
    expected = make_directional_plan(root, base, spec['checkpoint'], spec['sha256'],
                                     spec['base_plan_file'], spec['base_plan_sha256'])
    if plan != expected:
        raise ValueError('Directional pilot changed an undeclared time/material/objective setting')
    return dict(policy_hz=200, physics_hz=1000, new_actor_parameters=1656,
                initial_block_optimizer_updates=8, max_direction_group_kl=.02)
