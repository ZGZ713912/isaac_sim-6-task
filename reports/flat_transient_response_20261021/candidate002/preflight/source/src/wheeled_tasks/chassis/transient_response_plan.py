"""Next opt-in objective: asymmetric speed limits and rapid complete stops."""
from copy import deepcopy
from pathlib import Path

from wheeled_algo.transient_motion import ACTOR_SETTINGS
from .transient_response import REWARD_SETTINGS


SCHEMA = 'flat-base-finetune-plan-v20'


def response_envelope(cases):
    """Create new cases, preserving the historical command table byte for byte."""
    result = deepcopy(cases)
    for case in result:
        if case['name'] == 'backward_3':
            case['name'] = 'backward_2p5'
            case['command'][0] = -2.5
        elif case['name'] == 'start_stop_3':
            case['name'] = 'start_stop_3_to_2p5'
            case['skill'].update(segment_seconds=2.5, opposite_speed_ratio=5./6.)
    by_name = {case['name']: case for case in result}
    for name, source in [('yaw_start_stop_positive', 'rotate_4'),
                         ('yaw_start_stop_negative', 'rotate_4_reverse')]:
        case = deepcopy(by_name[source])
        case['name'] = name
        case['skill']['yaw_start_stop_segment_seconds'] = 2.5
        result.append(case)
    return result


def make_transient_plan(base, parent_file, parent_sha256, base_file, base_sha256):
    if base['schema'] != 'flat-base-finetune-plan-v19' or base['policy_dt'] != .005:
        raise ValueError('Transient pilot requires its frozen 200 Hz directional predecessor')
    plan = deepcopy(base)
    plan.update(schema=SCHEMA,
        training_scope='leg-assisted rapid launch/brake and bidirectional yaw response; reverse target 2.5m/s',
        transient_response=dict(checkpoint=parent_file, sha256=parent_sha256,
            base_plan_file=base_file, base_plan_sha256=base_sha256,
            actor=deepcopy(ACTOR_SETTINGS), reward=deepcopy(REWARD_SETTINGS)),
        first_block_updates=8, additional_critic_warmup_updates=2)
    source = {g['name']: g for g in base['train_groups']}
    groups = []
    # 35% standing/low-speed/steady retention, 65% complete transient cycles.
    for name, fraction in dict(stand_low=.05, stand_mid=.05, stand_high=.05,
            forward_low=.04, backward_low=.04, forward_edge3=.06, backward_edge3=.06,
            start_stop_3_0=.175, backward_start_stop3=.175,
            yaw_start_stop_positive=.15, yaw_start_stop_negative=.15).items():
        group = deepcopy(source[name])
        group['fraction'] = fraction
        group['name'] = {'backward_edge3': 'backward_edge2p5',
                         'backward_start_stop3': 'backward_start_stop2p5'}.get(name, name)
        skill = group['skill']
        if skill['command'][0] < -.75:
            skill['command'][0] = -2.5
            if 'amplitude_range' in skill:
                skill['amplitude_range'][1] = 1.
        if skill['kind'] == 'start_stop':
            skill['segment_seconds'] = 2.5
            # Forward 3 -> stop -> reverse 2.5 -> stop, or the opposite order.
            skill['opposite_speed_ratio'] = 5. / 6. if skill['command'][0] > 0. else 6. / 5.
        elif 'yaw_start_stop_segment_seconds' in skill:
            skill['yaw_start_stop_segment_seconds'] = 2.5
        groups.append(group)
    plan['train_groups'] = groups
    plan['group_episode_seconds'] = {g['name']: 10. for g in groups if
        g['skill']['kind'] == 'start_stop' or 'yaw_start_stop_segment_seconds' in g['skill']}
    # New requirements are prospective; historical -3m/s reports stay intact.
    plan['response_targets'] = dict(forward_max_m_s=3., reverse_max_m_s=2.5,
        yaw_max_rad_s=4., transition_sample_fraction=.65,
        selection_metrics=['response_time_s', 'stop_distance_m', 'overshoot_m_s',
            'settled_vx_m_s', 'settled_yaw_rad_s', 'slip_m_s', 'pitch_deg',
            'leg_excursion_rad', 'leg_speed_rad_s', 'height_error_m', 'full_episode_fraction'],
        historical_minus3_requirement='superseded for future objectives; never regrade old evidence')
    return plan


def validate_transient_plan(root, plan):
    from .flat_base_continuation import digest, read_plan
    root = Path(root)
    spec = plan.get('transient_response', {})
    required = {'checkpoint', 'sha256', 'base_plan_file', 'base_plan_sha256', 'actor', 'reward'}
    if set(spec) != required or digest(root / spec['checkpoint']) != spec['sha256']:
        raise ValueError('Transient parent is not authenticated')
    if digest(root / spec['base_plan_file']) != spec['base_plan_sha256']:
        raise ValueError('Transient base plan changed')
    base = read_plan(root, spec['base_plan_file'])
    expected = make_transient_plan(base, spec['checkpoint'], spec['sha256'],
                                  spec['base_plan_file'], spec['base_plan_sha256'])
    if plan != expected:
        raise ValueError('Transient pilot changed an undeclared setting')
    return dict(policy_hz=200, physics_hz=1000, transient_fraction=.65,
                reverse_target_m_s=2.5, optimizer_updates=8)
