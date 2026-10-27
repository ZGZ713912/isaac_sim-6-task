"""Unadopted reset catalogue for a fresh, continuously mixed six-motor policy.

Uses only validated pose-bank entries. Commands and reset headings are data,
not a controller. This module does not touch a simulator, model, global RNG,
live learner, or held-out evaluation specification.
"""
from collections import Counter
from copy import deepcopy
import math
import random


WEIGHTS = dict(standing=25, initial_recovery=20, low_speed_straight=10,
    high_speed_straight=8, stationary_spin=8, four_direction_arcs=8,
    standing_height_change=5, moving_height_change=4, spinning_height_change=4,
    high_speed_braking=4, moving_fall_recovery=4)
HEIGHTS = (.23, .305, .42)
FAMILIES = ('front', 'back', 'left', 'right', 'inverted', 'compound')


def allocate(count):
    """Deterministic largest remainders; every final task remains populated."""
    if type(count) is not int or not 128 <= count <= 4096:
        raise ValueError('Use128..4096 native training environments')
    exact = {name: count * weight / 100 for name, weight in WEIGHTS.items()}
    quota = {name: math.floor(value) for name, value in exact.items()}
    ranked = sorted(WEIGHTS, key=lambda name: -(exact[name] - quota[name]))
    for name in ranked[:count - sum(quota.values())]:
        quota[name] += 1
    if min(quota.values()) < 1 or sum(quota.values()) != count:
        raise ValueError('A permanent training task would be absent')
    return quota


class TrainingCases:
    def __init__(self, bank, count, seed, *, foundation_updates=192, full_tasks_at=768):
        if (type(seed) is not int or type(foundation_updates) is not int or foundation_updates < 1
                or type(full_tasks_at) is not int or full_tasks_at <= foundation_updates):
            raise ValueError('Declare independent integer seed and fresh-update clocks')
        self.bank, self.count, self.seed = deepcopy(bank), count, seed
        self.foundation_updates, self.full_tasks_at = foundation_updates, full_tasks_at
        self.quota = allocate(count)
        self.roles = [(name, ordinal) for name, number in self.quota.items() for ordinal in range(number)]
        mapping = bank['diagnostic_height_starts']['mapping']
        self.height_pose = dict(zip(HEIGHTS, (mapping[key] for key in ('stand_low', 'stand', 'stand_high')), strict=True))
        self.recovery_pool = {}
        for family in FAMILIES:
            for asymmetric in (False, True):
                pool = [i for i, pose in enumerate(bank['poses']) if pose['family'] == family
                    and ('50_85' in pose['leg_shape'] or '85_50' in pose['leg_shape']) == asymmetric
                    and max(abs(pose['roll_deg']), abs(pose['pitch_deg'])) ==
                        (180 if family == 'inverted' else 65 if family == 'compound' else 90)]
                if not pool:
                    raise ValueError('Validated recovery family/leg geometry is absent: ' + family)
                self.recovery_pool[family, asymmetric] = pool

    def exposure(self, completed_updates):
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError('Use completed fresh actor updates')
        return min(1., max(0., (completed_updates-self.foundation_updates) /
                              (self.full_tasks_at-self.foundation_updates)))

    def case(self, lane, reset_number, completed_updates):
        if (type(lane) is not int or not 0 <= lane < self.count
                or type(reset_number) is not int or reset_number < 0):
            raise ValueError('Need a valid lane and its actual reset counter')
        role, ordinal = self.roles[lane]
        exposure = self.exposure(completed_updates)
        # Curriculum changes only the next episode's requested task. The final
        # standing25% remains present while all other skills are rehearsed.
        active = role == 'standing' or exposure >= (ordinal+1) / self.quota[role]
        task = role if active else 'standing'
        rng = random.Random(self.seed + lane*1_000_003 + reset_number*10_000_019)
        variant = ordinal + reset_number
        height = HEIGHTS[variant % 3]
        case = dict(training_group=task, final_training_group=role, lane=lane,
            reset_number=reset_number, pose_index=self.height_pose[height], recovery=False,
            command=[0., 0., height], duration_seconds=30.,
            heading_rad=rng.uniform(-math.pi, math.pi), pitch_deg=rng.uniform(-.5, .5),
            state_writes='episode_reset_only', sequence=[])
        sign = -1. if variant % 2 else 1.
        high_speed = 3. if sign > 0 else -2.5
        yaw = sign * (80. if (variant//2) % 2 else 100.) * 2.*math.pi/60.
        if task == 'initial_recovery':
            family = FAMILIES[(variant//2) % len(FAMILIES)]
            asymmetric = bool(variant % 2)
            case.update(pose_index=rng.choice(self.recovery_pool[family, asymmetric]),
                recovery=True, pitch_deg=0., requested_spawn_family=family,
                asymmetric=asymmetric)
            # Alternate a stable stand and a continued nonzero motion request.
            if (variant//12) % 2:
                case['command'][:2] = [rng.uniform(.35, .7)*sign, rng.uniform(.5, 1.)*-sign]
        elif task == 'low_speed_straight':
            case['command'][0] = sign * (.1 if (variant//2) % 2 else .3)
        elif task == 'high_speed_straight':
            case['command'][0] = high_speed
            case['duration_seconds'] = 18.
        elif task == 'stationary_spin':
            case['command'][1] = yaw
            case['requested_rpm'] = yaw*60./(2.*math.pi)
        elif task == 'four_direction_arcs':
            forward = (variant//2) % 2 == 0
            vx = (1. if forward else -.75) if (variant//4) % 2 == 0 else (3. if forward else -2.5)
            wz = sign * (1. if abs(vx) <= 1. else .5)
            case['command'][:2] = [vx, wz]
            # Actual turn reversals and translation reversals, with independent
            # training times rather than copied held-out suite boundaries.
            case['sequence'] = [[rng.uniform(4., 6.), [vx, -wz, height]],
                [rng.uniform(10., 12.), [-vx, wz, height]],
                [rng.uniform(18., 20.), [0., 0., height]]]
        elif task in ('standing_height_change', 'moving_height_change', 'spinning_height_change'):
            vx = sign*rng.uniform(.65, 1.15) if task == 'moving_height_change' else 0.
            wz = (-.8, 0., .8)[variant % 3] if task == 'moving_height_change' else yaw if task == 'spinning_height_change' else 0.
            case['command'][:2] = [vx, wz]
            targets = [HEIGHTS[(variant+offset) % 3] for offset in (1, 2, 0)]
            case['sequence'] = [[rng.uniform(a, b), [vx, wz, target]]
                for (a, b), target in zip(((4., 6.), (10., 12.), (18., 20.)), targets, strict=True)]
        elif task == 'high_speed_braking':
            case['command'][0] = high_speed
            case['sequence'] = [[rng.uniform(4.5, 6.5), [0., 0., height]]]
        elif task == 'moving_fall_recovery':
            motions = ((.7, 0.), (-.7, 0.), (.7, 1.), (.7, -1.),
                (0., 100.*2.*math.pi/60.), (0., -100.*2.*math.pi/60.))
            case['command'][:2] = motions[(variant//2) % len(motions)]
            case.update(push=True, push_sign=sign, actual_fall_required_for_success=True)
        # Ramp requested difficulty as well as task population. An immediately
        # full100rpm/3m/s target makes the native joint Gaussian tracking reward
        # effectively zero near rest. This changes reset-time request data only;
        # it never edits actions or targets during a physical interval.
        scale = exposure if task != 'standing' else 1.
        case['command_scale'] = scale
        case['final_command'] = case['command'].copy()
        case['final_sequence'] = deepcopy(case['sequence'])
        case['command'][:2] = [scale*value for value in case['command'][:2]]
        initial_height = case['command'][2]
        for _, command in case['sequence']:
            command[:2] = [scale*value for value in command[:2]]
            command[2] = initial_height + scale*(command[2]-initial_height)
        if 'requested_rpm' in case:
            case['final_requested_rpm'] = case['requested_rpm']
            case['requested_rpm'] *= scale
        return case

    def summary(self, completed_updates):
        cases = [self.case(lane, 0, completed_updates) for lane in range(self.count)]
        return dict(schema='fresh-unified-training-cases-candidate-v1',
            completed_updates=completed_updates, exposure=self.exposure(completed_updates),
            final_quota=self.quota, current_tasks=dict(Counter(case['training_group'] for case in cases)),
            foundation_updates=self.foundation_updates, full_tasks_at=self.full_tasks_at,
            request_difficulty='reset-time speed/yaw and height-transition amplitude ramp with exposure',
            independent_training_seed=self.seed, required_task_categories_present_at_final=True,
            smoothness_and_robustness='cross-cutting; measured under every task',
            actor_critic_and_Adam='fresh only; no checkpoint or teacher',
            training_adopted=False, native_verified=False, qualified=False, hardware_ready=False)
