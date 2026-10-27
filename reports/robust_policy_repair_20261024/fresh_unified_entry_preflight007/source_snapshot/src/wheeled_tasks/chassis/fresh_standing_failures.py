"""Explicit failure signal for pure-standing training cases in a mixed task.

Other case roles retain their native recovery horizon. Fixed evaluation never
uses this wrapper. No action is altered and all physical writes are resets.
"""
import math

import torch


def contract():
    return dict(version='fresh-case-scoped-standing-failure-v1', training_group='standing',
        eligible_birth='upright pure-standing case; never an initial-recovery birth',
        max_tilt_deg=20., max_height_error_m=.025, additional_failure_cost=3.,
        enabled_for_fixed_evaluation=False, other_case_roles_unchanged=True,
        physical_state_writes='ordinary episode reset only', actions_changed=False,
        native_recovery_and_mechanical_guards_unchanged=True)


def install(env):
    if (hasattr(env,'fresh_standing_failure_training') or not hasattr(env,'precise_standing_reward')
            or not hasattr(env,'training_group') or not env.cfg.get('record_diagnostics')):
        raise ValueError('Attach once to fresh mixed training after the physical and precise observers')
    declared=contract()
    if env.getup_cfg.get('fresh_standing_failure_training')!=declared:
        raise ValueError('Case-scoped failure training must be declared before native construction')
    standing=env.training_case_names.index('standing')
    native_step=env.step

    def step(actions):
        groups=env.training_group.clone()
        birth_recovery=env.recovery_lane.clone()
        observation,reward,done,extras=native_step(actions)
        state=extras['diagnostics']
        if env.fixed_cases is not None:
            return observation,reward,done,extras
        tilted=state['gravity'][:,2]>-math.cos(math.radians(declared['max_tilt_deg']))
        wrong_height=(state['height']-state['commands'][:,2]).abs()>declared['max_height_error_m']
        eligible=(groups==standing)&~birth_recovery&extras['policy_learning_mask']
        failure=(tilted|wrong_height)&eligible&~done
        reward=reward-declared['additional_failure_cost']*failure
        done=done|failure
        extras['time_outs'][failure]=False
        extras['policy_bootstrap_mask'][failure]=False
        state['terminated']=state['terminated']|failure
        state['done']=done.clone()
        state['fresh_standing_failure']=failure.clone()
        log=extras['log']
        count=failure.float().sum()
        for tag in ('UnifiedGetup/episode_events','UnifiedGetup/terminated_events',
                    'FreshTasks/standing/episode_end_events'):
            log[tag]=log.get(tag,reward.new_zeros(()))+count
        log.update({'FreshStandingFailure/episode_events':count,
            'FreshStandingFailure/tilt_events':(failure&tilted).float().sum(),
            'FreshStandingFailure/height_events':(failure&wrong_height).float().sum()})
        if failure.any():
            env.reset(failure.nonzero().flatten())
            env.history.last_tick=None
            observation=env.get_observations()
        return observation,reward,done,extras

    env.step=step
    env.fresh_standing_failure_training=declared
    return declared
