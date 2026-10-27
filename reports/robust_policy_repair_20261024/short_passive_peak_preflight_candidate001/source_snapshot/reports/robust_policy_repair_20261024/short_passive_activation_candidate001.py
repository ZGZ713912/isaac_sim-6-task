"""Unadopted configuration-only candidate, for a later native comparison.

The original environment still performs one zero-effort5ms interval and then
evaluates its usual enable predicate. No pose, motor action or protection is
overridden. Preparing this file does not edit a live training configuration.
"""
from copy import deepcopy
import math


def validate_clocks(task):
    for key in ('passive_min_seconds','passive_max_seconds','passive_hold_seconds',
            'recovery_timeout_seconds','episode_seconds'):
        value=task[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0.:
            raise ValueError('Need positive finite original startup/recovery clocks')
    if not task['passive_min_seconds']<=task['passive_max_seconds']<task['recovery_timeout_seconds']<task['episode_seconds']:
        raise ValueError('Preserve ordered passive/recovery/episode clocks')


def prepare(task):
    if (task.get('policy_hz'),task.get('physics_hz'))!=(200,1000):
        raise ValueError('Candidate is explicitly bounded to200Hz/1000Hz')
    validate_clocks(task)
    result=deepcopy(task)
    keys=('passive_min_seconds','passive_max_seconds','passive_hold_seconds')
    previous={key:result[key] for key in keys}
    result.update({key:.005 for key in keys})
    validate_clocks(result)
    changed=[key for key in task if task[key]!=result[key]]
    if set(changed)!=set(keys):
        raise ValueError('Compare an original non-shortened startup protocol')
    declaration=dict(schema='short-passive-activation-candidate-v1',previous_clocks=previous,
        proposed_clocks={key:result[key] for key in keys},changed_settings=changed,
        native_first_policy_interval='zero motor effort exactly as the original environment',
        proposed_first_active_interval_one_based=2,training_adopted=False,native_verified=False,
        physical_state_writes='original reset only',actions_changed=False,PD_changed=False,
        guards_changed=False,randomization_widths_changed=False,acceptance_changed=False,
        old_samples_removed=False,behavior_qualified=False,hardware_ready=False,
        scope='CPU configuration only. The native impulse window starts on enable as before; earlier activation may expose more impacts/terminations. Neither reduced peak nor safe get-up is established.')
    return result,declaration
