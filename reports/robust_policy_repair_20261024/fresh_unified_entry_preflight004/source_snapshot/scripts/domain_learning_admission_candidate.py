"""Unadopted curriculum candidate: gradual PPO admission, full physical domains.

No mass, sensor, gain, torque, reset or action is changed. The existing native
RecoveryRolloutMask compacts admitted samples for both PPO and normalization.
This module is outside the active learner's frozen source set and is not used
by that learner. Integration requires a new, fully fresh experiment.
"""
import math
import torch


def plan(warmup_updates,ramp_updates,total_updates,seed):
    if (any(type(value) is not int for value in [warmup_updates,ramp_updates,total_updates,seed])
            or warmup_updates<0 or ramp_updates<1 or total_updates<=warmup_updates+ramp_updates):
        raise ValueError('Admission needs nonnegative warmup, positive ramp and at least one full-population update')
    return dict(schema='standing-domain-learning-admission-plan-v1',warmup_updates=warmup_updates,
        ramp_updates=ramp_updates,total_updates=total_updates,priority_seed=seed,
        full_population_at_completed_update=warmup_updates+ramp_updates,
        physical_domain_widths_changed=False,normalization_and_PPO_use_same_mask=True,
        native_behavior_verified=False,hardware_ready=False)


class Admission:
    def __init__(self,enabled,*,warmup_updates=128,ramp_updates=256,seed=261501):
        if (enabled.ndim!=1 or enabled.dtype!=torch.bool or not enabled.any() or enabled.all()
                or type(warmup_updates) is not int or warmup_updates<0
                or type(ramp_updates) is not int or ramp_updates<1 or type(seed) is not int):
            raise ValueError('Need both native domain populations and a nonnegative warmup/positive ramp')
        self.enabled=enabled.clone()
        self.warmup,self.ramp,self.seed=warmup_updates,ramp_updates,seed
        self.nominal=(~enabled).clone()
        ids=enabled.cpu().nonzero().flatten()
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.order=ids[torch.randperm(len(ids),generator=generator)].to(enabled.device)

    def mask(self,completed_updates):
        if type(completed_updates) is not int or completed_updates<0:
            raise ValueError('Admission clock is completed fresh actor updates')
        elapsed=min(self.ramp,max(0,completed_updates-self.warmup))
        count=len(self.order)*elapsed//self.ramp
        result=self.nominal.clone()
        result[self.order[:count]]=True
        return result


def diagnostic_groups(state,native_learning,selected,enabled,done):
    """True pre-reset state; expose each population separately in RSL events."""
    count=len(enabled)
    if (native_learning.shape!=(count,) or selected.shape!=(count,) or done.shape!=(count,)
            or any(state[name].shape!=(count,3) for name in ['velocity','gravity','commands'])
            or state['height'].shape!=(count,)):
        raise ValueError('Admission diagnostics require native pre-reset population shapes')
    result={}
    for name,group in [('nominal',~enabled),('randomized',enabled),('admitted',selected)]:
        active=group & native_learning
        prefix='DomainAdmission/'+name+'/'
        result[prefix+'sample_count']=active.float().sum()
        result[prefix+'episode_end_events']=(done & active).float().sum()
        # A zero count is explicit; an empty population must not emit NaN or
        # manufacture a passing measurement from post-reset state.
        zero=state['height'].new_zeros(())
        result[prefix+'planar_speed_rms_m_s']=state['velocity'][active,:2].square().sum(-1).mean().sqrt() if active.any() else zero
        result[prefix+'tilt_mean_deg']=torch.acos((-state['gravity'][active,2]).clamp(-1.,1.)).mean()*180./math.pi if active.any() else zero
        result[prefix+'height_mae_m']=(state['height'][active]-state['commands'][active,2]).abs().mean() if active.any() else zero
    return result


def install(env,algorithm,admission,*,record_group_metrics=False):
    if (hasattr(env,'domain_learning_admission') or env.fixed_cases is not None
            or algorithm.storage.step!=0 or not hasattr(algorithm,'_recovery_rollout_mask')
            or algorithm.storage.num_envs!=len(admission.enabled)
            or not torch.equal(env.domain.enabled,admission.enabled)):
        raise ValueError('Install once before fresh learning with native rollout masking and fixed physical domains')
    native_step=env.step

    def step(actions):
        if env.fixed_cases is not None:
            return native_step(actions)
        mask=admission.mask(env.actor_updates)
        obs,reward,done,extras=native_step(actions)
        if record_group_metrics:
            extras['log'].update(diagnostic_groups(extras['diagnostics'],
                extras['policy_learning_mask'],mask,admission.enabled,done))
        # Never re-enable a passive/terminated native interval or bootstrap.
        extras['policy_learning_mask']=extras['policy_learning_mask'] & mask
        extras['policy_bootstrap_mask']=extras['policy_bootstrap_mask'] & mask
        extras['log'].update({
            'DomainAdmission/admitted_nominal_lanes':(mask & ~admission.enabled).float().sum(),
            'DomainAdmission/admitted_randomized_lanes':(mask & admission.enabled).float().sum(),
            'DomainAdmission/population_fraction':mask.float().mean(),
            'DomainAdmission/completed_actor_updates':mask.new_tensor(env.actor_updates,dtype=torch.float32)})
        return obs,reward,done,extras

    env.step=step
    env.domain_learning_admission=dict(schema='standing-domain-learning-admission-candidate-v1',
        warmup_updates=admission.warmup,ramp_updates=admission.ramp,priority_seed=admission.seed,
        randomized_priority_order=admission.order.cpu().tolist(),
        full_population_at_completed_update=admission.warmup+admission.ramp,
        physical_domains='full requested domains sampled at startup; no mid-episode or mid-training model writes',
        widths_changed=False,actions_changed=False,state_writes=False,
        losses_and_normalization='existing native RecoveryRolloutMask compact valid population',
        training_adopted=False,native_behavior_verified=False,hardware_ready=False)
    env.domain_learning_admission['record_pre_reset_group_metrics']=record_group_metrics
    return env.domain_learning_admission
