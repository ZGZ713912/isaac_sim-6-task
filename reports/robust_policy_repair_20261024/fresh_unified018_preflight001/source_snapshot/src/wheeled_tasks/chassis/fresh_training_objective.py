"""Explicit fresh-training costs; all actions and native physics remain intact.

Dense attitude and velocity feedback apply only to upright-born pure standing.
The extra quadratic/quartic leg-speed cost covers each actual1ms active-control
sample, including recovery and motion. Passive placement receives no actor cost.
"""
import math
import torch


def contract():
    return dict(version='fresh-dense-standing-velocity-quartic-physical-speed-v2',
        standing_attitude_weight_per_second=2.,standing_angular_scale_deg=5.,
        standing_scope='upright-born pure-standing; neither pre-interval nor reward phase recovering',
        standing_formula='2*log1p(2*clamp(1+gravity_z,0,2)/sin(5deg)^2)',
        standing_velocity_weight_per_second=.5,standing_planar_speed_scale_m_s=.05,
        standing_yaw_speed_scale_rad_s=.2,
        standing_velocity_formula='0.5*log1p((vx^2+vy^2)/0.05^2+(wz/0.2)^2)',
        active_speed_weight_per_second=.2,active_speed_scale_rad_s=5.,
        active_speed_formula='0.2*mean_all5physical_samples_and4legs((dq/5)^2+(dq/5)^4)',
        active_speed_scope='every valid active-control role, including recovery, motion and completion',
        physical_dt_s=.001,policy_dt_s=.005,passive_placement_cost=False,
        fixed_evaluation_enabled=False,actions_changed=False,PD_changed=False,
        physical_state_writes=False,randomization_widths_changed=False,
        safety_speed_limit_claimed=False)


def attitude_density(gravity):
    if gravity.ndim!=2 or gravity.shape[1]!=3 or not torch.isfinite(gravity).all():
        raise ValueError('Need finite true gravity at the original pre-reset boundary')
    return 2.*torch.log1p(2.*(1.+gravity[:,2]).clamp(0.,2.)/math.sin(math.radians(5.))**2)


def active_speed_density(dq):
    if dq.ndim!=3 or dq.shape[0]!=5 or dq.shape[2]!=4 or not torch.isfinite(dq).all():
        raise ValueError('Need all5 actual1ms samples of all4 leg motors')
    squared=(dq/5.).square()
    return .2*(squared+squared.square()).mean((0,2))


def standing_velocity_density(velocity,omega):
    if (velocity.ndim!=2 or velocity.shape[1]!=3 or omega.shape!=velocity.shape
            or not torch.isfinite(velocity).all() or not torch.isfinite(omega).all()):
        raise ValueError('Need simultaneous finite true planar velocity and gyro')
    error=velocity[:,:2].square().sum(-1)/.05**2+(omega[:,2]/.2).square()
    return .5*torch.log1p(error)


def install(env):
    if (hasattr(env,'fresh_training_objective') or not hasattr(env,'fresh_standing_failure_training')
            or not hasattr(env,'physical_standing_reward') or not hasattr(env,'training_group')):
        raise ValueError('Attach once after physical, precision and scoped failure observers')
    declared=contract()
    if env.getup_cfg.get('fresh_training_objective')!=declared:
        raise ValueError('Declare the fresh objective before native construction')
    if env.dt!=.001 or env.policy_dt!=.005:
        raise ValueError('Fresh objective requires actual200Hz policy and1kHz samples')
    standing=env.training_case_names.index('standing')
    native_step=env.step

    def step(actions):
        groups=env.training_group.clone()
        recovery_birth=env.recovery_lane.clone()
        recovering=env.recovery_active.clone()
        active=~env.passive.clone()&~env.finished.clone()
        obs,reward,done,extras=native_step(actions)
        state=extras['diagnostics']
        active &= extras['policy_learning_mask']
        pure=(groups==standing)&~recovery_birth&~recovering&~state['reward_recovery_active']&active
        if env.fixed_cases is not None:
            active=torch.zeros_like(active)
            pure=torch.zeros_like(pure)
        attitude=attitude_density(state['gravity'])
        velocity=standing_velocity_density(state['velocity'],state['omega'])
        speed=active_speed_density(env.physical_standing_last['dq'])
        pose_paid=env.policy_dt*attitude*pure
        velocity_paid=env.policy_dt*velocity*pure
        speed_paid=env.policy_dt*speed*active
        env.fresh_training_objective_last=dict(attitude_density=attitude,active_speed_density=speed,
            velocity_density=velocity,paid_velocity=velocity_paid,
            pure_standing=pure,active=active,paid_attitude=pose_paid,paid_active_speed=speed_paid,
            unmodified_reward=reward.clone())
        extras['log'].update({'FreshObjective/pure_standing_lanes':pure.float().sum(),
            'FreshObjective/active_lanes':active.float().sum(),
            'FreshObjective/attitude_paid_cost_per_policy_step':pose_paid.mean(),
            'FreshObjective/velocity_paid_cost_per_policy_step':velocity_paid.mean(),
            'FreshObjective/active_speed_paid_cost_per_policy_step':speed_paid.mean(),
            'FreshObjective/attitude_density_per_second':(attitude*pure).mean(),
            'FreshObjective/velocity_density_per_second':(velocity*pure).mean(),
            'FreshObjective/active_speed_density_per_second':(speed*active).mean()})
        return obs,reward-pose_paid-velocity_paid-speed_paid,done,extras

    env.step=step
    env.fresh_training_objective=declared
    return declared
