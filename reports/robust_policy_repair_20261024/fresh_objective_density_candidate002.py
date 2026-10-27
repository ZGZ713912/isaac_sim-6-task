"""Unadopted dense standing velocity and transient leg-speed reward candidate.

Only pure numeric costs are prepared while017 keeps its frozen source. Native
phase eligibility, dynamics and acceptance are not changed by this module.
"""
import torch


def standing_velocity_density(velocity, omega):
    if (velocity.ndim != 2 or velocity.shape[1] != 3 or omega.shape != velocity.shape
            or not torch.isfinite(velocity).all() or not torch.isfinite(omega).all()):
        raise ValueError('Need simultaneous finite true planar velocity and gyro')
    error = velocity[:, :2].square().sum(-1)/.05**2 + (omega[:, 2]/.2).square()
    return .5*torch.log1p(error)


def active_speed_density(dq):
    if dq.ndim != 3 or dq.shape[0] != 5 or dq.shape[2] != 4 or not torch.isfinite(dq).all():
        raise ValueError('Need all5 actual1ms samples of all4 leg motors')
    squared = (dq/5.).square()
    return .2*(squared+squared.square()).mean((0, 2))


def contract():
    return dict(schema='unadopted-fresh-objective-density-candidate-v2',
        pure_standing_velocity_formula='0.5*log1p((vx^2+vy^2)/0.05^2+(wz/0.2)^2)',
        pure_standing_scope='upright-born standing; outside pre-interval and pre-reset recovery',
        active_speed_formula='0.2*mean_all5physical_samples_and4legs((dq/5)^2+(dq/5)^4)',
        active_speed_scope='valid active-control roles only; no passive or fixed-evaluation cost',
        unchanged_attitude_formula='2*log1p(2*clamp(1+gravity_z,0,2)/sin(5deg)^2)',
        physical_dt_s=.001, policy_dt_s=.005, candidate_adopted=False,
        physical_model_changed=False, action_changed=False, PD_changed=False,
        actual_joint_speed_clamped=False, domain_widths_changed=False,
        native_verified=False, behavior_qualified=False)
