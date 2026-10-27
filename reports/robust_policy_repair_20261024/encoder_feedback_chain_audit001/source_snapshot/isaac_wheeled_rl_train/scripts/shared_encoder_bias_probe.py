"""Prepared zero-PPO test of a common encoder zero in policy and leg PD.

The real encoder zero is one sensor calibration, not two independent zeros.
Retain the original PD draws/RNG advance and change only its four leg biases
to the policy's same motor biases. No simulator state, action or gains change.
This helper is not installed in training or the currently frozen native queue.
"""
import torch

VERSION='shared-policy-PD-encoder-zero-probe-v1'
CONTROL_LEGS=(0,1,3,4)


def motor_bias(policy_sensor_bias):
    if policy_sensor_bias.ndim!=2 or policy_sensor_bias.shape[-1]!=18:
        raise ValueError('Need the native18D sensor bias in policy order')
    if not torch.isfinite(policy_sensor_bias).all():
        raise ValueError('Encoder bias must be finite')
    result=policy_sensor_bias.new_zeros((len(policy_sensor_bias),6))
    result[:,CONTROL_LEGS]=policy_sensor_bias[:,6:10]
    return result


def install(env):
    if hasattr(env,'shared_encoder_bias_probe'):
        raise ValueError('Do not stack encoder bias interventions')
    feedback=env.getup_feedback
    domain=env.domain
    if not torch.equal(feedback.enabled,domain.enabled):
        raise ValueError('PD and policy must use the same actual nominal/randomized lane mask')
    if feedback.rng is not domain.rng:
        raise ValueError('The archived native shared RNG draw path changed')
    original=feedback.reset

    def reset(ids,q,dq):
        original(ids,q,dq)  # Consume exactly the native draws, including its discarded independent bias.
        feedback.bias[ids]=motor_bias(domain.bias[ids])

    feedback.reset=reset
    feedback.bias.copy_(motor_bias(domain.bias))
    report=dict(schema=VERSION,changed='PD static encoder bias only; same four leg zeros as the policy',
        policy_sensor_bias_unchanged=True,all_native_RNG_draws_retained=True,
        zero_widths_reduced=False,wheel_position_bias=0.,state_writes=False,
        new_PPO=0,policy_weights_changed=False,training_adopted=False,
        native_validated=False,behavior_qualified=False,hardware_ready=False)
    env.shared_encoder_bias_probe=report
    return report
