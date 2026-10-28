"""Fresh native PPO construction; no checkpoint or teacher is an input."""
from __future__ import annotations

import hashlib
import torch


def agent_config(seed, device, rollout_steps, learning_rate=3e-4, *, normalize_observations=False,
                 wheel_velocity_increment=False, separate_leg_velocity_feedback=False):
    if separate_leg_velocity_feedback and not wheel_velocity_increment:
        raise ValueError("Separated leg feedback requires the explicit wheel-increment actor ABI")
    actor_class = ("wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel"
                   if wheel_velocity_increment else "MLPModel")
    if separate_leg_velocity_feedback:
        actor_class = "wheeled_algo.leg_velocity_separated:LegVelocitySeparatedModel"
    return dict(seed=seed, device=device, num_steps_per_env=rollout_steps,
        max_iterations=20000, empirical_normalization={},
        obs_groups=dict(actor=["policy"], critic=["critic"]), clip_actions=None,
        check_for_nan=True, save_interval=16, experiment_name="scratch_standing",
        run_name="", logger="tensorboard", resume=False, class_name="OnPolicyRunner",
        actor=dict(class_name=actor_class, hidden_dims=[128, 128, 64], activation="elu",
            obs_normalization=normalize_observations, distribution_cfg=dict(class_name="GaussianDistribution",
                init_std=.01, std_type="log", std_range=[.003, .3])),
        critic=dict(class_name="MLPModel", hidden_dims=[128, 128, 64], activation="elu",
            obs_normalization=normalize_observations, distribution_cfg=None),
        algorithm=dict(class_name="PPO", num_learning_epochs=3, num_mini_batches=4,
            learning_rate=learning_rate, schedule="fixed", gamma=.9974905699336811,
            lam=.9872585449014338, entropy_coef=.0002, desired_kl=.01,
            max_grad_norm=1., optimizer="adam", value_loss_coef=1.,
            use_clipped_value_loss=False, clip_param=.2,
            normalize_advantage_per_mini_batch=False, share_cnn_encoders=False,
            rnd_cfg=None, symmetry_cfg=None), policy={})


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_fresh(algorithm, nominal_action, *, output_bias_source=None, initial_std=None):
    if algorithm.optimizer.state:
        raise ValueError("Scratch training requires empty Adam state")
    if not all(p.requires_grad for model in (algorithm.actor, algorithm.critic) for p in model.parameters()):
        raise ValueError("Scratch actor and critic must be fully trainable")
    initial_std = [.01]*4 + [.05]*2 if initial_std is None else initial_std
    std = nominal_action.new_tensor(initial_std)
    if std.shape != (6,) or not torch.isfinite(std).all() or (std < .003).any() or (std > .3).any():
        raise ValueError("Fresh six-axis exploration must be within the declared0.003..0.3 distribution range")
    layers = [layer for layer in algorithm.actor.mlp.modules() if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        layers[-1].weight.mul_(.01)
        # This bias is derived from declared mechanics, never from a policy.
        # It remains an ordinary fully trainable network weight.
        layers[-1].bias.copy_(nominal_action)
        algorithm.actor.distribution.log_std_param.copy_(std.log())
    return dict(schema="scratch-ppo-initialization-v1", checkpoint_loaded=False,
        teacher_loaded=False, actor_updates=0, critic_updates=0, optimizer_states=0,
        actor_tensor_sha256=tensor_hash(algorithm.actor.state_dict()),
        critic_tensor_sha256=tensor_hash(algorithm.critic.state_dict()),
        actor_trainable_parameters=sum(p.numel() for p in algorithm.actor.parameters()),
        critic_trainable_parameters=sum(p.numel() for p in algorithm.critic.parameters()),
        output_bias_source=(output_bias_source or "feasible 0.305 m reset joint geometry; all six outputs trainable"),
        output_bias_policy_order=nominal_action.cpu().tolist(),
        initial_std_policy_order=[float(v) for v in initial_std])
