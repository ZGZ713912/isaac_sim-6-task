"""Check actual Gaussian likelihoods, raw encoder units and inference parity."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel

from wheeled_algo.scratch_standing import agent_config, initialize_fresh
from wheeled_algo.wheel_velocity_increment import WheelVelocityIncrementModel
from wheeled_algo.native_ppo_audit import NativePPOAudit
from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask


def make(normalize=False):
    torch.manual_seed(261001)
    frame = torch.randn(8, 35) * .1
    frame[:, 20:22] = torch.tensor([.7, -.3])
    obs = TensorDict(dict(policy=frame, critic=torch.zeros(8, 83)), batch_size=[8])
    cfg = agent_config(261001, "cpu", 8, normalize_observations=normalize, wheel_velocity_increment=True)
    cfg["multi_gpu"] = None
    algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=6), cfg, "cpu")
    proof = initialize_fresh(algorithm, torch.tensor([.3, -.06, -.3, .06, 0., 0.]))
    return algorithm, obs, proof, cfg


def test_zero_learned_increment_preserves_signed_encoder_velocity_only_on_wheels():
    alg, obs, proof, _ = make()
    assert not alg.optimizer.state and proof["checkpoint_loaded"] is False
    with torch.no_grad():
        for parameter in alg.actor.mlp.parameters():
            parameter.zero_()
    result = alg.actor(obs)
    assert torch.equal(result[:, :4], torch.zeros(8, 4))
    assert torch.equal(result[:, 4:], obs["policy"][:, 20:22])
    torch.testing.assert_close(result[:, 4:] * 10., torch.tensor([[7., -3.]]).expand(8, 2))


@pytest.mark.parametrize("normalize", [False, True])
def test_final_sample_mean_and_log_probability_are_the_native_gaussian(normalize):
    alg, obs, _, _ = make(normalize)
    alg.actor.update_normalization(obs)
    expected = alg.actor(obs)
    sample = alg.actor(obs, stochastic_output=True)
    torch.testing.assert_close(alg.actor.output_mean, expected, rtol=0., atol=0.)
    expected_log_prob = torch.distributions.Normal(expected, alg.actor.output_std).log_prob(sample).sum(-1)
    torch.testing.assert_close(alg.actor.get_output_log_prob(sample), expected_log_prob, rtol=0., atol=0.)
    (-alg.actor.get_output_log_prob(sample.detach()).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in alg.actor.mlp.parameters())


@pytest.mark.parametrize("normalize", [False, True])
def test_jit_and_export_forward_retain_raw_identity_after_normalization(normalize):
    alg, obs, _, _ = make(normalize)
    alg.actor.update_normalization(obs)
    alg.actor.eval()
    frame = obs["policy"]
    expected = alg.actor(obs)
    scripted = torch.jit.script(alg.actor.as_jit())
    torch.testing.assert_close(scripted(frame), expected, rtol=0., atol=0.)
    torch.testing.assert_close(alg.actor.as_onnx()(frame), expected, rtol=0., atol=0.)
    scripted.reset()


def test_plain_mlp_checkpoint_loading_rejects_the_new_action_abi():
    alg, obs, _, cfg = make()
    actor_cfg = dict(cfg["actor"])
    actor_cfg.pop("class_name")
    plain = MLPModel(obs, cfg["obs_groups"], "actor", 6, **actor_cfg)
    with pytest.raises(RuntimeError, match="wheel_velocity_increment_abi"):
        plain.load_state_dict(alg.actor.state_dict(), strict=True)


def test_action_parameterization_rejects_critic_or_different_observation_layout():
    _, obs, _, _ = make()
    with pytest.raises(ValueError, match="exact35D"):
        WheelVelocityIncrementModel(obs, dict(actor=["critic"]), "actor", 6)


def test_fresh_model_completes_native_masked_ppo_without_likelihood_or_adam_shortcuts():
    alg, obs, proof, _ = make(normalize=True)
    assert proof["optimizer_states"] == 0 and not alg.optimizer.state
    attach_recovery_rollout_mask(alg, num_envs=8)
    before = {name: p.detach().clone() for name, p in alg.actor.named_parameters()}
    mask = torch.ones(8, dtype=torch.bool)
    for step in range(8):
        with torch.no_grad():
            action = alg.act(obs)
        next_obs = obs.clone()
        next_obs["policy"][:, 20:22] += .001
        rewards = torch.arange(8).float()*.01 + step*.001 - action.square().sum(-1)*.01
        alg.process_env_step(next_obs, rewards.detach(), torch.zeros_like(mask),
            dict(policy_learning_mask=mask, policy_bootstrap_mask=mask))
        obs = next_obs
    alg.compute_returns(obs)
    audit = NativePPOAudit(alg)
    try:
        losses = alg.update()
    finally:
        audit.close()
    report = audit.report()
    assert report["committed_optimizer_calls"] == 12 and report["all_recorded_values_finite"]
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(not torch.equal(before[name], p) for name, p in alg.actor.named_parameters())
    assert {int(state["step"]) for state in alg.optimizer.state.values()} == {12}
    assert alg.storage.step == 0
