"""Validate feedback separation in the actual Gaussian and native PPO path."""
from types import SimpleNamespace
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from wheeled_algo.scratch_standing import agent_config, initialize_fresh
from wheeled_algo.wheel_velocity_increment import WheelVelocityIncrementModel
from wheeled_algo.native_ppo_audit import NativePPOAudit
from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask


def make(normalize=False):
    torch.manual_seed(261001)
    frame = torch.randn(8, 35) * .1
    frame[:, 20:22] = torch.tensor([.7, -.3])
    obs = TensorDict(dict(policy=frame, critic=torch.randn(8, 83)), batch_size=[8])
    cfg = agent_config(261001, "cpu", 8, normalize_observations=normalize,
        wheel_velocity_increment=True, separate_leg_velocity_feedback=True)
    cfg["multi_gpu"] = None
    alg = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=6), cfg, "cpu")
    proof = initialize_fresh(alg, torch.tensor([.3, -.06, -.3, .06, 0., 0.]))
    return alg, obs, proof, cfg


@pytest.mark.parametrize("normalize", [False, True])
def test_leg_speed_has_zero_direct_leg_response_but_wheels_keep_full_feedback(normalize):
    alg, obs, _, _ = make(normalize)
    alg.actor.update_normalization(obs)
    alg.actor.eval()
    before = obs.clone()
    state = {k: v.clone() for k, v in alg.actor.state_dict().items()}
    changed = obs.clone()
    changed["policy"][:, 16:20] += torch.tensor([1., -.2, .7, -.6])
    original, result = alg.actor(obs), alg.actor(changed)
    assert torch.equal(original[:, :4], result[:, :4])
    assert not torch.equal(original[:, 4:], result[:, 4:])
    full = alg.actor.mlp(alg.actor.get_latent(changed))
    assert torch.equal(result[:, 4:], full[:, 4:] + changed["policy"][:, 20:22])
    assert torch.equal(obs["policy"], before["policy"])
    assert all(torch.equal(state[k], v) for k, v in alg.actor.state_dict().items())


def test_current_posture_and_joint_angles_still_reach_the_leg_targets():
    alg, obs, _, _ = make()
    frame = obs["policy"][0].clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda x: alg.actor(TensorDict(dict(policy=x[None]), batch_size=[1]))[0], frame)
    assert torch.equal(jacobian[:4, 16:20], torch.zeros(4, 4))
    for selected in (slice(3, 6), slice(6, 9), slice(9, 16), slice(22, 28)):
        assert jacobian[:4, selected].abs().max() > 0.
    assert jacobian[4:, 16:20].abs().max() > 0.


@pytest.mark.parametrize("normalize", [False, True])
def test_sampling_likelihood_gradients_and_inference_share_the_final_mean(normalize):
    alg, obs, _, _ = make(normalize)
    alg.actor.update_normalization(obs)
    alg.actor.eval()
    expected = alg.actor(obs)
    sample = alg.actor(obs, stochastic_output=True)
    assert torch.equal(alg.actor.output_mean, expected)
    expected_log = torch.distributions.Normal(expected, alg.actor.output_std).log_prob(sample).sum(-1)
    assert torch.equal(alg.actor.get_output_log_prob(sample), expected_log)
    (-alg.actor.get_output_log_prob(sample.detach()).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0. for p in alg.actor.mlp.parameters())
    assert alg.actor.distribution.log_std_param.grad.abs().max() > 0.
    scripted = torch.jit.script(alg.actor.as_jit())
    assert torch.equal(scripted(obs["policy"]), expected)
    assert torch.equal(alg.actor.as_onnx()(obs["policy"]), expected)
    scripted.reset()


def test_zero_network_wheel_identity_and_wrong_architecture_load_rejection():
    alg, obs, _, cfg = make()
    with torch.no_grad():
        for p in alg.actor.mlp.parameters():
            p.zero_()
    result = alg.actor(obs)
    assert not result[:, :4].any()
    assert torch.equal(result[:, 4:], obs["policy"][:, 20:22])
    model_cfg = dict(cfg["actor"])
    model_cfg.pop("class_name")
    old = WheelVelocityIncrementModel(obs, cfg["obs_groups"], "actor", 6, **model_cfg)
    with pytest.raises(RuntimeError, match="leg_velocity_separated_abi"):
        old.load_state_dict(alg.actor.state_dict(), strict=True)


def test_raw_normalization_updates_are_not_duplicated_or_neutralized():
    alg, obs, _, _ = make(normalize=True)
    model_cfg = agent_config(261001, "cpu", 8, normalize_observations=True,
        wheel_velocity_increment=True)["actor"]
    model_cfg.pop("class_name")
    old = WheelVelocityIncrementModel(obs, dict(actor=["policy"]), "actor", 6, **model_cfg)
    alg.actor.update_normalization(obs)
    old.update_normalization(obs)
    expected = old.obs_normalizer.state_dict()
    assert all(torch.equal(expected[k], v) for k, v in alg.actor.obs_normalizer.state_dict().items())


def test_separation_requires_the_declared_wheel_actor_abi():
    with pytest.raises(ValueError, match="wheel-increment"):
        agent_config(261001, "cpu", 8, separate_leg_velocity_feedback=True)


def test_fresh_native_masked_ppo_updates_actor_critic_and_adam_twelve_times():
    alg, obs, proof, _ = make(normalize=True)
    assert proof["optimizer_states"] == 0 and not alg.optimizer.state
    assert not proof["checkpoint_loaded"] and not proof["teacher_loaded"]
    attach_recovery_rollout_mask(alg, num_envs=8)
    before = [{n: p.detach().clone() for n, p in model.named_parameters()}
              for model in (alg.actor, alg.critic)]
    mask = torch.ones(8, dtype=torch.bool)
    for step in range(8):
        with torch.no_grad():
            action = alg.act(obs)
        next_obs = obs.clone()
        next_obs["policy"][:, 16:22] += .001
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
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    for previous, model in zip(before, (alg.actor, alg.critic)):
        assert any(not torch.equal(previous[n], p) for n, p in model.named_parameters())
    assert len(alg.optimizer.state) == 17
    assert {int(s["step"]) for s in alg.optimizer.state.values()} == {12}
    assert alg.storage.step == 0


def test_diagnostic_loader_uses_both_frozen_dependencies_and_rejects_tampering(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("feedback_reader_under_test", root / "scripts/inspect_scratch_feedback.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    hashes = {}
    for name in ("wheel_velocity_increment", "leg_velocity_separated"):
        relative = f"src/wheeled_algo/{name}.py"
        source = (root / relative).read_bytes()
        destination = tmp_path / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source)
        hashes[relative] = hashlib.sha256(source).hexdigest()
    cls = reader.frozen_actor_class("wheeled_algo.leg_velocity_separated:LegVelocitySeparatedModel",
                                    tmp_path, dict(source_hashes=hashes))
    assert Path(__import__("inspect").getsourcefile(cls)).is_relative_to(tmp_path)
    assert Path(__import__("inspect").getsourcefile(cls.__bases__[0])).is_relative_to(tmp_path)
    assert cls.__bases__[0] is not WheelVelocityIncrementModel
    alg, obs, _, cfg = make()
    model_cfg = dict(cfg["actor"])
    model_cfg.pop("class_name")
    archived = cls(obs, cfg["obs_groups"], "actor", 6, **model_cfg)
    archived.load_state_dict(alg.actor.state_dict(), strict=True)
    assert torch.equal(archived(obs), alg.actor(obs))
    (tmp_path / "source_snapshot/src/wheeled_algo/wheel_velocity_increment.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="dependency changed"):
        reader.frozen_actor_class("wheeled_algo.leg_velocity_separated:LegVelocitySeparatedModel",
                                  tmp_path, dict(source_hashes=hashes))
