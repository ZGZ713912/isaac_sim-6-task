"""Guard the new origin and the physical-rate cost missed by old training."""
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict

from wheeled_algo.scratch_standing import agent_config, initialize_fresh, tensor_hash
from wheeled_tasks.chassis.physical_standing_reward import motion_density


def test_first_four_substeps_are_not_aliased_out():
    velocity = torch.zeros(5, 2, 4)
    velocity[0, 0, 2] = .1  # Below the old 6 rad/s threshold; last sample is zero.
    speed, torque = motion_density(velocity, torch.zeros_like(velocity))
    assert speed[0] > 0. and speed[1] == 0. and not torque.any()


def test_no_deadband_or_sign_preference():
    velocity = torch.full((5, 2, 4), .001)
    positive, _ = motion_density(velocity, torch.zeros_like(velocity))
    negative, _ = motion_density(-velocity, torch.zeros_like(velocity))
    assert (positive > 0.).all() and torch.equal(positive, negative)
    zero, _ = motion_density(torch.zeros_like(velocity), torch.zeros_like(velocity))
    assert not zero.any()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_physics_fails_before_ppo(value):
    velocity = torch.zeros(5, 1, 4)
    velocity[2, 0, 1] = value
    with pytest.raises(ValueError, match="Nonfinite"):
        motion_density(velocity, torch.zeros_like(velocity))


def make_algorithm(seed, normalize=False):
    from rsl_rl.algorithms import PPO
    torch.manual_seed(seed)
    observation = TensorDict(dict(policy=torch.zeros(8, 35), critic=torch.zeros(8, 83)), batch_size=[8])
    cfg = agent_config(seed, "cpu", 8, normalize_observations=normalize)
    cfg["multi_gpu"] = None
    algorithm = PPO.construct_algorithm(observation, SimpleNamespace(num_envs=8, num_actions=6), cfg, "cpu")
    proof = initialize_fresh(algorithm, torch.tensor([.3, -.06, -.3, .06, 0., 0.]))
    return algorithm, proof


def test_fresh_initialization_is_reproducible_and_seed_specific():
    first, proof = make_algorithm(261001)
    same, same_proof = make_algorithm(261001)
    different, different_proof = make_algorithm(261002)
    assert proof == same_proof
    assert proof["actor_tensor_sha256"] != different_proof["actor_tensor_sha256"]
    assert proof["critic_tensor_sha256"] != different_proof["critic_tensor_sha256"]
    assert not first.optimizer.state and not same.optimizer.state and not different.optimizer.state
    assert proof["checkpoint_loaded"] is False and proof["actor_updates"] == 0
    assert all(p.requires_grad for p in first.actor.parameters())
    torch.testing.assert_close(first.actor.distribution.log_std_param.exp(), torch.tensor([.01]*4 + [.05]*2))


def test_existing_adam_state_is_rejected():
    algorithm, _ = make_algorithm(261001)
    parameter = next(algorithm.actor.parameters())
    algorithm.optimizer.state[parameter]["step"] = torch.tensor(1.)
    with pytest.raises(ValueError, match="empty Adam"):
        initialize_fresh(algorithm, torch.zeros(6))


def test_prone_exploration_is_explicit_fresh_and_does_not_inherit_standing_micro_noise():
    algorithm, _ = make_algorithm(261001)
    std = [.2]*4 + [.05]*2
    proof = initialize_fresh(algorithm, torch.zeros(6), initial_std=std)
    torch.testing.assert_close(algorithm.actor.distribution.log_std_param.exp(), torch.tensor(std))
    assert proof["initial_std_policy_order"] == std and not algorithm.optimizer.state
    with pytest.raises(ValueError, match="exploration"):
        initialize_fresh(algorithm, torch.zeros(6), initial_std=[.01]*5 + [float("nan")])


def test_normalized_models_are_fresh_and_store_their_own_statistics():
    algorithm, proof = make_algorithm(261001, normalize=True)
    assert algorithm.actor.obs_normalization and algorithm.critic.obs_normalization
    assert proof["checkpoint_loaded"] is False and not algorithm.optimizer.state
    assert any(name.startswith("obs_normalizer.") for name in algorithm.actor.state_dict())
