"""Check the extra loss against native PPO, its command mask and gradients."""
from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from wheeled_algo.anchored_ppo import TeacherAnchoredPPO


def make_algorithm(kind=PPO):
    torch.manual_seed(31)
    obs = TensorDict(dict(policy=torch.randn(8, 35), critic=torch.randn(8, 40)), batch_size=[8])
    groups = dict(actor=["policy"], critic=["critic"])
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[12, 8], distribution_cfg=dict(
        class_name="GaussianDistribution", init_std=.1, std_type="scalar"))
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[12, 8])
    storage = RolloutStorage("rl", 8, 4, obs, [6])
    return kind(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2,
                learning_rate=1e-3, schedule="fixed", use_clipped_value_loss=False), obs


def configure(alg, coefficient):
    teacher = deepcopy(alg.actor.mlp)
    with torch.no_grad():
        teacher[-1].bias.add_(.2)
    alg.configure_teacher(teacher, coefficient=coefficient, action_scales=[.1] * 6, min_vx=.75, min_yaw=1.5)


def test_zero_coefficient_matches_native_ppo_parameters_adam_and_rng():
    native, obs = make_algorithm()
    with torch.no_grad():
        for step in range(4):
            actions = native.act(obs)
            next_obs = obs.apply(lambda x: x + .01)
            native.process_env_step(next_obs, -.1 * actions.square().sum(-1), torch.zeros(8), {})
            obs = next_obs
        native.compute_returns(obs)
    anchored = TeacherAnchoredPPO(deepcopy(native.actor), deepcopy(native.critic), deepcopy(native.storage),
        num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3, schedule="fixed", use_clipped_value_loss=False)
    configure(anchored, 0.)
    torch.manual_seed(57)
    expected_loss = native.update()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(57)
    actual_loss = anchored.update()
    assert all(actual_loss[key] == value for key, value in expected_loss.items())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.equal(value, anchored.save()[key][name]) for name, value in native.save()[key].items())
    for key, state in native.optimizer.state_dict()["state"].items():
        assert all(torch.equal(value, anchored.optimizer.state_dict()["state"][key][name]) for name, value in state.items())


def test_anchor_only_changes_actor_mean_on_selected_commands():
    alg, obs = make_algorithm(TeacherAnchoredPPO)
    configure(alg, .02)
    obs["policy"][:, [0, 2]] = .1
    low_loss, _, fraction = alg.teacher_loss(obs)
    assert low_loss.item() == 0. and fraction.item() == 0.
    obs["policy"][:, 2] = -4.
    before, _, fraction = alg.teacher_loss(obs)
    assert before.item() > 0 and fraction.item() == 1.
    optimizer = torch.optim.SGD(alg.actor.parameters(), lr=.001)
    optimizer.zero_grad()
    before.backward()
    assert all(p.grad is None for p in alg.teacher.parameters())
    assert all(p.grad is None for p in alg.actor.distribution.parameters())
    optimizer.step()
    assert alg.teacher_loss(obs)[0] < before


def test_unsupported_update_mode_is_rejected():
    alg, _ = make_algorithm(TeacherAnchoredPPO)
    alg.schedule = "adaptive"
    with pytest.raises(ValueError, match="flat feedforward"):
        configure(alg, .02)
