"""The experimental branch must preserve source weights and sensor feedback."""
from copy import deepcopy

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
import pytest
import torch

from wheeled_algo.command_residual import (CommandResidualActor, configure_command_residual_training,
                                          make_command_residual_parent, mean_from_checkpoint)
from wheeled_algo.onnx_actor_import import RecoveredActor


@pytest.fixture
def actors():
    torch.manual_seed(315)
    source = RecoveredActor().eval()
    residual = CommandResidualActor(source)
    return source, residual


def test_zero_branch_is_exact_source_and_adam_never_changes_source(actors):
    source, actor = actors
    obs = torch.randn(37, 35)
    before = deepcopy(source.state_dict())
    assert torch.equal(source(obs), actor(obs))
    optimizer = torch.optim.Adam(actor.parameters(), lr=.003)
    for _ in range(3):
        optimizer.zero_grad()
        (actor(obs) - source(obs).detach() - .03).square().mean().backward()
        optimizer.step()
    assert not torch.equal(source(obs), actor(obs))
    assert actor.verify_original(source)["residual_parameters"] == 358
    assert all(torch.equal(before[k], v) for k, v in source.state_dict().items())
    assert all(p.grad is None and p not in optimizer.state for p in actor.base.parameters())
    actor.set_trainable(False)
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    actor.set_trainable(True)
    assert all(not p.requires_grad for p in actor.base.parameters())
    assert all(p.requires_grad for p in actor.residual.parameters())


def test_feedback_jacobian_and_command_only_residual_after_nonzero_learning(actors):
    source, actor = actors
    with torch.no_grad():
        actor.residual[2].weight.normal_(std=.3)
        actor.residual[2].bias.fill_(.2)
    obs = torch.randn(19, 35, requires_grad=True)
    other = obs.detach().clone(); other[:, 4:] += torch.randn_like(other[:, 4:])
    assert torch.equal(actor.residual_actions(obs), actor.residual_actions(other))
    for k in range(6):
        actual = torch.autograd.grad(actor(obs)[:, k].sum(), obs)[0]
        expected = torch.autograd.grad(source(obs)[:, k].sum(), obs)[0]
        assert torch.equal(actual[:, 4:], expected[:, 4:])
    extreme = torch.randn(100, 35) * 1000
    assert bool((actor.residual_actions(extreme).abs() <= actor.output_scale).all())
    with torch.no_grad():
        actor.base[0].weight[0, 0] += .001
    with pytest.raises(ValueError, match="feedback weights"):
        actor.verify_original(source)


def test_onnx_branch_keeps_original_initializers_and_matches_nonzero_mean(actors, tmp_path):
    source, actor = actors
    nodes, tensors, previous = [], [], "obs"
    for index in (0, 2, 4, 6):
        weight, bias = f"w{index}", f"b{index}"
        tensors += [numpy_helper.from_array(source.mlp[index].weight.detach().numpy(), weight),
                    numpy_helper.from_array(source.mlp[index].bias.detach().numpy(), bias)]
        output = "actions" if index == 6 else f"linear{index}"
        nodes.append(helper.make_node("Gemm", [previous, weight, bias], [output], transB=1))
        previous = output
        if index != 6:
            previous = f"elu{index}"
            nodes.append(helper.make_node("Elu", [output], [previous]))
    graph = helper.make_graph(nodes, "mean", [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, ["batch", 35])],
        [helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, ["batch", 6])], tensors)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    path = tmp_path / "source.onnx"; onnx.save(model, path)
    original_bytes = path.read_bytes()
    with torch.no_grad():
        actor.residual[2].weight.normal_(std=.3); actor.residual[2].bias.fill_(.1)
    exported = actor.export_graph(path)
    assert path.read_bytes() == original_bytes
    assert all(a.SerializeToString() == b.SerializeToString()
               for a, b in zip(model.graph.initializer, exported.graph.initializer))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(exported.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for n in (1, 17, 128):
        obs = torch.randn(n, 35) * 3
        np.testing.assert_allclose(session.run(None, {"obs": obs.numpy()})[0], actor(obs).detach().numpy(), rtol=2e-5, atol=2e-5)
    obs = torch.randn(7, 35)
    for mean, kind in ((source, None), (actor, "original-feedback-command-residual-v1")):
        state = mean.state_dict() if kind is None else {"mlp."+n: v for n, v in mean.state_dict().items()}
        checkpoint = dict(infos=dict(actor_adaptation=kind), actor_state_dict=state)
        assert torch.equal(mean_from_checkpoint(checkpoint, path)(obs), mean(obs))
    checkpoint["infos"]["actor_adaptation"] = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        mean_from_checkpoint(checkpoint, path)


def test_declared_branch_preserves_critic_std_adam_rng_and_consumption(actors):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    source, _ = actors
    obs = TensorDict({"policy": torch.zeros(1, 35), "critic": torch.zeros(1, 83)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": .04, "std_type": "scalar"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1e-5, name="actor"),
                                  dict(params=critic.parameters(), lr=5e-4, name="critic")])
    loss = sum(p.square().mean() for p in (*actor.parameters(), *critic.parameters()))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    parent = dict(actor_state_dict=deepcopy(actor.state_dict()), critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=deepcopy(optimizer.state_dict()), infos=dict(successful_updates_total=384,
        actor_updates_total=334, consumed_optimizer_updates_total=385), torch_rng_state=torch.get_rng_state().clone())
    names = list(dict(actor.named_parameters()))
    actor.mlp = CommandResidualActor(source)
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1e-4, name="actor"),
                                  dict(params=critic.parameters(), lr=5e-4, name="critic")])
    rng = torch.get_rng_state().clone()
    adapted, proof = make_command_residual_parent(parent, actor, critic, optimizer, names)
    assert torch.equal(rng, torch.get_rng_state())
    assert adapted["infos"] == parent["infos"] and proof["active_branch_actor_updates"] == 0
    assert torch.equal(adapted["torch_rng_state"], parent["torch_rng_state"])
    actor.load_state_dict(adapted["actor_state_dict"], strict=True)
    critic.load_state_dict(adapted["critic_state_dict"], strict=True)
    optimizer.load_state_dict(adapted["optimizer_state_dict"])
    assert all(torch.equal(v, critic.state_dict()[k]) for k, v in parent["critic_state_dict"].items())
    before = deepcopy(optimizer.state_dict())
    old_group = parent["optimizer_state_dict"]["param_groups"][1]["params"]
    new_group = before["param_groups"][1]["params"]
    for old_id, new_id in zip(old_group, new_group):
        assert all(torch.equal(v, before["state"][new_id][k])
                   for k, v in parent["optimizer_state_dict"]["state"][old_id].items())
    std_before = actor.distribution.std_param.detach().clone()
    configuration = configure_command_residual_training(actor, True)
    assert configuration["trainable_parameters"] == 358
    inputs = torch.randn(11, 35)
    (actor.mlp(inputs) - source(inputs).detach() - .1).square().mean().backward(); optimizer.step()
    actor.mlp.verify_original(source)
    assert torch.equal(std_before, actor.distribution.std_param)
    std_id = before["param_groups"][0]["params"][0]
    assert all(torch.equal(v, optimizer.state_dict()["state"][std_id][k]) for k, v in before["state"][std_id].items())
    assert all(torch.equal(v, parent["actor_state_dict"][k])
               for k, v in adapted["actor_state_dict"].items() if not k.startswith("mlp."))
    # The same transform must never reset a second residual continuation.
    adapted["infos"]["actor_adaptation"] = "original-feedback-command-residual-v1"
    with pytest.raises(ValueError, match="full state"):
        make_command_residual_parent(adapted, CommandHolder(source), critic, optimizer, names)


class CommandHolder(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.mlp = CommandResidualActor(source)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_ppo_update_only_changes_declared_residual_and_critic(actors, enabled):
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict
    from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
    source, _ = actors
    obs = TensorDict({"policy": torch.randn(8, 35), "critic": torch.randn(8, 83)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": .04, "std_type": "scalar"})
    actor.mlp = CommandResidualActor(source)
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[12, 8], activation="elu", obs_normalization=False)
    alg = TeacherAnchoredPPO(actor, critic, RolloutStorage("rl", 8, 4, obs, [6]),
        num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-4, schedule="fixed", use_clipped_value_loss=False)
    alg.configure_teacher(source, coefficient=0., action_scales=[.025]*4+[.04]*2, min_vx=.75, min_yaw=1.5)
    scoped = configure_command_residual_training(actor, enabled)
    before = deepcopy(actor.state_dict()); value_before = deepcopy(critic.state_dict())
    with torch.no_grad():
        for _ in range(4):
            actions = alg.act(obs)
            alg.process_env_step(obs, -(actions - .1).square().sum(-1), torch.zeros(8), {})
        alg.compute_returns(obs)
    loss = alg.update()
    assert all(np.isfinite(x) for x in loss.values())
    assert all(torch.equal(before[n], actor.state_dict()[n]) for n in scoped["frozen_state_names"])
    assert any(not torch.equal(v, critic.state_dict()[n]) for n, v in value_before.items())
    changes = [n for n, v in before.items() if not torch.equal(v, actor.state_dict()[n])]
    assert bool(changes) == enabled
    assert all(n.startswith("mlp.residual.") for n in changes)
