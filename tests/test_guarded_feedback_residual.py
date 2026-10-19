"""Candidate feedback extension: identity, command gate, Adam and real ONNX."""
from copy import deepcopy

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import pytest
import torch

from wheeled_algo.onnx_actor_import import RecoveredActor
from wheeled_algo.command_residual import CommandResidualActor, KIND as COMMAND_KIND
from wheeled_algo.guarded_feedback_residual import (GuardedFeedbackActor, configure_guarded_feedback_training,
                                                   extend_feedback_parent, restore_command_branch)


@pytest.fixture
def means():
    torch.manual_seed(611)
    original = RecoveredActor().eval()
    command = CommandResidualActor(original)
    with torch.no_grad():
        command.residual[2].weight.normal_(std=.05)
    return original, command, GuardedFeedbackActor(command)


def nonzero_feedback(actor):
    with torch.no_grad():
        actor.feedback[2].weight.normal_(std=.3)
        actor.feedback[2].bias.fill_(.2)


def test_initial_extension_is_exact_parent_and_freezes_original_after_learning(means):
    original, command, actor = means
    obs = torch.randn(24, 35); obs[:, :3] = 0.
    assert torch.equal(actor(obs), command(obs))
    optimizer = torch.optim.Adam(actor.parameters(), lr=.003)
    before = deepcopy(actor.command.base.state_dict())
    for _ in range(3):
        optimizer.zero_grad()
        (actor(obs) - command(obs).detach() - .1).square().mean().backward()
        optimizer.step()
    assert not torch.equal(actor(obs), command(obs))
    assert actor.verify_original(original)["feedback_parameters"] == 1126
    assert all(torch.equal(value, actor.command.base.state_dict()[name]) for name, value in before.items())
    assert all(parameter.grad is None and parameter not in optimizer.state for parameter in actor.command.base.parameters())
    actor.set_trainable(False)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in actor.parameters())


def test_tracking_high_command_feedback_jacobian_is_exact_parent_while_low_gate_can_learn(means):
    _, command, actor = means
    nonzero_feedback(actor)
    obs = torch.randn(12, 35); obs[:, :3] = 0.
    obs[:6, 0] = torch.tensor([-.75, .75, -3., 3., 0., 0.])
    obs[4:6, 2] = torch.tensor([-1.5, 1.5])
    turn = obs[:6, 2] * (.4373 / 2)
    obs[:6, 20] = (obs[:6, 0] - turn) / .6
    obs[:6, 21] = (-obs[:6, 0] - turn) / .6
    obs.requires_grad_(True)
    assert torch.equal(actor.feedback_gate(obs)[:6], torch.zeros(6, 1))
    assert torch.equal(actor(obs)[:6], command(obs)[:6])
    differs = False
    for k in range(6):
        actual = torch.autograd.grad(actor(obs)[:, k].sum(), obs)[0]
        expected = torch.autograd.grad(command(obs)[:, k].sum(), obs)[0]
        assert torch.equal(actual[:6, 4:], expected[:6, 4:])
        differs |= not torch.equal(actual[6:, 4:28], expected[6:, 4:28])
    assert differs
    extreme = torch.randn(64, 35) * 1000; extreme[:, :3] = 0.
    assert bool((actor.feedback_actions(extreme).abs() <= actor.output_scale).all())


def test_signed_encoder_gate_allows_failed_reverse_start_and_wrong_direction(means):
    _, _, actor = means
    obs = torch.zeros(5, 35)
    obs[:, 0] = -3.
    obs[:, 20] = torch.tensor([-5., -4.5, -.88, 0., 5.])
    obs[:, 21] = -obs[:, 20]
    gate = actor.feedback_gate(obs).squeeze(-1)
    assert torch.equal(gate[:2], torch.zeros(2))
    assert 0 < gate[2] < 1 and torch.equal(gate[3:], torch.ones(2))


def test_feedback_only_training_keeps_command_weights_and_existing_Adam_exact(means):
    _, _, mean = means
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mean
            self.std = torch.nn.Parameter(torch.ones(6))
    actor = Actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.003)
    x = torch.randn(16, 35); x[:, :3] = 0.
    for _ in range(2):
        configure_guarded_feedback_training(actor, True)
        optimizer.zero_grad(); actor.mlp(x).square().mean().backward(); optimizer.step()
    old_command = deepcopy(actor.mlp.command.state_dict())
    old_moments = {p: deepcopy(optimizer.state[p]) for p in actor.mlp.command.residual.parameters()}
    before_feedback = deepcopy(actor.mlp.feedback.state_dict())
    for _ in range(3):
        report = configure_guarded_feedback_training(actor, True, train_command=False)
        optimizer.zero_grad(); (actor.mlp(x) - .2).square().mean().backward(); optimizer.step()
    assert report["trainable_parameters"] == 1126
    assert all(n.startswith("mlp.feedback.") for n in report["trainable_names"])
    assert all(torch.equal(v, old_command[n]) for n, v in actor.mlp.command.state_dict().items())
    assert any(not torch.equal(v, before_feedback[n]) for n, v in actor.mlp.feedback.state_dict().items())
    for p, state in old_moments.items():
        assert all(torch.equal(v, optimizer.state[p][n]) for n, v in state.items())


def test_command_branch_composition_preserves_other_state_and_consumption(means):
    _, command, current = means
    def wrapper(mean):
        actor = torch.nn.Module()
        actor.register_parameter("std", torch.nn.Parameter(torch.ones(6)))
        actor.add_module("mlp", mean)
        return actor
    old_actor, new_actor = wrapper(command), wrapper(current)
    names = list(dict(new_actor.named_parameters()))
    shared = dict(flat_base_source_onnx_sha256="source", asset_manifest_sha256="asset", control_math_sha256="clock", num_envs=512)
    def state(actor, kind, step):
        parameters = list(actor.parameters())
        return dict(actor_state_dict=deepcopy(actor.state_dict()), critic_state_dict={"value": torch.tensor(6.)},
            optimizer_state_dict=dict(param_groups=[dict(params=list(range(len(parameters)))), dict(params=[100])],
                state={i:dict(step=torch.tensor(step), exp_avg=torch.full_like(p, step * .01), exp_avg_sq=torch.full_like(p, .2))
                       for i,p in enumerate(parameters)}),
            infos={**shared, "actor_adaptation":kind, "active_branch_actor_updates":step,
                   "successful_updates_total":512, "consumed_optimizer_updates_total":513},
            torch_rng_state=torch.get_rng_state())
    old = state(old_actor, COMMAND_KIND, 44)
    parent = state(new_actor, "original-command-and-gated-feedback-residual-v1", 88)
    with torch.no_grad():
        parent["actor_state_dict"]["mlp.command.residual.2.bias"].add_(.2)
    composed, proof = restore_command_branch(parent, old, names)
    assert not proof["current_parent_actor_restore"]
    assert composed["infos"]["active_branch_actor_updates"] == 44
    assert composed["infos"]["consumed_optimizer_updates_total"] == 513
    assert torch.equal(composed["torch_rng_state"], parent["torch_rng_state"])
    for name, value in parent["actor_state_dict"].items():
        expected = old["actor_state_dict"][name.replace("mlp.command.", "mlp.", 1)] if name.startswith("mlp.command.") else value
        assert torch.equal(composed["actor_state_dict"][name], expected)
    for i, name in enumerate(names):
        expected = old["optimizer_state_dict"]["state"][i] if name.startswith("mlp.command.residual.") else parent["optimizer_state_dict"]["state"][i]
        assert all(torch.equal(v, composed["optimizer_state_dict"]["state"][i][key]) for key, v in expected.items())


def write_source(path, source):
    nodes, tensors, previous = [], [], "obs"
    for index in (0, 2, 4, 6):
        names = [f"mlp.{index}.{suffix}" for suffix in ("weight", "bias")]
        tensors += [numpy_helper.from_array(value.detach().numpy(), name)
                    for name, value in zip(names, (source.mlp[index].weight, source.mlp[index].bias))]
        output = "actions" if index == 6 else f"linear{index}"
        nodes.append(helper.make_node("Gemm", [previous, *names], [output], transB=1))
        previous = output
        if index != 6:
            previous = f"elu{index}"
            nodes.append(helper.make_node("Elu", [output], [previous]))
    model = helper.make_model(helper.make_graph(nodes, "mean",
        [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, ["batch", 35])],
        [helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, ["batch", 6])], tensors),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.save(model, path)
    return model


@pytest.mark.parametrize("tracking_gate", (False, True))
def test_nonzero_onnx_keeps_all_source_tensors_and_dynamic_batches(means, tmp_path, tracking_gate):
    original, _, actor = means
    actor.tracking_gate = tracking_gate
    nonzero_feedback(actor)
    path = tmp_path / "original.onnx"; source = write_source(path, original)
    graph = actor.export_graph(path)
    assert all(a.SerializeToString() == b.SerializeToString()
               for a, b in zip(source.graph.initializer, graph.graph.initializer))
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    for n in (1, 19, 257):
        obs = torch.randn(n, 35) * 3
        obs[:n//2, :3] *= .1
        np.testing.assert_allclose(session.run(None, {"obs": obs.numpy()})[0], actor(obs).detach().numpy(), atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        actor.output_scale[0] *= 2
    with pytest.raises(ValueError, match="action bound"):
        actor.export_graph(path)


def test_tracking_gate_covers_both_signs_and_yaw_orthogonal_translation():
    x = torch.zeros(8, 35)
    x[:4, 0] = torch.tensor([3., 3., -3., -3.])
    x[4:, 2] = 4.
    turn = x[:, 2:3] * (.4373/2)
    expected = torch.cat((x[:, :1] - turn, -x[:, :1] - turn), -1) / .6
    x[:, 20:22] = expected
    x[1, 20:22] *= 1.05  # forward overspeed
    x[2, 20:22] *= .95  # backward underspeed
    x[3, 20:22] *= 1.05  # backward overspeed
    x[5, 20:22] += torch.tensor([.2, -.2])  # translation orthogonal to yaw demand
    x[6, 20:22] += .2  # yaw tracking error
    x[7, 20:22] += .01  # small per-wheel error, below the declared band
    actual = GuardedFeedbackActor.tracking_feedback_gate(x).flatten()
    torch.testing.assert_close(actual, torch.tensor([0., 1., 1., 1., 0., 1., 1., 0.]))
    assert GuardedFeedbackActor.feedback_gate(x)[1].item() == 0.
    assert GuardedFeedbackActor.feedback_gate(x)[5].item() == 0.


def test_tracking_gate_keeps_static_mean_exact_and_enables_high_error_gradients(means):
    _, _, legacy = means
    nonzero_feedback(legacy)
    revised = deepcopy(legacy); revised.tracking_gate = True
    static = torch.randn(32, 35); static[:, :3] = 0.
    assert torch.equal(legacy(static), revised(static))
    moving = static.clone(); moving[:, 0] = 3.; moving[:, 20] = 5.2; moving[:, 21] = -5.2
    assert not torch.equal(legacy(moving), revised(moving))
    revised(moving).square().mean().backward()
    assert revised.feedback[2].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in revised.command.base.parameters())
    assert all(torch.equal(v, revised.state_dict()[n]) for n,v in legacy.state_dict().items())


def test_tracking_gate_restores_real_rsl_weights_and_adam_without_branch_reset(means, tmp_path):
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.guarded_feedback_residual import TRACKING_KIND
    original, _, legacy = means
    nonzero_feedback(legacy)
    obs, actor, critic = rsl_models(legacy)
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1e-4, name="actor"),
                                  dict(params=critic.parameters(), lr=5e-4, name="critic")])
    (actor.mlp(obs["policy"]).square().mean() + critic(obs).square().mean()).backward()
    optimizer.step(); optimizer.zero_grad()
    state, moments = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
    source = tmp_path / "source.onnx"; write_source(source, original)
    recovered = mean_from_checkpoint(dict(actor_state_dict=state, infos=dict(actor_adaptation=TRACKING_KIND)), source)
    assert recovered.tracking_gate
    target = deepcopy(actor); target.mlp = recovered
    restored = torch.optim.Adam([dict(params=target.parameters(), lr=1e-4, name="actor"),
                                 dict(params=critic.parameters(), lr=5e-4, name="critic")])
    target.load_state_dict(state, strict=True); restored.load_state_dict(moments)
    assert list(dict(target.named_parameters())) == list(dict(actor.named_parameters()))
    assert all(torch.equal(v, target.state_dict()[n]) for n,v in state.items())
    for i, slots in moments["state"].items():
        assert all(torch.equal(v, restored.state_dict()["state"][i][n]) for n,v in slots.items())
    static = obs["policy"]
    assert torch.equal(target.mlp(static), actor.mlp(static))
    moving = static.clone(); moving[:, 0] = 3.; moving[:, 20] = 5.2; moving[:, 21] = -5.2
    assert not torch.equal(target.mlp(moving), actor.mlp(moving))
    configure_guarded_feedback_training(target, True, train_command=False)
    restored.zero_grad(); target.mlp(moving).square().mean().backward(); restored.step()
    assert all(torch.equal(v, target.state_dict()[n]) for n,v in state.items() if not n.startswith("mlp.feedback."))
    assert any(not torch.equal(v, target.state_dict()[n]) for n,v in state.items() if n.startswith("mlp.feedback."))


def rsl_models(mean):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    obs = TensorDict({"policy": torch.randn(8, 35), "critic": torch.randn(8, 83)}, batch_size=[8])
    obs["policy"][:, :3] = 0.
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": .04, "std_type": "scalar"})
    actor.mlp = mean
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    return obs, actor, critic


def test_identity_extension_retains_real_rsl_model_adam_rng_and_consumption(means):
    _, command, _ = means
    _, old_actor, critic = rsl_models(command)
    old_optimizer = torch.optim.Adam([dict(params=old_actor.parameters(), lr=1e-4, name="actor"),
                                      dict(params=critic.parameters(), lr=5e-4, name="critic")])
    sum(parameter.square().mean() for parameter in (*old_actor.parameters(), *critic.parameters())
        if parameter.requires_grad).backward()
    old_optimizer.step(); old_optimizer.zero_grad()
    parent = dict(actor_state_dict=deepcopy(old_actor.state_dict()), critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=deepcopy(old_optimizer.state_dict()), torch_rng_state=torch.get_rng_state().clone(),
        infos=dict(actor_adaptation=COMMAND_KIND, successful_updates_total=448, actor_updates_total=378,
                   active_branch_actor_updates=44, consumed_optimizer_updates_total=449))
    names = list(dict(old_actor.named_parameters()))
    actor = deepcopy(old_actor); actor.mlp = GuardedFeedbackActor(old_actor.mlp)
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1e-4, name="actor"),
                                  dict(params=critic.parameters(), lr=5e-4, name="critic")])
    rng = torch.get_rng_state().clone()
    adapted, proof = extend_feedback_parent(parent, actor, critic, optimizer, names)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(parent["torch_rng_state"], adapted["torch_rng_state"])
    assert adapted["infos"] == parent["infos"] and proof["feedback_branch_actor_updates"] == 0
    actor.load_state_dict(adapted["actor_state_dict"], strict=True); optimizer.load_state_dict(adapted["optimizer_state_dict"])
    obs = torch.randn(17, 35)
    assert torch.equal(actor.mlp(obs), old_actor.mlp(obs))
    for group in (0, 1):
        old_ids, new_ids = (opt["param_groups"][group]["params"] for opt in
                            (parent["optimizer_state_dict"], adapted["optimizer_state_dict"]))
        for old_id, new_id in zip(old_ids, new_ids):
            if old_id in parent["optimizer_state_dict"]["state"]:
                assert all(torch.equal(value, adapted["optimizer_state_dict"]["state"][new_id][name])
                           for name, value in parent["optimizer_state_dict"]["state"][old_id].items())
    assert all(i not in adapted["optimizer_state_dict"]["state"] for i in (13, 14, 15, 16))
    parent["infos"]["actor_adaptation"] = "already-extended"
    with pytest.raises(ValueError, match="previous command-residual"):
        extend_feedback_parent(parent, actor, critic, optimizer, names)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_cpu_ppo_updates_only_declared_corrections_and_critic(means, enabled):
    from rsl_rl.storage import RolloutStorage
    from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
    original, _, mean = means
    obs, actor, critic = rsl_models(mean)
    alg = TeacherAnchoredPPO(actor, critic, RolloutStorage("rl", 8, 4, obs, [6]),
        num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-4, schedule="fixed", use_clipped_value_loss=False)
    alg.configure_teacher(original, coefficient=0., action_scales=[.025]*4+[.04]*2, min_vx=.75, min_yaw=1.5)
    scope = configure_guarded_feedback_training(actor, enabled)
    assert scope["trainable_parameters"] == (1484 if enabled else 0)
    before = deepcopy(actor.state_dict()); value_before = deepcopy(critic.state_dict())
    with torch.no_grad():
        for _ in range(4):
            actions = alg.act(obs)
            alg.process_env_step(obs, -(actions - .1).square().sum(-1), torch.zeros(8), {})
        alg.compute_returns(obs)
    loss = alg.update()
    assert all(np.isfinite(value) for value in loss.values())
    assert all(torch.equal(before[name], actor.state_dict()[name]) for name in scope["frozen_state_names"])
    assert any(not torch.equal(value, critic.state_dict()[name]) for name, value in value_before.items())
    changes = [name for name, value in before.items() if not torch.equal(value, actor.state_dict()[name])]
    assert bool(changes) == enabled
    assert all(name.startswith(("mlp.command.residual.", "mlp.feedback.")) for name in changes)
    if enabled:
        assert any(name.startswith("mlp.feedback.") for name in changes)
