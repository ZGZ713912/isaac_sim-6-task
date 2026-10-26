"""Experimental feedback correction above the frozen source actor.

The legacy gate closes at 85% of signed encoder demand. The optional tracking
gate uses bidirectional per-wheel error so imperfect tracking remains trainable.
Both are local same-input properties, not closed-loop skill preservation.
Wheel encoders are not root velocity.
"""
from __future__ import annotations

from copy import deepcopy
import json

import torch
from torch import nn

from .command_residual import CommandResidualActor


KIND = "original-command-and-gated-feedback-residual-v1"
TRACKING_KIND = "original-command-and-tracking-feedback-residual-v2"


class GuardedFeedbackActor(nn.Module):
    def __init__(self, command: CommandResidualActor, *, tracking_gate=False):
        super().__init__()
        if type(command) is not CommandResidualActor:
            raise ValueError("Feedback extension requires the declared source/command mean")
        self.command = deepcopy(command)
        self.tracking_gate = bool(tracking_gate)
        center, scale = torch.zeros(28), torch.ones(28)
        center[:4], scale[:4] = command.command_center, command.command_scale
        self.register_buffer("input_center", center)
        self.register_buffer("input_scale", scale)
        self.register_buffer("output_scale", torch.tensor([.1, .1, .1, .1, .5, .5]))
        self.feedback = nn.Sequential(nn.Linear(28, 32), nn.ELU(), nn.Linear(32, 6))
        nn.init.zeros_(self.feedback[2].weight)
        nn.init.zeros_(self.feedback[2].bias)
        self.set_trainable(True)

    @staticmethod
    def feedback_gate(obs):
        demand = torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5)
        low = (1. - demand).clamp(0., 1.)
        # Policy wheel dq entries are physical L/R encoder velocities * .1.
        # Opposite wheel axes require the right target's negative sign.
        turn = obs[:, 2:3] * (.4373 / 2)
        expected = torch.cat((obs[:, :1] - turn, -obs[:, :1] - turn), -1) / .6
        agreement = (obs[:, 20:22] * expected).sum(-1, keepdim=True) / expected.square().sum(-1, keepdim=True).clamp_min(1e-6)
        underspeed = (1. - agreement / .85).clamp(0., 1.)
        return torch.maximum(low, underspeed).square()

    @staticmethod
    def tracking_feedback_gate(obs):
        demand = torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5)
        low = (1. - demand).clamp(0., 1.).square()
        turn = obs[:, 2:3] * (.4373 / 2)
        expected = torch.cat((obs[:, :1] - turn, -obs[:, :1] - turn), -1) / .6
        # Both signs and the component orthogonal to a yaw demand matter.
        error_m_s = (obs[:, 20:22] - expected).abs().amax(-1, keepdim=True) * .6
        tracking = ((error_m_s - .015) / .085).clamp(0., 1.)
        return torch.maximum(low, tracking)

    def feedback_actions(self, obs):
        features = (obs[:, :28] - self.input_center) * self.input_scale
        raw = self.output_scale * torch.tanh(self.feedback(features))
        gate = self.tracking_feedback_gate(obs) if self.tracking_gate else self.feedback_gate(obs)
        return gate * raw

    def forward(self, obs):
        return self.command(obs) + self.feedback_actions(obs)

    def set_trainable(self, enabled, *, train_command=True):
        self.command.set_trainable(enabled and train_command)
        self.feedback.requires_grad_(enabled)
        if not enabled:
            for parameter in self.feedback.parameters():
                parameter.grad = None

    def verify_original(self, original):
        self.command.verify_original(original)
        center, scale = torch.zeros(28), torch.ones(28)
        center[:4] = torch.tensor([0., 0., 0., 1.525])
        scale[:4] = torch.tensor([1/3, 1/3, 1/4, 1/.475])
        expected = dict(input_center=center, input_scale=scale,
                        output_scale=torch.tensor([.1, .1, .1, .1, .5, .5]))
        if any(not torch.equal(getattr(self, name).detach().cpu(), value) for name, value in expected.items()):
            raise ValueError("Feedback feature scaling or action bound changed")
        zero_gate = dict(low_command_limits=[.75, 1.5],
            wheel_dq_policy_indices=[20, 21], encoder_scale=.1, wheel_radius_m=.06, wheel_track_m=.4373)
        zero_gate.update(dict(max_linear_wheel_error_m_s=.015, full_gate_error_m_s=.1,
            tracking_gate="bidirectional_per_wheel_linear") if self.tracking_gate else
            dict(signed_encoder_demand_fraction_at_least=.85))
        return dict(original_feedback="exact_source_tensors", feedback_inputs=list(range(28)),
            feedback_parameters=sum(p.numel() for p in self.feedback.parameters()),
            feedback_zero_gate=zero_gate,
            bounded_feedback_action_offsets=self.output_scale.detach().cpu().tolist())

    def export_graph(self, source):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        from .onnx_actor_import import recover_actor

        original, _ = recover_actor(source)
        self.verify_original(original)
        model = self.command.export_graph(source)
        prefix = "gated_feedback."
        obs, output = model.graph.input[0].name, model.graph.output[0].name
        model.graph.node[-1].output[0] = prefix + "command"
        constants = {name: np.array([value], dtype=np.int64) for name, value in
                     dict(start0=0, start2=2, start20=20, end1=1, end3=3, end22=22, end28=28, axes=1, steps=1).items()}
        constants.update({name: np.array(value, dtype=np.float32) for name, value in
                          dict(vx_scale=1/.75, yaw_scale=1/1.5, zero=0., one=1., half_track=.4373/2,
                               encoder_velocity_scale=1/.6, inverse_agreement=1/.85, epsilon=1e-6).items()})
        constants["sum_axes"] = np.array([1], dtype=np.int64)
        constants.update({name: getattr(self, name).detach().cpu().numpy()
                          for name in ("input_center", "input_scale", "output_scale")})
        constants.update({"feedback." + n: value.detach().cpu().numpy()
                          for n, value in self.feedback.state_dict().items()})
        model.graph.initializer.extend(numpy_helper.from_array(value.copy(), prefix + name)
                                       for name, value in constants.items())
        names = lambda *values: [prefix + value for value in values]
        slice_inputs = lambda start, end: [obs, *names(start, end, "axes", "steps")]
        model.graph.node.extend([
            helper.make_node("Slice", slice_inputs("start0", "end28"), names("features")),
            helper.make_node("Sub", names("features", "input_center"), names("centered")),
            helper.make_node("Mul", names("centered", "input_scale"), names("normalized")),
            helper.make_node("Gemm", names("normalized", "feedback.0.weight", "feedback.0.bias"), names("hidden0"), transB=1),
            helper.make_node("Elu", names("hidden0"), names("hidden"), alpha=1.),
            helper.make_node("Gemm", names("hidden", "feedback.2.weight", "feedback.2.bias"), names("raw"), transB=1),
            helper.make_node("Tanh", names("raw"), names("bounded")),
            helper.make_node("Mul", names("bounded", "output_scale"), names("offset")),
            helper.make_node("Slice", slice_inputs("start0", "end1"), names("vx")),
            helper.make_node("Slice", slice_inputs("start2", "end3"), names("yaw")),
            helper.make_node("Abs", names("vx"), names("abs_vx")),
            helper.make_node("Abs", names("yaw"), names("abs_yaw")),
            helper.make_node("Mul", names("abs_vx", "vx_scale"), names("vx_demand")),
            helper.make_node("Mul", names("abs_yaw", "yaw_scale"), names("yaw_demand")),
            helper.make_node("Max", names("vx_demand", "yaw_demand"), names("demand")),
            helper.make_node("Sub", names("one", "demand"), names("inverse_demand")),
            helper.make_node("Clip", names("inverse_demand", "zero", "one"), names("gate")),
            helper.make_node("Mul", names("yaw", "half_track"), names("turn")),
            helper.make_node("Sub", names("vx", "turn"), names("left_demand")),
            helper.make_node("Neg", names("vx"), names("negative_vx")),
            helper.make_node("Sub", names("negative_vx", "turn"), names("right_demand")),
            helper.make_node("Concat", names("left_demand", "right_demand"), names("wheel_demand"), axis=1),
            helper.make_node("Mul", names("wheel_demand", "encoder_velocity_scale"), names("expected")),
            helper.make_node("Slice", slice_inputs("start20", "end22"), names("wheel_dq")),
            helper.make_node("Mul", names("wheel_dq", "expected"), names("agreement_product")),
            helper.make_node("ReduceSum", names("agreement_product", "sum_axes"), names("agreement_sum"), keepdims=1),
            helper.make_node("Mul", names("expected", "expected"), names("demand_squared")),
            helper.make_node("ReduceSum", names("demand_squared", "sum_axes"), names("demand_sum"), keepdims=1),
            helper.make_node("Max", names("demand_sum", "epsilon"), names("denominator")),
            helper.make_node("Div", names("agreement_sum", "denominator"), names("agreement")),
            helper.make_node("Mul", names("agreement", "inverse_agreement"), names("agreement_scaled")),
            helper.make_node("Sub", names("one", "agreement_scaled"), names("inverse_agreement_value")),
            helper.make_node("Clip", names("inverse_agreement_value", "zero", "one"), names("underspeed_gate")),
            helper.make_node("Max", names("gate", "underspeed_gate"), names("combined_gate")),
            helper.make_node("Mul", names("combined_gate", "combined_gate"), names("gate_squared")),
            helper.make_node("Mul", names("gate_squared", "offset"), names("correction")),
            helper.make_node("Add", names("command", "correction"), [output]),
        ])
        if self.tracking_gate:
            # Keep the legacy graph construction unchanged for old checkpoints.
            # ORT removes its now-unused agreement path for the tracking mode.
            tail = [deepcopy(node) for node in model.graph.node[-2:]]
            del model.graph.node[-2:]
            extra = dict(start1=np.array([1], dtype=np.int64), end2=np.array([2], dtype=np.int64),
                linear_wheel_scale=np.array(.6, dtype=np.float32),
                deadband=np.array(.015, dtype=np.float32), inverse_error_band=np.array(1/.085, dtype=np.float32))
            model.graph.initializer.extend(numpy_helper.from_array(value, prefix + key) for key,value in extra.items())
            model.graph.node.extend([
                helper.make_node("Sub", names("wheel_dq", "expected"), names("wheel_error")),
                helper.make_node("Abs", names("wheel_error"), names("absolute_wheel_error")),
                helper.make_node("Slice", names("absolute_wheel_error", "start0", "end1", "axes", "steps"), names("left_error")),
                helper.make_node("Slice", names("absolute_wheel_error", "start1", "end2", "axes", "steps"), names("right_error")),
                helper.make_node("Max", names("left_error", "right_error"), names("maximum_wheel_error")),
                helper.make_node("Mul", names("maximum_wheel_error", "linear_wheel_scale"), names("linear_error")),
                helper.make_node("Sub", names("linear_error", "deadband"), names("excess_error")),
                helper.make_node("Mul", names("excess_error", "inverse_error_band"), names("normalized_error")),
                helper.make_node("Clip", names("normalized_error", "zero", "one"), names("error_gate")),
                helper.make_node("Mul", names("gate", "gate"), names("low_gate_squared")),
                helper.make_node("Max", names("low_gate_squared", "error_gate"), names("tracking_gate")),
            ])
            tail[0].input[0] = prefix + "tracking_gate"
            model.graph.node.extend(tail)
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        gate_contract = self.verify_original(original)["feedback_zero_gate"]
        declaration = (dict(inputs=list(range(28)), gate=gate_contract,
            action_bounds=self.output_scale.detach().cpu().tolist()) if self.tracking_gate else
            dict(inputs=list(range(28)), vx_gate_m_s=.75, yaw_gate_rad_s=1.5,
                signed_encoder_demand_fraction=.85, action_bounds=self.output_scale.detach().cpu().tolist()))
        metadata.update(actor_adaptation=TRACKING_KIND if self.tracking_gate else KIND,
            feedback_residual=json.dumps(declaration, separators=(",", ":")))
        helper.set_model_props(model, metadata)
        onnx.checker.check_model(model)
        return model


def configure_guarded_feedback_training(actor, enabled, *, train_command=True):
    if type(actor.mlp) is not GuardedFeedbackActor:
        raise ValueError("Actor does not contain the declared guarded feedback extension")
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled, train_command=train_command)
    trainable_prefixes = (("mlp.command.residual.", "mlp.feedback.") if train_command else ("mlp.feedback.",))
    return dict(enabled=enabled,
        trainable_names=[name for name, parameter in actor.named_parameters() if parameter.requires_grad],
        trainable_parameters=sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad),
        frozen_state_names=[name for name in actor.state_dict() if not name.startswith(trainable_prefixes)])


def restore_command_branch(parent, command_parent, actor_parameter_names):
    """Explicit branch composition; retain all consumed work and other state.

    Only the named command mean and its four Adam slots come from the older
    command-only checkpoint. Feedback, critic, std, their Adam, RNG and lifetime
    counters stay with the current parent. This is not a full actor restoration.
    """
    from .command_residual import KIND as COMMAND_KIND
    if (parent["infos"].get("actor_adaptation") != KIND
            or command_parent["infos"].get("actor_adaptation") != COMMAND_KIND):
        raise ValueError("Command restoration requires declared feedback and command-only parents")
    for key in ("flat_base_source_onnx_sha256", "asset_manifest_sha256", "control_math_sha256", "num_envs"):
        if parent["infos"][key] != command_parent["infos"][key]:
            raise ValueError("Command restoration changes source, mechanics, timing or population")
    current_opt, previous_opt = parent["optimizer_state_dict"], command_parent["optimizer_state_dict"]
    old_names = [name.replace("mlp.command.", "mlp.", 1) for name in actor_parameter_names
                 if not name.startswith("mlp.feedback.")]
    if (len(current_opt["param_groups"]) != 2 or len(previous_opt["param_groups"]) != 2
            or len(actor_parameter_names) != len(current_opt["param_groups"][0]["params"])
            or len(old_names) != len(previous_opt["param_groups"][0]["params"])):
        raise ValueError("Command Adam parameter ownership differs")
    current_ids = dict(zip(actor_parameter_names, current_opt["param_groups"][0]["params"], strict=True))
    previous_ids = dict(zip(old_names, previous_opt["param_groups"][0]["params"], strict=True))
    adapted = deepcopy(parent)
    replaced = []
    for name, value in parent["actor_state_dict"].items():
        if not name.startswith("mlp.command."):
            continue
        old_name = name.replace("mlp.command.", "mlp.", 1)
        old_value = command_parent["actor_state_dict"][old_name]
        if value.shape != old_value.shape or (not name.startswith("mlp.command.residual.")
                and not torch.equal(value, old_value)):
            raise ValueError("Command restoration changes original weights or interface buffers")
        adapted["actor_state_dict"][name] = old_value.clone()
        if name.startswith("mlp.command.residual."):
            current_id, previous_id = current_ids[name], previous_ids[old_name]
            adapted["optimizer_state_dict"]["state"].pop(current_id, None)
            if previous_id in previous_opt["state"]:
                adapted["optimizer_state_dict"]["state"][current_id] = deepcopy(previous_opt["state"][previous_id])
            replaced.append(name)
    if len(replaced) != 4:
        raise ValueError("Command restoration must name exactly four learned tensors")
    adapted["infos"]["active_branch_actor_updates"] = command_parent["infos"]["active_branch_actor_updates"]
    return adapted, dict(command_weights_and_Adam="restored_from_declared_command_parent", replaced_tensors=replaced,
        feedback_critic_std_other_Adam_RNG_lifetime_counters="current_parent_exact",
        prior_consumption="retained_without_refund", current_parent_actor_restore=False,
        command_branch_actor_updates=adapted["infos"]["active_branch_actor_updates"])


def extend_feedback_parent(parent, actor, critic, optimizer, old_actor_names):
    """Identity-extension of a declared command-residual checkpoint and Adam.

    The source mean, learned command correction, critic, std, their existing
    optimizer slots, counters and saved RNG are retained. Four new feedback
    parameters have fresh Adam slots; their last layer is exactly zero.
    """
    from .command_residual import KIND as COMMAND_KIND

    if type(actor.mlp) is not GuardedFeedbackActor or parent["infos"].get("actor_adaptation") != COMMAND_KIND:
        raise ValueError("Feedback extension requires the previous command-residual branch")
    if any(bool(value.count_nonzero()) for value in actor.mlp.feedback[2].state_dict().values()):
        raise ValueError("Feedback identity extension must have exactly zero initial output")
    old_opt, new_opt = parent["optimizer_state_dict"], deepcopy(optimizer.state_dict())
    if (len(old_opt["param_groups"]) != 2 or len(new_opt["param_groups"]) != 2
            or len(old_opt["param_groups"][0]["params"]) != len(old_actor_names)
            or set(parent["critic_state_dict"]) != set(critic.state_dict())
            or any(value.shape != critic.state_dict()[name].shape
                   for name, value in parent["critic_state_dict"].items())):
        raise ValueError("Feedback extension changed Adam ownership or critic architecture")
    adapted = deepcopy(parent)
    new_state = deepcopy(actor.state_dict())
    renamed = lambda name: "mlp.command." + name.removeprefix("mlp.") if name.startswith("mlp.") else name
    for name, value in parent["actor_state_dict"].items():
        target = renamed(name)
        if target not in new_state or new_state[target].shape != value.shape:
            raise ValueError("Feedback extension cannot retain parent actor state: " + name)
        new_state[target] = value.clone()
    mapped = {renamed(name) for name in parent["actor_state_dict"]}
    expected_parent = {name for name in new_state if not name.startswith(
        ("mlp.feedback.", "mlp.input_center", "mlp.input_scale", "mlp.output_scale"))}
    if mapped != expected_parent:
        raise ValueError("Incomplete previous actor state for the identity extension")
    adapted["actor_state_dict"] = new_state
    current_names = list(dict(actor.named_parameters()))
    new_actor_ids = dict(zip(current_names, new_opt["param_groups"][0]["params"], strict=True))
    old_actor_ids = dict(zip(old_actor_names, old_opt["param_groups"][0]["params"], strict=True))
    new_opt["state"] = {}
    for name, old_id in old_actor_ids.items():
        if renamed(name) not in new_actor_ids:
            raise ValueError("Previous actor Adam parameter has no destination: " + name)
        if old_id in old_opt["state"]:
            new_opt["state"][new_actor_ids[renamed(name)]] = deepcopy(old_opt["state"][old_id])
    for index in (0, 1):
        ids = new_opt["param_groups"][index]["params"]
        new_opt["param_groups"][index] = deepcopy(old_opt["param_groups"][index])
        new_opt["param_groups"][index]["params"] = ids
    for old_id, new_id in zip(old_opt["param_groups"][1]["params"], new_opt["param_groups"][1]["params"], strict=True):
        if old_id in old_opt["state"]:
            new_opt["state"][new_id] = deepcopy(old_opt["state"][old_id])
    adapted["optimizer_state_dict"] = new_opt
    return adapted, dict(actor_function="exact_parent_plus_zero_feedback_extension",
        prior_mean_and_Adam="retained_exact_with_parameter_rename", critic_and_Adam="retained_exact",
        std_and_Adam="retained_exact_and_frozen", saved_RNG_and_consumption="retained_without_refund",
        new_feedback_Adam="empty", feedback_branch_actor_updates=0)
