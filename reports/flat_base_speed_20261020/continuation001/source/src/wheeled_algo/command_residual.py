"""Candidate flat adaptation that cannot train the original sensor feedback.

This is an experimental mean architecture, not a behavioral qualification.
Only the four visible command entries feed the bounded residual. The original
35D actor is copied exactly, so at the same input the sensor/previous-action
Jacobian is unchanged. That local property does not prove closed-loop safety.
"""
from __future__ import annotations

from copy import deepcopy
import json

import torch
from torch import nn

from .onnx_actor_import import RecoveredActor, recover_actor


KIND = "original-feedback-command-residual-v1"


class CommandResidualActor(nn.Module):
    def __init__(self, original: RecoveredActor):
        super().__init__()
        if type(original) is not RecoveredActor:
            raise ValueError("Command residual requires the authenticated recovered mean")
        self.base = deepcopy(original.mlp).requires_grad_(False)
        # Policy order: command vx, vy, yaw, height*5. Center the height only.
        self.register_buffer("command_center", torch.tensor([0., 0., 0., 1.525]))
        self.register_buffer("command_scale", torch.tensor([1/3, 1/3, 1/4, 1/.475]))
        # Policy output order: four leg offsets, then two wheel targets.
        # Bounds correspond to .05 rad legs and 7.5 rad/s wheel setpoints.
        self.register_buffer("output_scale", torch.tensor([.2, .2, .2, .2, .75, .75]))
        self.residual = nn.Sequential(nn.Linear(4, 32), nn.ELU(), nn.Linear(32, 6))
        nn.init.zeros_(self.residual[2].weight)
        nn.init.zeros_(self.residual[2].bias)
        self.set_trainable(True)

    def residual_actions(self, obs):
        command = (obs[:, :4] - self.command_center) * self.command_scale
        return self.output_scale * torch.tanh(self.residual(command))

    def forward(self, obs):
        return self.base(obs) + self.residual_actions(obs)

    def set_trainable(self, enabled):
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        self.residual.requires_grad_(enabled)
        if not enabled:
            for parameter in self.residual.parameters():
                parameter.grad = None

    def verify_original(self, original):
        actual, expected = self.base.state_dict(), original.mlp.state_dict()
        if actual.keys() != expected.keys() or any(
                not torch.equal(actual[k].detach().cpu(), v.detach().cpu()) for k, v in expected.items()):
            raise ValueError("Frozen original feedback weights changed")
        fixed = dict(command_center=[0., 0., 0., 1.525], command_scale=[1/3, 1/3, 1/4, 1/.475],
                     output_scale=[.2, .2, .2, .2, .75, .75])
        for name, values in fixed.items():
            tensor = getattr(self, name)
            if not torch.equal(tensor.detach().cpu(), torch.tensor(values, dtype=torch.float32)):
                raise ValueError("Command residual bound/normalization changed: " + name)
        return dict(original_feedback="exact_source_tensors", command_only_inputs=[0, 1, 2, 3],
                    residual_parameters=sum(p.numel() for p in self.residual.parameters()),
                    bounded_action_offsets=self.output_scale.detach().cpu().tolist())

    def export_graph(self, source):
        """Append an explicit command-only ONNX branch to the exact source graph."""
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        original, recovered = recover_actor(source)
        self.verify_original(original)
        model = onnx.load(str(source))
        prefix = "command_residual."
        if any(v.name.startswith(prefix) for v in model.graph.initializer):
            raise ValueError("Source already contains a residual branch")
        obs, output = model.graph.input[0].name, model.graph.output[0].name
        model.graph.node[-1].output[0] = prefix + "base"
        constants = {"starts": np.array([0], dtype=np.int64), "ends": np.array([4], dtype=np.int64),
                     "axes": np.array([1], dtype=np.int64), "steps": np.array([1], dtype=np.int64)}
        constants.update({n: getattr(self, n).detach().cpu().numpy()
                          for n in ("command_center", "command_scale", "output_scale")})
        constants.update({"residual." + n: v.detach().cpu().numpy() for n, v in self.residual.state_dict().items()})
        model.graph.initializer.extend(numpy_helper.from_array(v.copy(), prefix + n) for n, v in constants.items())
        names = lambda *values: [prefix + v for v in values]
        model.graph.node.extend([
            helper.make_node("Slice", [obs, *names("starts", "ends", "axes", "steps")], names("command")),
            helper.make_node("Sub", names("command", "command_center"), names("centered")),
            helper.make_node("Mul", names("centered", "command_scale"), names("scaled")),
            helper.make_node("Gemm", names("scaled", "residual.0.weight", "residual.0.bias"), names("hidden"), transB=1),
            helper.make_node("Elu", names("hidden"), names("activated"), alpha=1.),
            helper.make_node("Gemm", names("activated", "residual.2.weight", "residual.2.bias"), names("raw"), transB=1),
            helper.make_node("Tanh", names("raw"), names("bounded")),
            helper.make_node("Mul", names("bounded", "output_scale"), names("delta")),
            helper.make_node("Add", [prefix + "base", prefix + "delta"], [output]),
        ])
        metadata = {v.key: v.value for v in model.metadata_props}
        metadata.update(actor_adaptation=KIND, source_onnx_sha256=recovered["source_sha256"],
                        residual_action_bounds=json.dumps(self.output_scale.detach().cpu().tolist()))
        helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model


def configure_command_residual_training(actor, enabled):
    if type(actor.mlp) is not CommandResidualActor:
        raise ValueError("Actor does not contain the declared command residual")
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    return dict(enabled=enabled, trainable_names=[n for n, p in actor.named_parameters() if p.requires_grad],
                trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
                frozen_state_names=[n for n in actor.state_dict() if not n.startswith("mlp.residual.")])


def make_command_residual_parent(parent, actor, critic, optimizer, old_actor_names):
    """Declared branch: source feedback + zero residual, parent critic/std/RNG.

    This deliberately discards the previous mean and its Adam slots. It never
    claims full actor continuation. All prior learning consumption is retained.
    The parent object is unchanged and its original full checkpoint stays valid.
    """
    if type(actor.mlp) is not CommandResidualActor:
        raise ValueError("Expected a zero-initialized command residual actor")
    if any(bool(v.count_nonzero()) for v in actor.mlp.residual[2].state_dict().values()):
        raise ValueError("A new residual branch must start with exactly zero output")
    if parent["infos"].get("actor_adaptation") == KIND:
        raise ValueError("An existing residual branch must resume its full state")
    expected_mean = {f"mlp.{i}.{suffix}" for i in (0, 2, 4, 6) for suffix in ("weight", "bias")}
    old_actor = parent["actor_state_dict"]
    if ({n for n in old_actor if n.startswith("mlp.")} != expected_mean
            or set(old_actor) != set(old_actor_names)):
        raise ValueError("The branch source is not the declared original-width RSL actor")
    old_opt = parent["optimizer_state_dict"]
    new_opt = deepcopy(optimizer.state_dict())
    if len(old_opt["param_groups"]) != 2 or len(new_opt["param_groups"]) != 2:
        raise ValueError("Residual migration requires separate actor and critic Adam groups")
    critic_names = list(dict(critic.named_parameters()))
    if (len(old_opt["param_groups"][0]["params"]) != len(old_actor_names)
            or len(old_opt["param_groups"][1]["params"]) != len(critic_names)
            or set(parent["critic_state_dict"]) != set(critic.state_dict())):
        raise ValueError("Parent Adam parameter ownership differs")
    adapted = deepcopy(parent)
    new_actor = deepcopy(actor.state_dict())
    for name in set(new_actor) - {n for n in new_actor if n.startswith("mlp.")}:
        if name not in old_actor or new_actor[name].shape != old_actor[name].shape:
            raise ValueError("Exploration distribution changed while creating the branch")
        new_actor[name] = old_actor[name].clone()
    adapted["actor_state_dict"] = new_actor
    # Critic optimizer hyperparameters and moments remain exact. Only its
    # integer parameter identifiers move after insertion of residual weights.
    new_critic_ids = new_opt["param_groups"][1]["params"]
    new_opt["param_groups"][1] = deepcopy(old_opt["param_groups"][1])
    new_opt["param_groups"][1]["params"] = new_critic_ids
    new_opt["state"] = {}
    for old_id, new_id in zip(old_opt["param_groups"][1]["params"], new_critic_ids, strict=True):
        if old_id in old_opt["state"]:
            new_opt["state"][new_id] = deepcopy(old_opt["state"][old_id])
    old_ids = dict(zip(old_actor_names, old_opt["param_groups"][0]["params"], strict=True))
    new_ids = dict(zip(dict(actor.named_parameters()), new_opt["param_groups"][0]["params"], strict=True))
    for name, old_id in old_ids.items():
        if not name.startswith("mlp.") and old_id in old_opt["state"]:
            new_opt["state"][new_ids[name]] = deepcopy(old_opt["state"][old_id])
    adapted["optimizer_state_dict"] = new_opt
    proof = dict(actor_mean="reset_to_exact_original_feedback_plus_zero_command_residual",
        previous_actor_mean_Adam="discarded_in_this_new_branch_parent_checkpoint_retained",
        critic_and_Adam="retained_exact", exploration_std_and_Adam="retained_exact_and_frozen",
        saved_RNG="retained_for_restore_before_new_physical_reset", previous_consumption="retained_without_refund",
        active_branch_actor_updates=0)
    return adapted, proof


def mean_from_checkpoint(checkpoint, source):
    """Load only a declared mean; reject changed original feedback or buffers."""
    kind = checkpoint["infos"].get("actor_adaptation")
    from .guarded_feedback_residual import GuardedFeedbackActor, KIND as FEEDBACK_KIND, TRACKING_KIND
    from .separated_feedback import SeparatedFeedbackActor, KIND as SEPARATED_KIND
    from .low_demand_feedback import LowDemandFeedbackActor, KIND as LOW_DEMAND_KIND
    from .static_feedback import StaticFeedbackActor, KIND as STATIC_KIND
    from .wheel_authority import WheelAuthorityActor, KIND as WHEEL_KIND
    from .static_grip import StaticGripActor, KIND as GRIP_KIND
    from .motion_grip import MotionGripActor, KIND as MOTION_GRIP_KIND
    from .speed_reference import SpeedReferenceActor, KIND as SPEED_REFERENCE_KIND
    if kind == SPEED_REFERENCE_KIND:
        from pathlib import Path
        import hashlib
        settings = checkpoint['infos'].get('speed_reference')
        root = Path(__file__).resolve().parents[2]
        if not isinstance(settings, dict) or set(settings) != {'checkpoint', 'sha256', 'margin_m_s', 'lean_gain_rad_per_m_s'}:
            raise ValueError('Speed reference requires the declared authenticated v7 parent and initialization')
        path = root / settings['checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings['sha256']:
            raise ValueError('Speed-reference parent checkpoint changed')
        parent = torch.load(path, map_location='cpu', weights_only=True)
        mean = SpeedReferenceActor(mean_from_checkpoint(parent, source), settings['sha256'],
            settings['margin_m_s'], settings['lean_gain_rad_per_m_s'])
        mean.load_state_dict({n.removeprefix('mlp.'): value for n, value in checkpoint['actor_state_dict'].items()
                              if n.startswith('mlp.')}, strict=True)
        mean.verify_original_from_source_full(source)
        return mean.eval()
    if kind == MOTION_GRIP_KIND:
        from pathlib import Path
        import hashlib
        settings = checkpoint['infos'].get('motion_grip_parent')
        root = Path(__file__).resolve().parents[2]
        if (not isinstance(settings, dict) or set(settings) != {'checkpoint', 'sha256'}
                or not isinstance(settings['checkpoint'], str)):
            raise ValueError('Motion grip requires the declared authenticated v7 parent')
        path = root / settings['checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings['sha256']:
            raise ValueError('Motion grip parent checkpoint changed')
        parent = torch.load(path, map_location='cpu', weights_only=True)
        if parent['infos'].get('actor_adaptation') != GRIP_KIND:
            raise ValueError('Motion grip parent must be the declared v7 checkpoint')
        mean = MotionGripActor(mean_from_checkpoint(parent, source), settings['sha256'])
        mean.load_state_dict({n.removeprefix('mlp.'): value
            for n, value in checkpoint['actor_state_dict'].items() if n.startswith('mlp.')}, strict=True)
        mean.verify_original_from_source_full(source)
        return mean.eval()
    if kind not in (None, KIND, FEEDBACK_KIND, TRACKING_KIND, SEPARATED_KIND, LOW_DEMAND_KIND, STATIC_KIND, WHEEL_KIND, GRIP_KIND):
        raise ValueError("Unknown checkpoint actor adaptation")
    original, _ = recover_actor(source)
    mean = (GuardedFeedbackActor(CommandResidualActor(original), tracking_gate=kind == TRACKING_KIND)
            if kind in (FEEDBACK_KIND, TRACKING_KIND)
            else CommandResidualActor(original) if kind == KIND else original)
    state = {n.removeprefix("mlp."): v for n, v in checkpoint["actor_state_dict"].items() if n.startswith("mlp.")}
    if kind in (SEPARATED_KIND, LOW_DEMAND_KIND, STATIC_KIND, WHEEL_KIND, GRIP_KIND):
        mean = SeparatedFeedbackActor(GuardedFeedbackActor(CommandResidualActor(original), tracking_gate=True))
    if kind in (LOW_DEMAND_KIND, STATIC_KIND, WHEEL_KIND, GRIP_KIND):
        mean = LowDemandFeedbackActor(mean)
    if kind in (STATIC_KIND, WHEEL_KIND, GRIP_KIND):
        from pathlib import Path
        import hashlib
        settings = checkpoint["infos"].get("static_feedback_recall")
        root = Path(__file__).resolve().parents[2]
        path = root / settings["checkpoint"] if isinstance(settings, dict) else None
        if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != settings.get("sha256"):
            raise ValueError("Static feedback reference checkpoint changed")
        recalled = mean_from_checkpoint(torch.load(path, map_location="cpu", weights_only=True), source)
        mean.load_state_dict({n:v for n,v in state.items() if not n.startswith(("static_feedback.","wheel_authority.")) and n != "authority_bound"},strict=True)
        mean = StaticFeedbackActor(mean, recalled, settings["sha256"])
        if kind in (WHEEL_KIND, GRIP_KIND):
            mean = WheelAuthorityActor(mean)
        if kind == GRIP_KIND:
            mean = StaticGripActor(mean)
    if kind in (KIND, FEEDBACK_KIND, TRACKING_KIND, SEPARATED_KIND, LOW_DEMAND_KIND, STATIC_KIND, WHEEL_KIND, GRIP_KIND):
        mean.load_state_dict(state, strict=True)
        mean.verify_original(original)
    else:
        mean.mlp.load_state_dict(state, strict=True)
    return mean.eval()
