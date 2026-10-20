"""Six-parameter high-speed conditioning above an authenticated v7 policy.

The requested command and reward keep their original meaning. Only the frozen
policy's internal reference is limited by observable rolling speed and gyro.
"""
from copy import deepcopy
import io
import json

import torch
from torch import nn

from .static_grip import StaticGripActor

KIND = "original-static-grip-with-reachable-speed-reference-v9"


def condition_reference(obs, margin, lean_gain=None):
    gate = (torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5) - 1.).clamp(0., 1.)
    speed = (obs[:, 20:21] - obs[:, 21:22]) * .3
    yaw = obs[:, 6:7] * 2.
    vx = speed + (obs[:, :1] - speed).clamp(-margin, margin)
    yaw_margin = margin / (.4373 / 2.)
    wz = yaw + (obs[:, 2:3] - yaw).clamp(-yaw_margin, yaw_margin)
    reference = torch.cat((vx.clamp(-3., 3.), obs[:, 1:2], wz.clamp(-4., 4.), obs[:, 3:]), -1)
    reference = torch.where(gate <= 0., obs, obs + gate * (reference - obs))
    if lean_gain is not None:
        angle = (lean_gain * (obs[:, :1] - speed)).clamp(-.05, .05) * gate
        gx, gz = reference[:, 7:8], reference[:, 9:10]
        reference = torch.cat((reference[:, :7], gx*angle.cos()-gz*angle.sin(), reference[:, 8:9],
                               gx*angle.sin()+gz*angle.cos(), reference[:, 10:]), -1)
    return reference


class SpeedReferenceActor(StaticGripActor):
    def __init__(self, parent, parent_sha256, margins, lean_gains=(0., 0.)):
        if type(parent) is not StaticGripActor:
            raise ValueError("Speed reference requires the exact authenticated v7 parent")
        if len(parent_sha256) != 64 or any(c not in "0123456789abcdef" for c in parent_sha256):
            raise ValueError("Invalid speed-reference parent SHA256")
        values = torch.tensor(margins, dtype=torch.float32)
        if values.shape != (4,) or not torch.isfinite(values).all() or (values < .05).any() or (values > 1.2).any():
            raise ValueError("Four reference margins in [.05,1.2] m/s are required")
        nn.Module.__init__(self)
        for name, module in parent.named_children():
            setattr(self, name, deepcopy(module))
        for name, value in parent.named_buffers(recurse=False):
            self.register_buffer(name, value.detach().clone())
        self.reference_margin = nn.Parameter(values)
        gains = torch.tensor(lean_gains, dtype=torch.float32)
        if gains.shape != (2,) or not torch.isfinite(gains).all() or (gains.abs() > .04).any():
            raise ValueError("Two finite lean gains within +/-.04 rad/(m/s) are required")
        self.reference_lean_gain = nn.Parameter(gains)
        self.tracking_gate = True
        self.recalled_sha256 = parent.recalled_sha256
        self.reference_parent_sha256 = parent_sha256
        self._expected_parent = {n: v.detach().cpu().clone() for n, v in parent.state_dict().items()}
        self._expected_static = {n: v.detach().cpu().clone() for n, v in parent.static_feedback.state_dict().items()}
        self.set_trainable(True)

    def forward(self, obs):
        translation = obs[:, :1].abs() / .75 >= obs[:, 2:3].abs() / 1.5
        index = torch.where(translation, (obs[:, :1] < 0.).long(), 2 + (obs[:, 2:3] < 0.).long())
        margin = self.reference_margin.clamp(.05, 1.2)[index]
        lean = self.reference_lean_gain.clamp(-.04, .04)[(obs[:, :1] < 0.).long()]
        return super().forward(condition_reference(obs, margin, lean))

    def set_trainable(self, enabled, *, train_command=False):
        if train_command:
            raise ValueError("Speed-reference adaptation freezes the entire parent")
        self.requires_grad_(False)
        self.reference_margin.requires_grad_(enabled)
        self.reference_lean_gain.requires_grad_(enabled)
        for parameter in self.parameters():
            parameter.grad = None

    def static_feedback_contract(self):
        return dict(checkpoint_sha256=self.reference_parent_sha256, static_trainable=False,
            stationary_action_function="exact authenticated v7 for identical low-command observations")

    def verify_original(self, original):
        current = self.state_dict()
        if any(not torch.equal(current[n].detach().cpu(), value) for n, value in self._expected_parent.items()):
            raise ValueError("Frozen speed-reference parent changed")
        if not torch.isfinite(self.reference_margin).all() or not torch.isfinite(self.reference_lean_gain).all():
            raise ValueError("Nonfinite speed-reference margins")
        result = super().verify_original(original)
        result.pop("static_grip_adaptation")
        result["separated_feedback"]["kind"] = KIND
        result["speed_reference"] = dict(kind=KIND, parent_checkpoint_sha256=self.reference_parent_sha256,
            trainable_parameters=6, order=["forward", "backward", "yaw_positive", "yaw_negative"],
            margin_m_s=self.reference_margin.clamp(.05, 1.2).detach().cpu().tolist(),
            margin_bounds_m_s=[.05, 1.2], low_command_exact_limits=[.75, 1.5],
            lean_gain_rad_per_m_s=self.reference_lean_gain.clamp(-.04, .04).detach().cpu().tolist(),
            lean_reference_abs_bound_rad=.05,
            feedback="wheel encoders and IMU gyro only; no privileged velocity",
            scoring="external true command unchanged; body velocity and survival must be measured",
            initial_mean_is_identity=False)
        return result

    def export_graph(self, source):
        import onnx
        from .onnx_actor_import import recover_actor
        original, _ = recover_actor(source)
        declaration = self.verify_original(original)
        stream = io.BytesIO()
        torch.onnx.export(deepcopy(self).cpu().eval(), torch.zeros(2, 35), stream,
            input_names=["obs"], output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17, dynamo=False)
        model = onnx.load_model_from_string(stream.getvalue())
        metadata = {p.key: p.value for p in onnx.load(str(source)).metadata_props}
        metadata.update(actor_adaptation=KIND, speed_reference=json.dumps(declaration["speed_reference"], separators=(",", ":")))
        onnx.helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model


def configure_speed_reference_training(actor, enabled):
    if type(actor.mlp) is not SpeedReferenceActor:
        raise ValueError("Missing declared reachable-reference actor")
    actor.requires_grad_(False)
    actor.mlp.set_trainable(enabled)
    return dict(enabled=enabled, trainable_names=[n for n,p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict() if n not in ("mlp.reference_margin", "mlp.reference_lean_gain")])


def extend_speed_reference_parent(parent, actor, critic, optimizer, old_names):
    from .static_grip import KIND as PARENT_KIND
    if type(actor.mlp) is not SpeedReferenceActor or parent["infos"]["actor_adaptation"] != PARENT_KIND:
        raise ValueError("Require the explicit v7 to speed-reference revision")
    old, new = parent["actor_state_dict"], deepcopy(actor.state_dict())
    added = {"mlp.reference_margin", "mlp.reference_lean_gain"}
    if set(new) - set(old) != added or not set(old) <= set(new):
        raise ValueError("Only six speed-reference parameters may be added")
    for name, value in old.items():
        if name not in new or new[name].shape != value.shape:
            raise ValueError("Parent actor ownership changed")
        new[name] = value.clone()
    names = list(dict(actor.named_parameters()))
    if old_names != [n for n in names if n not in added]:
        raise ValueError("Parent actor parameter order changed")
    if set(critic.state_dict()) != set(parent["critic_state_dict"]) or any(
            v.shape != parent["critic_state_dict"][n].shape for n,v in critic.state_dict().items()):
        raise ValueError("Critic architecture changed")
    old_opt, new_opt = parent["optimizer_state_dict"], deepcopy(optimizer.state_dict())
    if len(old_opt["param_groups"]) != 2 or len(new_opt["param_groups"]) != 2:
        raise ValueError("Separate actor and critic Adam groups are required")
    old_ids = dict(zip(old_names, old_opt["param_groups"][0]["params"], strict=True))
    new_ids = dict(zip(names, new_opt["param_groups"][0]["params"], strict=True))
    new_opt["state"] = {new_ids[n]: deepcopy(old_opt["state"][i]) for n,i in old_ids.items() if i in old_opt["state"]}
    for a,b in zip(old_opt["param_groups"][1]["params"], new_opt["param_groups"][1]["params"], strict=True):
        if a in old_opt["state"]:
            new_opt["state"][b] = deepcopy(old_opt["state"][a])
    for index in (0,1):
        ids = new_opt["param_groups"][index]["params"]
        new_opt["param_groups"][index] = deepcopy(old_opt["param_groups"][index])
        new_opt["param_groups"][index]["params"] = ids
    state = deepcopy(parent)
    state["actor_state_dict"], state["optimizer_state_dict"] = new, new_opt
    return state, dict(initial_mean_is_identity=False, existing_actor_std_critic_Adam_RNG_counters="retained",
        added_parameters=6, new_Adam_slots="empty", parent_low_command_function="exact; native regression still required")
