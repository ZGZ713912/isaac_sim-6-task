"""Explicit static-only adaptation of v6, with identical tensors and forward math."""
from copy import deepcopy
import json

import torch
from torch import nn

from .wheel_authority import WheelAuthorityActor

KIND = 'original-wheel-authority-with-static-grip-adaptation-v7'


class StaticGripActor(WheelAuthorityActor):
    """Train the existing stationary head while retaining every moving head."""

    def __init__(self, parent):
        if type(parent) is not WheelAuthorityActor:
            raise ValueError('Static grip adaptation requires the exact v6 parent')
        nn.Module.__init__(self)
        for name, module in parent.named_children():
            setattr(self, name, deepcopy(module))
        for name, value in parent.named_buffers(recurse=False):
            self.register_buffer(name, value.detach().clone())
        self.tracking_gate = True
        self.recalled_sha256 = parent.recalled_sha256
        self._expected_static = deepcopy(parent._expected_static)
        self.set_trainable(True)

    def set_trainable(self, enabled, *, train_command=False):
        if train_command:
            raise ValueError('Static grip adaptation freezes the command branch')
        self.requires_grad_(False)
        self.static_feedback.requires_grad_(enabled)
        for parameter in self.parameters():
            parameter.grad = None

    def static_feedback_contract(self):
        if any(not torch.isfinite(v).all() for v in self.static_feedback.state_dict().values()):
            raise ValueError('Static grip adaptation contains nonfinite static tensors')
        return dict(checkpoint_sha256=self.recalled_sha256,
            reference_role='initial static snapshot; current tensors may be adapted',
            static_parameters=sum(p.numel() for p in self.static_feedback.parameters()),
            static_trainable=True, moving_full_vx_m_s=.1, moving_full_yaw_rad_s=.2,
            stationary_action_function='adapted existing bounded static feedback and frozen command branch',
            moving_action_function='same v6 function at full moving endpoint for identical observations',
            stopping='adapted zero-command feedback; native stop replay required')

    def verify_original(self, original):
        result = super().verify_original(original)
        result['separated_feedback']['kind'] = KIND
        result['static_grip_adaptation'] = dict(kind=KIND,
            initial_mean_is_identity=True, added_parameters=0,
            trainable_branch='static_feedback', moving_heads_and_wheel_authority='frozen',
            static_gate='1-clamp(max(abs(vx)/.1,abs(yaw)/.2),0,1)',
            preservation_scope='same observations at full moving mix; closed-loop motion and stops require replay')
        return result

    def export_graph(self, source):
        import onnx
        model = super().export_graph(source)
        metadata = {p.key: p.value for p in model.metadata_props}
        metadata['actor_adaptation'] = KIND
        metadata['static_grip_adaptation'] = json.dumps(
            self.verify_original_from_source_full(source)['static_grip_adaptation'], separators=(',', ':'))
        onnx.helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model


def configure_static_grip_training(actor, enabled):
    if type(actor.mlp) is not StaticGripActor:
        raise ValueError('Actor does not contain the declared static grip adaptation')
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    return dict(enabled=enabled,
        trainable_names=[name for name, p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[name for name in actor.state_dict() if not name.startswith('mlp.static_feedback.')])
