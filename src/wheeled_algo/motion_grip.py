"""Unadopted identity revision: adapt existing motion heads on low friction."""
from copy import deepcopy
import json

import torch
from torch import nn

from wheeled_algo.static_grip import StaticGripActor
from wheeled_algo.wheel_authority import WheelAuthorityActor

KIND = 'original-static-grip-with-motion-feedback-adaptation-v8'
BRANCHES = ('feedback', 'fast_feedback', 'wheel_authority')


class MotionGripActor(StaticGripActor):
    """Retain the adapted static head and open only existing moving heads."""

    def __init__(self, parent, parent_sha256):
        if type(parent) is not StaticGripActor:
            raise ValueError('Motion grip adaptation requires the exact v7 parent')
        if (len(parent_sha256) != 64
                or any(char not in '0123456789abcdef' for char in parent_sha256)):
            raise ValueError('Motion grip requires its authenticated parent checkpoint')
        nn.Module.__init__(self)
        for name, module in parent.named_children():
            setattr(self, name, deepcopy(module))
        for name, value in parent.named_buffers(recurse=False):
            self.register_buffer(name, value.detach().clone())
        self.tracking_gate = True
        self.recalled_sha256 = parent.recalled_sha256
        self.motion_parent_sha256 = parent_sha256
        self._expected_static = {
            name: value.detach().cpu().clone() for name, value in parent.static_feedback.state_dict().items()}
        self.set_trainable(True)

    def set_trainable(self, enabled, *, train_command=False):
        if train_command:
            raise ValueError('Motion grip adaptation freezes the command branch')
        self.requires_grad_(False)
        for name in BRANCHES:
            getattr(self, name).requires_grad_(enabled)
        for parameter in self.parameters():
            parameter.grad = None

    def static_feedback_contract(self):
        if any(not torch.equal(value.detach().cpu(), self._expected_static[name])
               for name, value in self.static_feedback.state_dict().items()):
            raise ValueError('Frozen adapted static feedback changed')
        return dict(checkpoint_sha256=self.motion_parent_sha256,
            reference_role='exact adapted static tensors of the authenticated v7 parent',
            static_parameters=sum(p.numel() for p in self.static_feedback.parameters()),
            static_trainable=False, moving_full_vx_m_s=.1, moving_full_yaw_rad_s=.2,
            stationary_action_function='same parent at zero command and encoder rim speed at most .4m/s',
            stopping='moving-zero-command wheel authority remains trainable; native replay required')

    def verify_original(self, original):
        result = super().verify_original(original)
        result['separated_feedback']['kind'] = KIND
        result.pop('static_grip_adaptation')
        result['motion_grip_adaptation'] = dict(kind=KIND, initial_mean_is_identity=True,
            added_parameters=0, parent_checkpoint_sha256=self.motion_parent_sha256,
            trainable_branches=list(BRANCHES), frozen_branches=['command', 'static_feedback'],
            trainable_parameters=sum(p.numel() for name in BRANCHES for p in getattr(self, name).parameters()),
            preservation_scope='static head tensors and low-encoder zero-command action function; all closed-loop behavior requires paired replay')
        return result

    def export_graph(self, source):
        import onnx
        model = WheelAuthorityActor.export_graph(self, source)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        metadata['actor_adaptation'] = KIND
        metadata['motion_grip_adaptation'] = json.dumps(
            self.verify_original_from_source_full(source)['motion_grip_adaptation'], separators=(',', ':'))
        onnx.helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model


def configure_motion_grip_training(actor, enabled):
    if type(actor.mlp) is not MotionGripActor:
        raise ValueError('Actor does not contain the declared motion grip adaptation')
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    prefixes = tuple('mlp.' + name + '.' for name in BRANCHES)
    return dict(enabled=enabled,
        trainable_names=[name for name, p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[name for name in actor.state_dict() if not name.startswith(prefixes)])
