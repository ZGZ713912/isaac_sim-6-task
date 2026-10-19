"""Unadopted prototype: separate low-command and faster-motion feedback weights."""
from copy import deepcopy
import torch
from torch import nn
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor


class SeparatedFeedbackCandidate(GuardedFeedbackActor):
    def __init__(self, parent):
        nn.Module.__init__(self)
        if type(parent) is not GuardedFeedbackActor or not parent.tracking_gate:
            raise ValueError('Prototype requires the exact tracking-feedback parent')
        self.command = deepcopy(parent.command)
        self.tracking_gate = True
        for name in ('input_center', 'input_scale', 'output_scale'):
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.feedback = deepcopy(parent.feedback)
        self.fast_feedback = deepcopy(parent.feedback)
        self.set_trainable(True, train_command=False)

    @staticmethod
    def fast_mix(obs):
        # Low commands have their own weights. The interpolation keeps the
        # original action bound; it reaches the fast branch before teacher selection.
        demand = torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5)
        return (demand - 1.).clamp(0., 1.)

    def feedback_actions(self, obs):
        features = (obs[:, :28] - self.input_center) * self.input_scale
        slow = self.output_scale * torch.tanh(self.feedback(features))
        fast = self.output_scale * torch.tanh(self.fast_feedback(features))
        mix = self.fast_mix(obs)
        raw = slow + mix * (fast - slow)
        raw = torch.where(mix >= 1., fast, raw)
        return self.tracking_feedback_gate(obs) * raw

    def set_trainable(self, enabled, *, train_command=False):
        if train_command:
            raise ValueError('The unadopted prototype only trains feedback weights')
        self.command.set_trainable(False)
        for branch in (self.feedback, self.fast_feedback):
            branch.requires_grad_(enabled)
            for parameter in branch.parameters():
                parameter.grad = None

    def export_graph(self, source):
        raise ValueError('Unadopted separated-feedback prototype has no approved ONNX export')
