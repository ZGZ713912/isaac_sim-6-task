"""Keep low-demand feedback available without encoder-dependent attenuation."""
from copy import deepcopy
import json
import torch
from torch import nn
from .guarded_feedback_residual import GuardedFeedbackActor
from .separated_feedback import SeparatedFeedbackActor

KIND = 'original-command-and-continuous-low-feedback-residual-v4'


class LowDemandFeedbackActor(SeparatedFeedbackActor):
    """Same parameters and bounds as v3; only its visible-input gate changes."""
    def __init__(self, parent):
        if type(parent) is not SeparatedFeedbackActor:
            raise ValueError('Low-demand gate requires the exact separated feedback parent')
        nn.Module.__init__(self)
        self.command = deepcopy(parent.command)
        self.tracking_gate = True
        for name in ('input_center', 'input_scale', 'output_scale'):
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.feedback = deepcopy(parent.feedback)
        self.fast_feedback = deepcopy(parent.fast_feedback)
        self.set_trainable(True)

    @staticmethod
    def tracking_feedback_gate(obs):
        old = GuardedFeedbackActor.tracking_feedback_gate(obs)
        demand = torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5)
        return torch.maximum(old, (2. - demand).clamp(0., 1.))

    def verify_original(self, original):
        result = super().verify_original(original)
        result['separated_feedback']['kind'] = KIND
        result['low_demand_gate'] = dict(version=1,
            expression='max(previous_tracking_gate,clamp(2-max(abs(vx)/.75,abs(yaw)/1.5),0,1))',
            unchanged_static_and_demand_at_least2=True,
            parameters_Adam_and_action_bounds='unchanged',
            initial_mean_is_identity=False)
        return result

    def export_graph(self, source):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        model = super().export_graph(source)
        prefix = 'gated_feedback.'
        corrections = [i for i, n in enumerate(model.graph.node)
                       if list(n.output) == [prefix + 'correction']]
        if len(corrections) != 1:
            raise ValueError('Previous feedback export changed its correction node')
        index = corrections[0]
        correction = model.graph.node[index]
        if correction.input[0] != prefix + 'tracking_gate':
            raise ValueError('Expected the unchanged v3 tracking gate')
        model.graph.initializer.append(numpy_helper.from_array(np.array(2., dtype=np.float32), prefix + 'two'))
        added = [
            helper.make_node('Sub', [prefix + 'two', prefix + 'demand'], [prefix + 'low_floor_unclipped']),
            helper.make_node('Clip', [prefix + 'low_floor_unclipped', prefix + 'zero', prefix + 'one'],
                             [prefix + 'low_floor']),
            helper.make_node('Max', [prefix + 'tracking_gate', prefix + 'low_floor'], [prefix + 'continuous_gate']),
        ]
        nodes = [deepcopy(n) for n in model.graph.node]
        nodes[index].input[0] = prefix + 'continuous_gate'
        del model.graph.node[:]
        model.graph.node.extend(nodes[:index] + added + nodes[index:])
        metadata = {p.key: p.value for p in model.metadata_props}
        metadata['actor_adaptation'] = KIND
        metadata['separated_feedback'] = json.dumps(self.verify_original_from_source(source), separators=(',', ':'))
        metadata['low_demand_gate'] = json.dumps(self.verify_original_from_source_full(source)['low_demand_gate'],
                                                  separators=(',', ':'))
        helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model

    def verify_original_from_source_full(self, source):
        from .onnx_actor_import import recover_actor
        original, _ = recover_actor(source)
        return self.verify_original(original)


def configure_low_demand_feedback_training(actor, enabled):
    if type(actor.mlp) is not LowDemandFeedbackActor:
        raise ValueError('Actor does not contain the declared low-demand gate')
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    prefixes = ('mlp.feedback.', 'mlp.fast_feedback.')
    return dict(enabled=enabled,
        trainable_names=[n for n, p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict() if not n.startswith(prefixes)])
