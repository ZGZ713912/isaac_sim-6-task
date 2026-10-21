"""Keep the parent's measured ONNX arithmetic outside the motion gate.

This export-only composition is numerically checked against the training mean.
It prevents retracing the low-speed parent when adding the new motion graph.
"""
from copy import deepcopy
import io
import json

import torch
from torch import nn

from .directional_motion import DirectionalMotionActor
from .motion_state_residual import MotionStateResidualActor


class MotionSelector(nn.Module):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = deepcopy(settings)

    def forward(self, obs):
        gate = MotionStateResidualActor.motion_gate(obs)
        if self.settings is not None:
            from .transient_motion import transient_gate
            gate = transient_gate(obs, gate, self.settings)
        return gate > 0.


class PreservedDirectionalRuntime(nn.Module):
    def __init__(self, parent, candidate):
        super().__init__()
        if type(candidate) is not DirectionalMotionActor:
            raise ValueError('Preserved directional runtime requires the declared v12 mean')
        if candidate.leg_feedback_settings is not None:
            raise ValueError('Parent-leg recurrence changes outside the selector; export the declared complete mean')
        self.parent = deepcopy(parent).eval().requires_grad_(False)
        self.candidate = deepcopy(candidate).eval().requires_grad_(False)

    def forward(self, obs):
        return torch.where(MotionSelector(self.candidate.transient_settings)(obs), self.candidate(obs), self.parent(obs))

    def export_graph(self, source):
        import onnx
        from onnx import helper
        parent = self.parent.export_graph(source)
        candidate = self.candidate.export_graph(source)
        buffer = io.BytesIO()
        torch.onnx.export(MotionSelector(self.candidate.transient_settings), torch.zeros(1, 35), buffer,
            input_names=['obs'], output_names=['active'], dynamic_axes={'obs': {0: 'batch'}, 'active': {0: 'batch'}},
            opset_version=17, dynamo=False)
        selector = onnx.load_model_from_string(buffer.getvalue())
        models = (parent, candidate, selector)
        expected_opsets = [(x.domain, x.version) for x in parent.opset_import]
        if any([(x.domain, x.version) for x in m.opset_import] != expected_opsets for m in models):
            raise ValueError('Directional graphs use different operators')
        graph = deepcopy(parent)
        nodes, constants, outputs = [], [], []
        input_name = parent.graph.input[0].name
        for prefix, model in zip(('directional.parent.', 'directional.motion.', 'directional.selector.'), models, strict=True):
            old_input = model.graph.input[0].name
            def rename(name):
                return input_name if name == old_input else prefix + name if name else ''
            outputs.append(rename(model.graph.output[0].name))
            for index, node in enumerate(model.graph.node):
                copied = deepcopy(node)
                copied.name = prefix + str(index)
                copied.input[:] = [rename(n) for n in node.input]
                copied.output[:] = [rename(n) for n in node.output]
                nodes.append(copied)
            for value in model.graph.initializer:
                copied = deepcopy(value)
                copied.name = rename(value.name)
                constants.append(copied)
        nodes.append(helper.make_node('Where', [outputs[2], outputs[1], outputs[0]], [parent.graph.output[0].name]))
        del graph.graph.node[:]
        graph.graph.node.extend(nodes)
        del graph.graph.initializer[:]
        graph.graph.initializer.extend(constants)
        del graph.graph.value_info[:]
        metadata = {p.key: p.value for p in candidate.metadata_props}
        metadata['parent_graph_preservation'] = json.dumps(dict(
            selection=('transient error/pitch or inherited motion gate > 0'
                       if self.candidate.transient_settings is not None else 'original motion gate > 0'),
            outside_gate='unchanged measured parent ONNX nodes',
            numerical_scope='pointwise original graph; native episode retention still required'), separators=(',', ':'))
        helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph, full_check=True)
        return graph


def preserve_directional_graph(mean, source, parent_file=None):
    if type(mean) is not DirectionalMotionActor:
        return mean
    if mean.leg_feedback_settings is not None:
        # There is no unchanged quiet-parent endpoint for this declared
        # recurrence change. Its complete mean has its own numerical export.
        return mean
    from pathlib import Path
    from .command_residual import mean_from_checkpoint
    if parent_file is None:
        raise ValueError('Directional export requires its checkpoint-declared parent path')
    root = Path(__file__).resolve().parents[2]
    parent_file = root / parent_file
    import hashlib
    if hashlib.sha256(parent_file.read_bytes()).hexdigest() != mean.directional_parent_sha256:
        raise ValueError('Directional export parent changed')
    parent = mean_from_checkpoint(torch.load(parent_file, map_location='cpu', weights_only=True), source)
    return PreservedDirectionalRuntime(parent, mean)
