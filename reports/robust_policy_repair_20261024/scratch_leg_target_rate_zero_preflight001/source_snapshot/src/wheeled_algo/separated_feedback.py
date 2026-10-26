"""Identity fork for low-command and faster-motion feedback learning.

The two bounded feedback branches initially copy the same parent. Native PPO
still clips their combined actor gradient, so direct gradient separation is
not a claim that all indirect learning or closed-loop interference disappears.
"""
from copy import deepcopy
import json

import torch
from torch import nn
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor, TRACKING_KIND

KIND = 'original-command-and-separated-tracking-feedback-residual-v3'


class SeparatedFeedbackActor(GuardedFeedbackActor):
    def __init__(self, parent):
        nn.Module.__init__(self)
        if type(parent) is not GuardedFeedbackActor or not parent.tracking_gate:
            raise ValueError('Separated feedback requires the exact tracking-feedback parent')
        self.command = deepcopy(parent.command)
        self.tracking_gate = True
        for name in ('input_center', 'input_scale', 'output_scale'):
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.feedback = deepcopy(parent.feedback)
        self.fast_feedback = deepcopy(parent.feedback)
        self.set_trainable(True)

    @staticmethod
    def fast_mix(obs):
        demand = torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5)
        return (demand - 1.).clamp(0., 1.)

    def feedback_actions(self, obs):
        features = (obs[:, :28] - self.input_center) * self.input_scale
        low = self.output_scale * torch.tanh(self.feedback(features))
        fast = self.output_scale * torch.tanh(self.fast_feedback(features))
        mix = self.fast_mix(obs)
        # Exact parent output for identical branches; exact fast endpoint even
        # after they diverge. Convex mixing preserves the original action bound.
        raw = low + mix * (fast - low)
        raw = torch.where(mix >= 1., fast, raw)
        return self.tracking_feedback_gate(obs) * raw

    def set_trainable(self, enabled, *, train_command=False):
        if train_command:
            raise ValueError('Separated feedback freezes the learned command branch')
        self.command.set_trainable(False)
        for branch in (self.feedback, self.fast_feedback):
            branch.requires_grad_(enabled)
            for parameter in branch.parameters():
                parameter.grad = None

    def verify_original(self, original):
        result = super().verify_original(original)
        result.update(feedback_parameters=sum(p.numel() for p in self.feedback.parameters()) +
            sum(p.numel() for p in self.fast_feedback.parameters()),
            separated_feedback=dict(kind=KIND, low_endpoint_max_abs_vx_m_s=.75,
                low_endpoint_max_abs_yaw_rad_s=1.5, fast_endpoint_abs_vx_m_s=1.5,
                fast_endpoint_abs_yaw_rad_s=3., interpolation='clamp(max(abs(vx)/.75,abs(yaw)/1.5)-1,0,1)',
                direct_high_teacher_gradient_to_low_branch='zero', native_actor_gradient_clip='joint; unchanged'))
        return result

    def export_graph(self, source):
        """Append the fast clone and convex blend, retaining every source tensor."""
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        # The established exporter writes the low branch and tracking gate.
        model = GuardedFeedbackActor.export_graph(self, source)
        prefix = 'gated_feedback.'
        if any(value.name.startswith(prefix + 'fast_feedback.') for value in model.graph.initializer):
            raise ValueError('Source already contains separated feedback')
        names = lambda *values: [prefix + value for value in values]
        model.graph.initializer.extend(numpy_helper.from_array(value.detach().cpu().numpy().copy(),
            prefix + 'fast_feedback.' + name) for name,value in self.fast_feedback.state_dict().items())
        offsets = [node for node in model.graph.node if list(node.output) == names('offset')]
        corrections = [i for i,node in enumerate(model.graph.node) if list(node.output) == names('correction')]
        if len(offsets) != 1 or len(corrections) != 1:
            raise ValueError('Established feedback ONNX graph shape changed')
        offsets[0].output[0] = prefix + 'low_offset'
        added = [
            helper.make_node('Gemm', names('normalized', 'fast_feedback.0.weight', 'fast_feedback.0.bias'), names('fast_hidden0'), transB=1),
            helper.make_node('Elu', names('fast_hidden0'), names('fast_hidden'), alpha=1.),
            helper.make_node('Gemm', names('fast_hidden', 'fast_feedback.2.weight', 'fast_feedback.2.bias'), names('fast_raw'), transB=1),
            helper.make_node('Tanh', names('fast_raw'), names('fast_bounded')),
            helper.make_node('Mul', names('fast_bounded', 'output_scale'), names('fast_offset')),
            helper.make_node('Sub', names('demand', 'one'), names('mix_unclipped')),
            helper.make_node('Clip', names('mix_unclipped', 'zero', 'one'), names('fast_mix')),
            helper.make_node('Sub', names('fast_offset', 'low_offset'), names('offset_difference')),
            helper.make_node('Mul', names('fast_mix', 'offset_difference'), names('mixed_difference')),
            helper.make_node('Add', names('low_offset', 'mixed_difference'), names('interpolated_offset')),
            helper.make_node('Equal', names('fast_mix', 'one'), names('fast_endpoint')),
            helper.make_node('Where', names('fast_endpoint', 'fast_offset', 'interpolated_offset'), names('offset')),
        ]
        index = corrections[0]
        nodes = [deepcopy(node) for node in model.graph.node]
        del model.graph.node[:]
        model.graph.node.extend(nodes[:index] + added + nodes[index:])
        metadata = {entry.key:entry.value for entry in model.metadata_props}
        metadata.update(actor_adaptation=KIND,
            separated_feedback=json.dumps(self.verify_original_from_source(source), separators=(',', ':')))
        helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model

    def verify_original_from_source(self, source):
        from wheeled_algo.onnx_actor_import import recover_actor
        original, _ = recover_actor(source)
        return self.verify_original(original)['separated_feedback']


def configure_separated_feedback_training(actor, enabled):
    if type(actor.mlp) is not SeparatedFeedbackActor:
        raise ValueError('Actor does not contain the declared separated feedback')
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    prefixes = ('mlp.feedback.', 'mlp.fast_feedback.')
    return dict(enabled=enabled,
        trainable_names=[n for n,p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict() if not n.startswith(prefixes)])


def extend_separated_feedback_parent(parent, actor, critic, optimizer, old_actor_names):
    """Duplicate only feedback weights/Adam; preserve every existing state slot.

    All inherited moments and their step are cloned for the fast branch. The
    existing actor/critic parameter identifiers are explicitly remapped, not
    interpreted by positional coincidence. RNG and counters are copied intact.
    """
    if type(actor.mlp) is not SeparatedFeedbackActor or parent['infos'].get('actor_adaptation') != TRACKING_KIND:
        raise ValueError('Separated identity extension requires the tracking-feedback parent')
    old_actor = parent['actor_state_dict']
    new_actor = deepcopy(actor.state_dict())
    extra = set(new_actor) - set(old_actor)
    expected_extra = {'mlp.fast_feedback.' + name for name in actor.mlp.fast_feedback.state_dict()}
    if extra != expected_extra or len(extra) != 4:
        raise ValueError('Separated extension must add exactly four feedback tensors')
    for name,value in old_actor.items():
        if name not in new_actor or new_actor[name].shape != value.shape:
            raise ValueError('Separated extension changes an existing actor tensor: ' + name)
        new_actor[name] = value.clone()
    for name in extra:
        new_actor[name] = old_actor[name.replace('mlp.fast_feedback.', 'mlp.feedback.', 1)].clone()
    old_opt, new_opt = parent['optimizer_state_dict'], deepcopy(optimizer.state_dict())
    if (len(old_opt['param_groups']) != 2 or len(new_opt['param_groups']) != 2
            or len(old_opt['param_groups'][0]['params']) != len(old_actor_names)
            or set(critic.state_dict()) != set(parent['critic_state_dict'])
            or any(critic.state_dict()[n].shape != value.shape for n,value in parent['critic_state_dict'].items())):
        raise ValueError('Separated extension changes Adam ownership or critic architecture')
    new_names = list(dict(actor.named_parameters()))
    if old_actor_names != [name for name in new_names if not name.startswith('mlp.fast_feedback.')]:
        raise ValueError('Parent actor parameter order differs from the declared identity extension')
    old_ids = dict(zip(old_actor_names, old_opt['param_groups'][0]['params'], strict=True))
    new_ids = dict(zip(new_names, new_opt['param_groups'][0]['params'], strict=True))
    if set(new_names) - set(old_actor_names) != extra or len(set(new_names)) != len(new_names):
        raise ValueError('Incomplete actor parameter-name mapping')
    new_opt['state'] = {}
    for name,new_id in new_ids.items():
        old_name = name.replace('mlp.fast_feedback.', 'mlp.feedback.', 1)
        if old_name not in old_ids:
            raise ValueError('New actor parameter has no authenticated parent: ' + name)
        old_id = old_ids[old_name]
        if old_id in old_opt['state']:
            new_opt['state'][new_id] = deepcopy(old_opt['state'][old_id])
    for old_id,new_id in zip(old_opt['param_groups'][1]['params'],new_opt['param_groups'][1]['params'],strict=True):
        if old_id in old_opt['state']:
            new_opt['state'][new_id] = deepcopy(old_opt['state'][old_id])
    for index in (0,1):
        ids = new_opt['param_groups'][index]['params']
        new_opt['param_groups'][index] = deepcopy(old_opt['param_groups'][index])
        new_opt['param_groups'][index]['params'] = ids
    adapted = deepcopy(parent)
    adapted['actor_state_dict'], adapted['optimizer_state_dict'] = new_actor, new_opt
    return adapted, dict(actor_function='exact_parent_mean_with_identical_feedback_clones',
        existing_weights_Adam_std_critic_RNG_counters='retained_without_refund',
        fast_feedback_weights_and_Adam='cloned_from_parent_shared_feedback_including_step',
        new_trainable_feedback_parameters=2252, new_feedback_tensors=4,
        inherited_shared_feedback_updates=parent['infos']['feedback_branch_actor_updates'],
        post_split_actor_updates=0, actor_clip_semantics='native_joint_actor_clip_unchanged')
