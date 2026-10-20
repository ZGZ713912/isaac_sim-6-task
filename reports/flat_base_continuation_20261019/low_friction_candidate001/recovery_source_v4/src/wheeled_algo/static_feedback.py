"""Unadopted trainable extension with an explicitly frozen static snapshot."""
from copy import deepcopy
import json
import torch
from torch import nn
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor
from wheeled_algo.separated_feedback import SeparatedFeedbackActor

KIND = 'original-command-and-static-recall-feedback-residual-v5'


class StaticFeedbackActor(LowDemandFeedbackActor):
    """Preserve all v4 parameter names and append only the frozen static head."""
    def __init__(self, parent, recalled, recalled_sha256):
        if type(parent) is not LowDemandFeedbackActor or type(recalled) is not SeparatedFeedbackActor:
            raise ValueError('Static feedback requires exact v4 and v3 means')
        if len(recalled_sha256) != 64 or any(c not in '0123456789abcdef' for c in recalled_sha256):
            raise ValueError('Static feedback requires its exact checkpoint SHA256')
        current, old = parent.command.state_dict(), recalled.command.state_dict()
        if current.keys() != old.keys() or any(not torch.equal(v, old[n]) for n,v in current.items()):
            raise ValueError('Static and moving means must share the unchanged command branch')
        nn.Module.__init__(self)
        self.command = deepcopy(parent.command)
        self.tracking_gate = True
        for name in ('input_center', 'input_scale', 'output_scale'):
            if not torch.equal(getattr(parent, name), getattr(recalled, name)):
                raise ValueError('Static recall changes feedback scaling or bounds')
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.feedback = deepcopy(parent.feedback)
        self.fast_feedback = deepcopy(parent.fast_feedback)
        self.static_feedback = deepcopy(recalled.feedback)
        self.recalled_sha256 = recalled_sha256
        self._expected_static = {n:v.detach().cpu().clone() for n,v in recalled.feedback.state_dict().items()}
        self.set_trainable(True)

    @staticmethod
    def moving_mix(obs):
        return torch.maximum(obs[:, :1].abs()/.1, obs[:, 2:3].abs()/.2).clamp(0., 1.)

    def feedback_actions(self, obs):
        features = (obs[:, :28]-self.input_center)*self.input_scale
        static = self.output_scale*torch.tanh(self.static_feedback(features))
        moving = super().feedback_actions(obs)
        mix = self.moving_mix(obs)
        blended = static+mix*(moving-static)
        return torch.where(mix <= 0., static, torch.where(mix >= 1., moving, blended))

    def set_trainable(self, enabled, *, train_command=False):
        super().set_trainable(enabled, train_command=train_command)
        self.static_feedback.requires_grad_(False)
        for parameter in self.static_feedback.parameters():
            parameter.grad = None

    def verify_original(self, original):
        result = super().verify_original(original)
        if any(not torch.equal(v.detach().cpu(), self._expected_static[n])
               for n,v in self.static_feedback.state_dict().items()):
            raise ValueError('Frozen recalled static feedback changed')
        result['separated_feedback']['kind'] = KIND
        result['static_feedback_recall'] = dict(checkpoint_sha256=self.recalled_sha256,
            static_parameters=sum(p.numel() for p in self.static_feedback.parameters()),
            static_trainable=False, moving_full_vx_m_s=.1, moving_full_yaw_rad_s=.2,
            stationary_action_function='exact_recalled_feedback_and_same_command_branch',
            moving_action_function='exact_v4_at_full_moving_endpoint',
            stopping='zero-command braking also uses recalled feedback; native replay required')
        return result

    def export_graph(self, source):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        model = super().export_graph(source)
        prefix = 'static_feedback.'
        obs = model.graph.input[0].name
        corrections = [i for i,n in enumerate(model.graph.node) if list(n.output) == ['gated_feedback.correction']]
        if len(corrections) != 1:
            raise ValueError('v4 graph changed its feedback correction node')
        index = corrections[0]
        nodes = [deepcopy(n) for n in model.graph.node]
        nodes[index].output[0] = prefix+'moving'
        constants = dict(vx_inverse=np.array(10.,dtype=np.float32),
                         yaw_inverse=np.array(5.,dtype=np.float32))
        constants.update({'head.'+n:v.detach().cpu().numpy().copy() for n,v in self.static_feedback.state_dict().items()})
        model.graph.initializer.extend(numpy_helper.from_array(v,prefix+n) for n,v in constants.items())
        n = lambda name: prefix+name
        added = [
            helper.make_node('Gemm',['gated_feedback.normalized',n('head.0.weight'),n('head.0.bias')],[n('hidden0')],transB=1),
            helper.make_node('Elu',[n('hidden0')],[n('hidden')],alpha=1.),
            helper.make_node('Gemm',[n('hidden'),n('head.2.weight'),n('head.2.bias')],[n('raw')],transB=1),
            helper.make_node('Tanh',[n('raw')],[n('bounded')]),
            helper.make_node('Mul',[n('bounded'),'gated_feedback.output_scale'],[n('static')]),
            helper.make_node('Mul',['gated_feedback.abs_vx',n('vx_inverse')],[n('vx_demand')]),
            helper.make_node('Mul',['gated_feedback.abs_yaw',n('yaw_inverse')],[n('yaw_demand')]),
            helper.make_node('Max',[n('vx_demand'),n('yaw_demand')],[n('demand')]),
            helper.make_node('Clip',[n('demand'),'gated_feedback.zero','gated_feedback.one'],[n('mix')]),
            helper.make_node('Sub',[n('moving'),n('static')],[n('difference')]),
            helper.make_node('Mul',[n('mix'),n('difference')],[n('weighted')]),
            helper.make_node('Add',[n('static'),n('weighted')],[n('blend')]),
            helper.make_node('Equal',[n('mix'),'gated_feedback.zero'],[n('static_endpoint')]),
            helper.make_node('Equal',[n('mix'),'gated_feedback.one'],[n('moving_endpoint')]),
            helper.make_node('Where',[n('moving_endpoint'),n('moving'),n('blend')],[n('moving_or_blend')]),
            helper.make_node('Where',[n('static_endpoint'),n('static'),n('moving_or_blend')],['gated_feedback.correction']),
        ]
        del model.graph.node[:]
        model.graph.node.extend(nodes[:index+1]+added+nodes[index+1:])
        from wheeled_algo.onnx_actor_import import recover_actor
        original,_ = recover_actor(source)
        metadata = {p.key:p.value for p in model.metadata_props}
        metadata['actor_adaptation'] = KIND
        metadata['static_feedback_recall'] = json.dumps(self.verify_original(original)['static_feedback_recall'],separators=(',',':'))
        helper.set_model_props(model,metadata)
        onnx.checker.check_model(model,full_check=True)
        return model


def extend_static_feedback_parent(parent, actor, critic, optimizer, old_actor_names):
    """Append four frozen static tensors and preserve every existing Adam slot."""
    from wheeled_algo.low_demand_feedback import KIND as PARENT_KIND
    if type(actor.mlp) is not StaticFeedbackActor or parent['infos']['actor_adaptation'] != PARENT_KIND:
        raise ValueError('Require the explicit v4 to static-recall composition')
    old = parent['actor_state_dict']
    new = deepcopy(actor.state_dict())
    extras = set(new)-set(old)
    if extras != {'mlp.static_feedback.'+n for n in actor.mlp.static_feedback.state_dict()} or len(extras) != 4:
        raise ValueError('Static extension must append exactly four frozen head tensors')
    for name,value in old.items():
        if name not in new or new[name].shape != value.shape:
            raise ValueError('An existing actor tensor changed ownership or shape')
        new[name] = value.clone()
    old_opt, new_opt = parent['optimizer_state_dict'],deepcopy(optimizer.state_dict())
    names = list(dict(actor.named_parameters()))
    if old_actor_names != [name for name in names if name not in extras]:
        raise ValueError('Existing actor parameter order changed')
    if (len(old_opt['param_groups']) != 2 or len(new_opt['param_groups']) != 2
            or len(old_opt['param_groups'][0]['params']) != len(old_actor_names)
            or len(new_opt['param_groups'][0]['params']) != len(names)
            or set(critic.state_dict()) != set(parent['critic_state_dict'])
            or any(v.shape != parent['critic_state_dict'][n].shape for n,v in critic.state_dict().items())):
        raise ValueError('Static extension changed Adam groups or critic architecture')
    old_ids = dict(zip(old_actor_names,old_opt['param_groups'][0]['params'],strict=True))
    new_ids = dict(zip(names,new_opt['param_groups'][0]['params'],strict=True))
    new_opt['state'] = {}
    for name,old_id in old_ids.items():
        if old_id in old_opt['state']:
            new_opt['state'][new_ids[name]] = deepcopy(old_opt['state'][old_id])
    for old_id,new_id in zip(old_opt['param_groups'][1]['params'],new_opt['param_groups'][1]['params'],strict=True):
        if old_id in old_opt['state']:
            new_opt['state'][new_id] = deepcopy(old_opt['state'][old_id])
    for index in (0,1):
        ids = new_opt['param_groups'][index]['params']
        new_opt['param_groups'][index] = deepcopy(old_opt['param_groups'][index])
        new_opt['param_groups'][index]['params'] = ids
    transformed = deepcopy(parent)
    transformed['actor_state_dict'],transformed['optimizer_state_dict'] = new,new_opt
    return transformed,dict(existing_weights_Adam_std_critic_RNG_and_consumption='retained',
        static_feedback='four_frozen_tensors_from_declared_recalled_checkpoint; no_Adam_state',
        recalled_checkpoint_sha256=actor.mlp.recalled_sha256,initial_mean_is_identity=False)


def configure_static_feedback_training(actor, enabled):
    if type(actor.mlp) is not StaticFeedbackActor:
        raise ValueError('Actor does not contain the declared static feedback recall')
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    prefixes = ('mlp.feedback.', 'mlp.fast_feedback.')
    return dict(enabled=enabled,
        trainable_names=[n for n,p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict() if not n.startswith(prefixes)])
