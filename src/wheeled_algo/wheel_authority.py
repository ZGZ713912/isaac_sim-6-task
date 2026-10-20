"""Unadopted zero-extension for high demand and moving zero-command braking."""
from copy import deepcopy
import json
import torch
from torch import nn
from wheeled_algo.static_feedback import StaticFeedbackActor

KIND = 'original-static-feedback-and-wheel-authority-residual-v6'


class WheelAuthorityActor(StaticFeedbackActor):
    """Append two bounded wheel outputs; preserve the v5 function at entry."""
    def __init__(self, parent):
        if type(parent) is not StaticFeedbackActor:
            raise ValueError('Wheel authority requires the exact v5 parent')
        nn.Module.__init__(self)
        for name in ('command', 'feedback', 'fast_feedback', 'static_feedback'):
            setattr(self, name, deepcopy(getattr(parent, name)))
        self.tracking_gate = True
        self.recalled_sha256 = parent.recalled_sha256
        self._expected_static = deepcopy(parent._expected_static)
        for name in ('input_center', 'input_scale', 'output_scale'):
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.register_buffer('authority_bound', torch.tensor([6., 6.]))
        self.wheel_authority = nn.Sequential(nn.Linear(28, 32), nn.ELU(), nn.Linear(32, 2))
        nn.init.zeros_(self.wheel_authority[2].weight)
        nn.init.zeros_(self.wheel_authority[2].bias)
        self.set_trainable(True)

    @staticmethod
    def authority_gate(obs):
        demand = torch.maximum(obs[:, :1].abs()/.75, obs[:, 2:3].abs()/1.5)
        high = (demand - 1.).clamp(0., 1.)
        # Encoder speed is visible to the policy; no body velocity is used.
        rim = obs[:, 20:22].abs().amax(-1, keepdim=True)*.6
        braking = ((rim - .4)/.4).clamp(0., 1.) * (1. - StaticFeedbackActor.moving_mix(obs))
        return torch.maximum(high, braking)

    def authority_actions(self, obs):
        features = (obs[:, :28] - self.input_center)*self.input_scale
        raw = self.wheel_authority(features)
        # Unit initial derivative avoids multiplying first-step sensitivity by6.
        wheel = self.authority_bound*torch.tanh(raw/self.authority_bound)*self.authority_gate(obs)
        return torch.cat((torch.zeros_like(obs[:, :4]), wheel), -1)

    def forward(self, obs):
        return super().forward(obs) + self.authority_actions(obs)

    def set_trainable(self, enabled, *, train_command=False):
        # Preserve all existing learned feedback/Adam. This candidate initially
        # trains only the newly appended high-demand/braking wheel head.
        super().set_trainable(False, train_command=False)
        self.wheel_authority.requires_grad_(enabled)
        for parameter in self.parameters():
            if not parameter.requires_grad:
                parameter.grad = None

    def verify_original(self, original):
        result = super().verify_original(original)
        if not torch.equal(self.authority_bound.detach().cpu(), torch.tensor([6., 6.])):
            raise ValueError('Wheel authority bound changed')
        result['wheel_authority'] = dict(kind=KIND, bounded_raw_wheel_offsets=[6., 6.],
            wheel_rad_s_bound=60., feature_indices=list(range(28)), head_parameters=994,
            initial_extension='exact_parent_function; zero final layer',
            low_command_invariance='abs(vx)<=.75 and abs(yaw)<=1.5 with full moving mix, or encoder rim<=.4 at zero command',
            high_gate='clip(max(abs(vx)/.75,abs(yaw)/1.5)-1,0,1)',
            braking_gate='clip((max(abs(obs20),abs(obs21))*.6-.4)/.4,0,1)*(1-moving_mix)',
            controller_wheel_clip='unchanged; final total actions remain clipped by the existing actuator contract')
        return result

    def export_graph(self, source):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        model = super().export_graph(source)
        output = model.graph.output[0].name
        producers = [n for n in model.graph.node if output in n.output]
        if len(producers) != 1:
            raise ValueError('Expected one final parent output')
        prefix = 'wheel_authority.'
        n = lambda name: prefix + name
        index = list(producers[0].output).index(output)
        producers[0].output[index] = n('parent')
        values = {'bound': self.authority_bound.detach().cpu().numpy().copy(),
            'rim_scale': np.array(.6, dtype=np.float32),
            'rim_floor': np.array(.4, dtype=np.float32), 'rim_gain': np.array(2.5, dtype=np.float32),
            'starts': np.array([20], dtype=np.int64), 'ends': np.array([22], dtype=np.int64),
            'axes': np.array([1], dtype=np.int64), 'steps': np.array([1], dtype=np.int64),
            'zero_legs': np.zeros(4, dtype=np.float32)}
        values.update({'head.'+key:value.detach().cpu().numpy().copy()
                       for key, value in self.wheel_authority.state_dict().items()})
        model.graph.initializer.extend(numpy_helper.from_array(value, n(key)) for key, value in values.items())
        model.graph.node.extend([
            helper.make_node('Gemm', ['gated_feedback.normalized',n('head.0.weight'),n('head.0.bias')], [n('hidden0')], transB=1),
            helper.make_node('Elu', [n('hidden0')], [n('hidden')], alpha=1.),
            helper.make_node('Gemm', [n('hidden'),n('head.2.weight'),n('head.2.bias')], [n('raw')], transB=1),
            helper.make_node('Div', [n('raw'),n('bound')], [n('normalized_output')]),
            helper.make_node('Tanh', [n('normalized_output')], [n('bounded_unit')]),
            helper.make_node('Mul', [n('bounded_unit'),n('bound')], [n('bounded_wheel')]),
            helper.make_node('Sub', ['gated_feedback.demand','gated_feedback.one'], [n('high_raw')]),
            helper.make_node('Clip', [n('high_raw'),'gated_feedback.zero','gated_feedback.one'], [n('high')]),
            helper.make_node('Slice', [model.graph.input[0].name,n('starts'),n('ends'),n('axes'),n('steps')], [n('encoder')]),
            helper.make_node('Abs', [n('encoder')], [n('encoder_abs')]),
            helper.make_node('ReduceMax', [n('encoder_abs')], [n('encoder_peak')], axes=[1], keepdims=1),
            helper.make_node('Mul', [n('encoder_peak'),n('rim_scale')], [n('rim')]),
            helper.make_node('Sub', [n('rim'),n('rim_floor')], [n('excess_rim')]),
            helper.make_node('Mul', [n('excess_rim'),n('rim_gain')], [n('brake_raw')]),
            helper.make_node('Clip', [n('brake_raw'),'gated_feedback.zero','gated_feedback.one'], [n('brake_clip')]),
            helper.make_node('Sub', ['gated_feedback.one','static_feedback.mix'], [n('static_mix')]),
            helper.make_node('Mul', [n('brake_clip'),n('static_mix')], [n('braking')]),
            helper.make_node('Max', [n('high'),n('braking')], [n('gate')]),
            helper.make_node('Mul', [n('bounded_wheel'),n('gate')], [n('wheel')]),
            helper.make_node('Mul', ['command_residual.command',n('zero_legs')], [n('legs')]),
            helper.make_node('Concat', [n('legs'),n('wheel')], [n('correction')], axis=1),
            helper.make_node('Add', [n('parent'),n('correction')], [output]),
        ])
        metadata = {p.key:p.value for p in model.metadata_props}
        metadata['actor_adaptation'] = KIND
        metadata['wheel_authority'] = json.dumps(self.verify_original_from_source_full(source)['wheel_authority'], separators=(',',':'))
        helper.set_model_props(model, metadata)
        onnx.checker.check_model(model, full_check=True)
        return model


def extend_wheel_authority_parent(parent, actor, critic, optimizer, old_actor_names):
    """Append only the zero head and remap old actor/critic Adam by ownership."""
    from wheeled_algo.static_feedback import KIND as PARENT_KIND
    if type(actor.mlp) is not WheelAuthorityActor or parent['infos']['actor_adaptation'] != PARENT_KIND:
        raise ValueError('Wheel authority requires the explicit v5 identity extension')
    if any(torch.count_nonzero(p.detach()).item() for p in actor.mlp.wheel_authority[2].parameters()):
        raise ValueError('Identity extension requires a zero-output final layer')
    if not torch.equal(actor.mlp.authority_bound.detach().cpu(), torch.tensor([6., 6.])):
        raise ValueError('Identity extension requires the fixed wheel bound')
    transformed = deepcopy(parent)
    old, new = parent['actor_state_dict'], deepcopy(actor.state_dict())
    extras = set(new)-set(old)
    expected = {'mlp.wheel_authority.'+n for n in actor.mlp.wheel_authority.state_dict()}
    if extras != expected | {'mlp.authority_bound'} or len(expected) != 4:
        raise ValueError('Wheel authority must append exactly its four head tensors and fixed bound')
    for name, value in old.items():
        if name not in new or new[name].shape != value.shape:
            raise ValueError('Existing actor state changed architecture')
        new[name] = value.clone()
    names = list(dict(actor.named_parameters()))
    if old_actor_names != [n for n in names if n not in expected]:
        raise ValueError('Existing actor parameter order changed')
    old_opt, new_opt = parent['optimizer_state_dict'], deepcopy(optimizer.state_dict())
    if len(old_opt['param_groups']) != 2 or len(new_opt['param_groups']) != 2:
        raise ValueError('Separate actor/critic Adam groups are required')
    if set(critic.state_dict()) != set(parent['critic_state_dict']) or any(
            v.shape != parent['critic_state_dict'][n].shape for n,v in critic.state_dict().items()):
        raise ValueError('Critic architecture changed')
    old_ids = dict(zip(old_actor_names,old_opt['param_groups'][0]['params'],strict=True))
    new_ids = dict(zip(names,new_opt['param_groups'][0]['params'],strict=True))
    new_opt['state'] = {new_ids[name]:deepcopy(old_opt['state'][old_id])
                        for name,old_id in old_ids.items() if old_id in old_opt['state']}
    for a,b in zip(old_opt['param_groups'][1]['params'],new_opt['param_groups'][1]['params'],strict=True):
        if a in old_opt['state']:
            new_opt['state'][b] = deepcopy(old_opt['state'][a])
    for index in (0,1):
        ids = new_opt['param_groups'][index]['params']
        new_opt['param_groups'][index] = deepcopy(old_opt['param_groups'][index])
        new_opt['param_groups'][index]['params'] = ids
    transformed['actor_state_dict'], transformed['optimizer_state_dict'] = new,new_opt
    return transformed, dict(initial_mean_is_identity=True, existing_weights_Adam_std_critic_RNG_counters='retained',
        new_head='zero_output; no Adam state;994 trainable parameters', action_bound='extra wheels +/-6 raw; final existing controller clips unchanged')


def configure_wheel_authority_training(actor, enabled):
    if type(actor.mlp) is not WheelAuthorityActor:
        raise ValueError('Actor does not contain the declared wheel-authority extension')
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    actor.mlp.set_trainable(enabled)
    return dict(enabled=enabled,
        trainable_names=[n for n,p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict() if not n.startswith('mlp.wheel_authority.')])
