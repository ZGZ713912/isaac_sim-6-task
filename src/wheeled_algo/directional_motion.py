"""Small direction-specific corrections on the authenticated flat controller.

The four output banks separate forward, reverse and both yaw directions.
Only bounded, visible features reach them; a moving stop remains trainable.
Reference conditioning changes the actor input, never the external reward goal.
"""
from copy import deepcopy
import io
import json

import torch
from torch import nn

from .low_demand_feedback import LowDemandFeedbackActor
from .motion_state_residual import MotionStateResidualActor
from .speed_reference import condition_reference
from .static_grip import StaticGripActor, KIND as PARENT_KIND

KIND = 'original-static-grip-with-directional-motion-v12'


class DirectionalMotionActor(StaticGripActor):
    def __init__(self, parent, parent_sha256, *, transient_settings=None, leg_feedback_settings=None):
        if type(parent) is not StaticGripActor or len(parent_sha256) != 64:
            raise ValueError('Directional motion requires its exact v7 parent')
        nn.Module.__init__(self)
        for name, module in parent.named_children():
            setattr(self, name, deepcopy(module))
        for name, value in parent.named_buffers(recurse=False):
            self.register_buffer(name, value.detach().clone())
        self.tracking_gate = True
        self.recalled_sha256 = parent.recalled_sha256
        self._expected_static = deepcopy(parent._expected_static)
        self._expected_parent = {n: v.detach().cpu().clone() for n, v in parent.state_dict().items()}
        self.directional_parent_sha256 = parent_sha256
        self.motion_banks = nn.Linear(68, 24)
        nn.init.zeros_(self.motion_banks.weight)
        nn.init.zeros_(self.motion_banks.bias)
        self.register_buffer('motion_bound', torch.tensor([.15] * 4 + [.8] * 2))
        self.register_buffer('motion_feature_scale', torch.tensor(.1))
        self.transient_settings = None
        self.leg_feedback_settings = None
        if transient_settings is not None:
            from .transient_motion import validate_settings
            self.transient_settings = validate_settings(transient_settings)
            self.motion_bound.copy_(self.motion_bound.new_tensor(self.transient_settings['raw_action_bounds']))
        if leg_feedback_settings is not None:
            from .transient_leg_feedback import validate_settings
            if self.transient_settings is None:
                raise ValueError('Parent-leg feedback requires the declared transient actor')
            self.leg_feedback_settings = validate_settings(leg_feedback_settings)
        self.set_trainable(True)

    @staticmethod
    def bank_index(obs):
        rolling = (obs[:, 20:21] - obs[:, 21:22]) * .3
        gyro = obs[:, 6:7] * 2.
        stopping = (obs[:, :1].abs() < .2) & (obs[:, 2:3].abs() < .4)
        vx = torch.where(stopping, rolling, obs[:, :1])
        yaw = torch.where(stopping, gyro, obs[:, 2:3])
        translation = vx.abs() / .75 >= yaw.abs() / 1.5
        return torch.where(translation, (vx < 0.).long(), 2 + (yaw < 0.).long())

    def motion_gate(self, obs):
        inherited = MotionStateResidualActor.motion_gate(obs)
        if self.transient_settings is None:
            return inherited
        from .transient_motion import transient_gate
        return transient_gate(obs, inherited, self.transient_settings)

    def trust_region_groups(self, obs):
        index = self.bank_index(obs)
        return (index == torch.arange(4, device=obs.device)[None, :]) & (self.motion_gate(obs) > 0.)

    def conditioned_parent(self, obs):
        ordinary = super().forward(obs)
        # Keep pure turns and low commands on their existing controller. The
        # forward reference has no lean/encoder extrapolation. Reverse retains
        # the previously measured high-height reference, with no PPO credit.
        reverse = obs[:, :1] < -.75
        straight = obs[:, 2:3].abs() <= 1.5
        selected = (obs[:, :1].abs() > .75) & straight
        margin = torch.where(reverse, .3, .15)
        lean = torch.where(reverse, -.01, 0.)
        reference = condition_reference(obs, margin, lean)
        rolling = (obs[:, 20:21] - obs[:, 21:22]) * .3
        height = ((obs[:, 3:4] / 5. - .32) / .1).clamp(0., 1.)
        progress = (rolling * obs[:, :1].sign()).clamp(0., 1.)
        shift = (.5 * height * (obs[:, :1] - rolling) / .6).clamp(-2.5, 2.5) * progress
        shift = torch.where(reverse, shift, torch.zeros_like(shift))
        reference = torch.cat((reference[:, :20], reference[:, 20:22] + torch.cat((shift, -shift), -1),
                               reference[:, 22:]), -1)
        moving = (self.command(reference) + LowDemandFeedbackActor.feedback_actions(self, reference)
                  + self.authority_actions(reference))
        gate = self.motion_gate(obs)
        candidate = ordinary + gate * (moving - ordinary)
        return torch.where(selected, candidate, ordinary)

    def bounded_features(self, obs):
        hidden = obs
        for layer in list(self.command.base.children())[:-1]:
            hidden = layer(hidden)
        visible = torch.cat((obs[:, :1] / 3., obs[:, 2:3] / 4.,
            (obs[:, 20:21] - obs[:, 21:22]) * .1, obs[:, 6:7] / 2.), -1)
        return torch.cat((torch.tanh(hidden * self.motion_feature_scale), torch.tanh(visible)), -1)

    def forward(self, obs):
        parent = self.conditioned_parent(obs)
        if self.leg_feedback_settings is not None:
            from .transient_leg_feedback import parent_leg_actions
            parent = parent_leg_actions(parent, obs, self.leg_feedback_settings)
        features = self.bounded_features(obs)
        banks = self.motion_banks(features).reshape(-1, 4, 6)
        selector = (self.bank_index(obs) == torch.arange(4, device=obs.device)[None, :]).to(obs.dtype)
        raw = (banks * selector[:, :, None]).sum(1)
        delta = self.motion_bound * torch.tanh(raw / self.motion_bound)
        gate = self.motion_gate(obs)
        return torch.where(gate <= 0., parent, parent + gate * delta)

    def set_trainable(self, enabled, *, train_command=False):
        if train_command:
            raise ValueError('Directional motion freezes all existing controller branches')
        self.requires_grad_(False)
        if hasattr(self, 'motion_banks'):
            self.motion_banks.requires_grad_(enabled)
        for parameter in self.parameters():
            parameter.grad = None

    def verify_original(self, original):
        result = super().verify_original(original)
        current = self.state_dict()
        if any(not torch.equal(current[n].detach().cpu(), v) for n, v in self._expected_parent.items()):
            raise ValueError('Frozen v7 controller changed')
        bounds = ([.15] * 4 + [.8] * 2 if self.transient_settings is None
                  else self.transient_settings['raw_action_bounds'])
        if (not torch.equal(self.motion_bound.cpu(), torch.tensor(bounds))
                or self.motion_feature_scale.item() != torch.tensor(.1).item()):
            raise ValueError('Directional feature or action bounds changed')
        result['separated_feedback']['kind'] = KIND
        result['directional_motion'] = dict(kind=KIND, parent_sha256=self.directional_parent_sha256,
            trainable_parameters=1656, banks=['forward', 'reverse', 'yaw_positive', 'yaw_negative'],
            features='tanh(64 frozen features * .1), four tanh request/encoder/gyro features',
            gate='high command or moving stop; quiet and low moving commands retain the parent function',
            references='straight high translation only; gyro turns remain on the original controller',
            raw_action_bounds=self.motion_bound.detach().cpu().tolist(),
            preservation_scope='identical inputs outside active gate; native trajectory retention still required')
        if self.transient_settings is not None:
            result['directional_motion'].update(variant=deepcopy(self.transient_settings),
                gate='inherited motion OR visible velocity/yaw error OR pitch excursion',
                migration='same learned weights and Adam; declared gate and leg-bound function change')
        if self.leg_feedback_settings is not None:
            result['directional_motion'].update(leg_feedback=deepcopy(self.leg_feedback_settings),
                preservation_scope='same model tensors and direct wheel function; parent leg recurrence changes even outside the learned motion gate; native retention required',
                fast_leg_correction='bounded learned corrections added directly after parent-leg relaxation, with full gradient; no leg-motion penalty added')
        return result

    def export_graph(self, source):
        import onnx
        declaration = self.verify_original_from_source_full(source)
        buffer = io.BytesIO()
        torch.onnx.export(self.cpu().eval(), torch.zeros(1, 35), buffer, input_names=['obs'],
            output_names=['actions'], dynamic_axes={'obs': {0: 'batch'}, 'actions': {0: 'batch'}},
            opset_version=17, dynamo=False)
        graph = onnx.load_model_from_string(buffer.getvalue())
        onnx.helper.set_model_props(graph, dict(actor_adaptation=KIND,
            directional_motion=json.dumps(declaration['directional_motion'], separators=(',', ':'))))
        onnx.checker.check_model(graph, full_check=True)
        return graph


def configure_directional_training(actor, enabled):
    if type(actor.mlp) is not DirectionalMotionActor:
        raise ValueError('Missing declared directional controller')
    actor.requires_grad_(False)
    actor.mlp.set_trainable(enabled)
    for parameter in actor.parameters():
        parameter.grad = None
    return dict(enabled=enabled, trainable_names=[n for n, p in actor.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict() if not n.startswith('mlp.motion_banks.')])


def extend_directional_parent(parent, actor, critic, optimizer, old_names):
    if type(actor.mlp) is not DirectionalMotionActor or parent['infos']['actor_adaptation'] != PARENT_KIND:
        raise ValueError('Directional extension requires its authenticated v7 parent')
    old, new = parent['actor_state_dict'], deepcopy(actor.state_dict())
    added = {'mlp.motion_banks.weight', 'mlp.motion_banks.bias'}
    if set(new) - set(old) != added | {'mlp.motion_bound', 'mlp.motion_feature_scale'}:
        raise ValueError('Unexpected directional extension state')
    if any(p.detach().count_nonzero() for p in actor.mlp.motion_banks.parameters()):
        raise ValueError('New directional banks must start at zero')
    for name, value in old.items():
        if name not in new or new[name].shape != value.shape:
            raise ValueError('Existing actor ownership changed')
        new[name] = value.clone()
    names = list(dict(actor.named_parameters()))
    if old_names != [name for name in names if name not in added]:
        raise ValueError('Existing Adam owners changed')
    if set(critic.state_dict()) != set(parent['critic_state_dict']):
        raise ValueError('Critic architecture changed')
    old_opt, new_opt = parent['optimizer_state_dict'], deepcopy(optimizer.state_dict())
    if len(old_opt['param_groups']) != 2 or len(new_opt['param_groups']) != 2:
        raise ValueError('Separate actor and critic Adam groups required')
    old_ids = dict(zip(old_names, old_opt['param_groups'][0]['params'], strict=True))
    new_ids = dict(zip(names, new_opt['param_groups'][0]['params'], strict=True))
    new_opt['state'] = {new_ids[n]: deepcopy(old_opt['state'][i]) for n, i in old_ids.items() if i in old_opt['state']}
    for a, b in zip(old_opt['param_groups'][1]['params'], new_opt['param_groups'][1]['params'], strict=True):
        if a in old_opt['state']:
            new_opt['state'][b] = deepcopy(old_opt['state'][a])
    for index in (0, 1):
        ids = new_opt['param_groups'][index]['params']
        new_opt['param_groups'][index] = deepcopy(old_opt['param_groups'][index])
        new_opt['param_groups'][index]['params'] = ids
    state = deepcopy(parent)
    state.update(actor_state_dict=new, optimizer_state_dict=new_opt)
    return state, dict(added_parameters=1656, existing_actor_std_critic_Adam_RNG_counters='retained',
        new_Adam_slots='empty', initial_mean_is_identity=False,
        initial_reference_change='straight high forward/reverse only; native comparison required')
