"""Learn high-speed and moving-stop corrections on a frozen v10 controller."""
from copy import deepcopy
import json

import torch
from torch import nn

from .intent_speed_reference import IntentSpeedReferenceActor, KIND as PARENT_KIND

KIND='original-intent-speed-with-motion-state-residual-v11'


class MotionStateResidualActor(IntentSpeedReferenceActor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.motion_residual=nn.Linear(68,6)
        nn.init.zeros_(self.motion_residual.weight);nn.init.zeros_(self.motion_residual.bias)
        self.register_buffer('motion_residual_bound',torch.tensor([.15]*4+[.8]*2))
        self.set_trainable(True)

    @staticmethod
    def motion_gate(obs):
        rolling=(obs[:,20:21]-obs[:,21:22])*.3
        gyro=obs[:,6:7]*2.
        high=(torch.maximum(obs[:,:1].abs()/.75,obs[:,2:3].abs()/1.5)-1.).clamp(0.,1.)
        state=(torch.maximum(rolling.abs()/.5,gyro.abs()/1.5)-.3).clamp(0.,.7)/.7
        stop=(1.-obs[:,:1].abs()/.2).clamp(0.,1.)*(1.-obs[:,2:3].abs()/.4).clamp(0.,1.)
        return torch.maximum(high,state*stop)

    def forward(self,obs):
        parent=super().forward(obs)
        features=obs
        for layer in list(self.command.base.children())[:-1]:features=layer(features)
        visible=torch.cat((obs[:,:1]/3.,obs[:,2:3]/4.,(obs[:,20:21]-obs[:,21:22])*.1,obs[:,6:7]/2.),-1)
        raw=self.motion_residual(torch.cat((features,visible),-1))
        bounded=self.motion_residual_bound*torch.tanh(raw/self.motion_residual_bound)
        gate=self.motion_gate(obs)
        return torch.where(gate<=0.,parent,parent+gate*bounded)

    def set_trainable(self,enabled,*,train_command=False):
        super().set_trainable(False,train_command=False)
        if hasattr(self,'motion_residual'):self.motion_residual.requires_grad_(enabled)

    def project_parameters_(self):
        # All v10 reference parameters are frozen; the new head is bounded in forward.
        pass

    def static_feedback_contract(self):
        result=super().static_feedback_contract()
        result['stationary_action_function']='exact v10 when the motion-state gate is zero; moving-stop states are trainable'
        return result

    def verify_original(self,original):
        result=super().verify_original(original)
        result['separated_feedback']['kind']=KIND
        result['speed_reference']['kind']=KIND
        result['speed_reference']['trainable_parameters']=0
        result['speed_reference']['base_low_command_exact_limits']=result['speed_reference'].pop('low_command_exact_limits')
        if not torch.equal(self.motion_residual_bound.detach().cpu(),torch.tensor([.15]*4+[.8]*2)):
            raise ValueError('Motion-state action bounds changed')
        result['motion_state_residual']=dict(kind=KIND,trainable_parameters=414,
            initial_mean='exact v10 identity extension',parent='all v10 weights and reference parameters frozen',
            gate='high request OR near-zero request while encoder rolling/IMU yaw still high',
            quiet_state='zero branch below .15m/s rolling and .45rad/s yaw',
            low_movement='zero branch for .5m/s translation and 1rad/s yaw',
            raw_action_bounds=[.15]*4+[.8]*2)
        return result

    def export_graph(self,source):
        import onnx
        graph=super().export_graph(source)
        metadata={p.key:p.value for p in graph.metadata_props}
        metadata.update(actor_adaptation=KIND,motion_state_residual=json.dumps(
            self.verify_original_from_source_full(source)['motion_state_residual'],separators=(',',':')))
        onnx.helper.set_model_props(graph,metadata)
        return graph


def configure_motion_state_training(actor,enabled):
    if type(actor.mlp)is not MotionStateResidualActor:raise ValueError('Missing declared v11 actor')
    actor.requires_grad_(False);actor.mlp.set_trainable(enabled)
    for p in actor.parameters():p.grad=None
    return dict(enabled=enabled,trainable_names=[n for n,p in actor.named_parameters()if p.requires_grad],
        trainable_parameters=sum(p.numel()for p in actor.parameters()if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict()if not n.startswith('mlp.motion_residual.')])


def extend_motion_state_parent(parent,actor,critic,optimizer,old_names):
    if type(actor.mlp)is not MotionStateResidualActor or parent['infos']['actor_adaptation']!=PARENT_KIND:
        raise ValueError('v11 requires the exact authenticated v10 parent')
    old,new=parent['actor_state_dict'],deepcopy(actor.state_dict())
    added={'mlp.motion_residual.weight','mlp.motion_residual.bias'}
    if set(new)-set(old)!=added|{'mlp.motion_residual_bound'}or not set(old)<=set(new):
        raise ValueError('Unexpected motion-state extension')
    if any(p.detach().count_nonzero()for p in actor.mlp.motion_residual.parameters()):
        raise ValueError('Motion-state head must start at zero')
    for n,v in old.items():
        if new[n].shape!=v.shape:raise ValueError('Existing actor shape changed')
        new[n]=v.clone()
    names=list(dict(actor.named_parameters()))
    if old_names!=[n for n in names if n not in added]:raise ValueError('Existing Adam owners changed')
    old_opt,new_opt=parent['optimizer_state_dict'],deepcopy(optimizer.state_dict())
    if len(old_opt['param_groups'])!=2 or len(new_opt['param_groups'])!=2:raise ValueError('Separate Adam groups required')
    old_ids=dict(zip(old_names,old_opt['param_groups'][0]['params'],strict=True))
    new_ids=dict(zip(names,new_opt['param_groups'][0]['params'],strict=True))
    new_opt['state']={new_ids[n]:deepcopy(old_opt['state'][i])for n,i in old_ids.items()if i in old_opt['state']}
    if set(critic.state_dict())!=set(parent['critic_state_dict']):raise ValueError('Critic architecture changed')
    for a,b in zip(old_opt['param_groups'][1]['params'],new_opt['param_groups'][1]['params'],strict=True):
        if a in old_opt['state']:new_opt['state'][b]=deepcopy(old_opt['state'][a])
    for index in (0,1):
        ids=new_opt['param_groups'][index]['params'];new_opt['param_groups'][index]=deepcopy(old_opt['param_groups'][index])
        new_opt['param_groups'][index]['params']=ids
    state=deepcopy(parent);state['actor_state_dict']=new;state['optimizer_state_dict']=new_opt
    return state,dict(initial_mean_is_identity=True,added_parameters=414,existing_actor_std_critic_Adam_RNG_counters='retained',new_Adam_slots='empty')
