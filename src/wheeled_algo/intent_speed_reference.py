"""Keep motion intent separate from the frozen policy's reachable reference."""
from copy import deepcopy
import json

import torch
from torch import nn

from .low_demand_feedback import LowDemandFeedbackActor
from .speed_reference import SpeedReferenceActor, condition_reference

KIND='original-static-grip-with-intent-speed-reference-v10'


class IntentSpeedReferenceActor(SpeedReferenceActor):
    def __init__(self,parent,parent_sha256,margins,lean_gains,encoder_gain=.5):
        values=torch.as_tensor(margins,dtype=torch.float32)
        lower=torch.tensor([.05,.05,.005,.005])
        upper=torch.tensor([1.2,1.2,.5,.5])
        if values.shape!=(4,) or not torch.isfinite(values).all() or (values<lower).any() or (values>upper).any():
            raise ValueError('Invalid translation/yaw reference margins')
        if not 0. <= encoder_gain <= .6:
            raise ValueError('Encoder reference gain must be within [0,.6]')
        super().__init__(parent,parent_sha256,values.clamp_min(.05).tolist(),lean_gains)
        self.reference_margin=nn.Parameter(values.clone())
        self.reference_encoder_gain=nn.Parameter(torch.tensor(float(encoder_gain)))
        self.speed_residual=nn.Linear(68,6)
        nn.init.zeros_(self.speed_residual.weight);nn.init.zeros_(self.speed_residual.bias)
        self.set_trainable(True)

    def forward(self,obs):
        gate=(torch.maximum(obs[:,:1].abs()/.75,obs[:,2:3].abs()/1.5)-1.).clamp(0.,1.)
        translation=obs[:,:1].abs()/.75>=obs[:,2:3].abs()/1.5
        index=torch.where(translation,(obs[:,:1]<0.).long(),2+(obs[:,2:3]<0.).long())
        lower=self.reference_margin.new_tensor([.05,.05,.005,.005])
        upper=self.reference_margin.new_tensor([1.2,1.2,.5,.5])
        margin=torch.maximum(torch.minimum(self.reference_margin,upper),lower)[index]
        lean=self.reference_lean_gain.clamp(-.04,.04)[(obs[:,:1]<0.).long()]
        reference=condition_reference(obs,margin,lean)
        rolling=(obs[:,20:21]-obs[:,21:22])*.3
        height=((obs[:,3:4]/5.-.32)/.1).clamp(0.,1.)
        progress=(rolling*obs[:,:1].sign()).clamp(0.,1.)
        shift=(self.reference_encoder_gain.clamp(0.,.6)*height*(obs[:,:1]-rolling)/.6).clamp(-2.5,2.5)*progress*gate
        shift=torch.where(obs[:,:1]<-.75,shift,torch.zeros_like(shift))
        reference=torch.cat((reference[:,:20],reference[:,20:22]+torch.cat((shift,-shift),-1),reference[:,22:]),-1)
        # Internal references can be small while the actual request is fast.
        # Such requests must not activate the stationary-feedback correction.
        original_feedback=self.feedback_actions(reference)
        moving=LowDemandFeedbackActor.feedback_actions(self,reference)
        feedback=torch.where(gate<=0.,original_feedback,original_feedback+gate*(moving-original_feedback))
        parent=self.command(reference)+feedback+self.authority_actions(reference)
        features=reference
        for layer in list(self.command.base.children())[:-1]:
            features=layer(features)
        # Retain the external request after reference conditioning. Without
        # these features distinct high targets can become the same input.
        visible=torch.cat((obs[:,:1]/3.,obs[:,2:3]/4.,rolling/3.,obs[:,6:7]/2.),-1)
        raw=self.speed_residual(torch.cat((features,visible),-1))
        bounded=self.speed_residual_bound*torch.tanh(raw/self.speed_residual_bound)
        return torch.where(gate<=0.,parent,parent+gate*bounded)

    def set_trainable(self,enabled,*,train_command=False):
        super().set_trainable(enabled,train_command=train_command)
        if hasattr(self,'reference_encoder_gain'):
            self.reference_encoder_gain.requires_grad_(enabled)

    @torch.no_grad()
    def project_parameters_(self):
        lower=self.reference_margin.new_tensor([.05,.05,.005,.005])
        upper=self.reference_margin.new_tensor([1.2,1.2,.5,.5])
        self.reference_margin.copy_(torch.maximum(torch.minimum(self.reference_margin,upper),lower))
        self.reference_lean_gain.clamp_(-.04,.04)
        self.reference_encoder_gain.clamp_(0.,.6)

    def verify_original(self,original):
        result=super().verify_original(original)
        if not torch.isfinite(self.reference_encoder_gain):
            raise ValueError('Nonfinite encoder reference')
        declaration=result['speed_reference']
        declaration.update(kind=KIND,trainable_parameters=421,
            margin_m_s=self.reference_margin.detach().cpu().tolist(),
            margin_lower_bounds=[.05,.05,.005,.005],margin_upper_bounds=[1.2,1.2,.5,.5],
            encoder_gain=float(self.reference_encoder_gain.clamp(0.,.6).detach()),
            encoder_reference='reverse only; height .32 to .42m; progressive rolling; external high gate',
            motion_intent='external high-command gate selects moving feedback independently of the internal reference',
            optimizer_projection='bounded reference parameters after each enabled actor Adam step',
            feature_residual=dict(feature_source='frozen 64D features plus four unconditioned request/sensor features',
                trainable_parameters=414,initial_output='zero',raw_action_bounds=[.3,.3,.3,.3,3.,3.]))
        result['separated_feedback']['kind']=KIND
        return result

    def export_graph(self,source):
        import onnx
        graph=super().export_graph(source)
        metadata={p.key:p.value for p in graph.metadata_props}
        metadata.update(actor_adaptation=KIND,speed_reference=json.dumps(
            self.verify_original_from_source_full(source)['speed_reference'],separators=(',',':')))
        onnx.helper.set_model_props(graph,metadata)
        return graph


def configure_intent_speed_training(actor,enabled):
    if type(actor.mlp) is not IntentSpeedReferenceActor:
        raise ValueError('Missing declared v10 actor')
    actor.requires_grad_(False);actor.mlp.set_trainable(enabled)
    for p in actor.parameters():
        p.grad=None
    return dict(enabled=enabled,trainable_names=[n for n,p in actor.named_parameters()if p.requires_grad],
        trainable_parameters=sum(p.numel()for p in actor.parameters()if p.requires_grad),
        frozen_state_names=[n for n in actor.state_dict()if n not in
            ('mlp.reference_margin','mlp.reference_lean_gain','mlp.reference_encoder_gain')
            and not n.startswith('mlp.speed_residual.')])


def extend_intent_speed_parent(parent,actor,critic,optimizer,old_names):
    from .static_grip import KIND as PARENT_KIND
    if type(actor.mlp) is not IntentSpeedReferenceActor or parent['infos']['actor_adaptation']!=PARENT_KIND:
        raise ValueError('v10 requires its explicit authenticated v7 parent')
    old,new=parent['actor_state_dict'],deepcopy(actor.state_dict())
    added={'mlp.reference_margin','mlp.reference_lean_gain','mlp.reference_encoder_gain',
           'mlp.speed_residual.weight','mlp.speed_residual.bias'}
    if set(new)-set(old)!=added|{'mlp.speed_residual_bound'} or not set(old)<=set(new):
        raise ValueError('Unexpected v10 state extension')
    if any(p.detach().count_nonzero()for p in actor.mlp.speed_residual.parameters()):
        raise ValueError('New feature output must start at zero')
    for n,v in old.items():
        if v.shape!=new[n].shape:
            raise ValueError('Original actor shape changed')
        new[n]=v.clone()
    names=list(dict(actor.named_parameters()))
    if old_names!=[n for n in names if n not in added]:
        raise ValueError('Actor parameter ownership changed')
    old_opt,new_opt=parent['optimizer_state_dict'],deepcopy(optimizer.state_dict())
    if len(old_opt['param_groups'])!=2 or len(new_opt['param_groups'])!=2:
        raise ValueError('Separate actor/critic optimizer groups required')
    old_ids=dict(zip(old_names,old_opt['param_groups'][0]['params'],strict=True))
    new_ids=dict(zip(names,new_opt['param_groups'][0]['params'],strict=True))
    new_opt['state']={new_ids[n]:deepcopy(old_opt['state'][i])for n,i in old_ids.items()if i in old_opt['state']}
    if any(n not in parent['critic_state_dict']or v.shape!=parent['critic_state_dict'][n].shape
           for n,v in critic.state_dict().items()):
        raise ValueError('Critic architecture changed')
    for a,b in zip(old_opt['param_groups'][1]['params'],new_opt['param_groups'][1]['params'],strict=True):
        if a in old_opt['state']:
            new_opt['state'][b]=deepcopy(old_opt['state'][a])
    for index in (0,1):
        ids=new_opt['param_groups'][index]['params']
        new_opt['param_groups'][index]=deepcopy(old_opt['param_groups'][index])
        new_opt['param_groups'][index]['params']=ids
    state=deepcopy(parent);state['actor_state_dict']=new;state['optimizer_state_dict']=new_opt
    return state,dict(initial_mean_is_identity=False,added_parameters=421,
        existing_actor_std_critic_Adam_RNG_counters='retained',new_Adam_slots='empty',
        low_command_function='exact for identical input; native retention still required')
