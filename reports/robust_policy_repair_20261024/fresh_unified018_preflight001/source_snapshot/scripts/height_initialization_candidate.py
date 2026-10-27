"""Unadopted trainable height features for fresh three-height policy learning.

Coefficients initialize from declared static mechanics, never policy weights.
They remain ordinary actor parameters. Commands and all feedback still enter
the six-action neural model; no pose table, action hold or recovery controller
is evaluated at runtime. This candidate is not in the live learner source set.
"""
from copy import deepcopy

import torch
from rsl_rl.utils import unpad_trajectories

from wheeled_algo.leg_velocity_separated import LegVelocitySeparatedModel,_SeparatedInference


def height_basis(frame):
    # The current scut35 builder stores requested height*5 at raw index3.
    # Keep this commanded coordinate outside empirical normalization.
    z=(frame[...,3:4]/5.-.305)/.1
    return torch.cat((z,z.square()),dim=-1)


class HeightInitializedModel(LegVelocitySeparatedModel):
    def __init__(self,*args,height_coefficients=None,**kwargs):
        super().__init__(*args,**kwargs)
        value=torch.as_tensor(height_coefficients,dtype=torch.float32)
        if value.shape!=(2,4) or not torch.isfinite(value).all():
            raise ValueError('Explicit2x4 static-mechanics coefficients are required for this fresh candidate')
        self.height_leg_coefficients=torch.nn.Parameter(value.clone())
        self.register_buffer('trainable_height_initialization_abi',torch.tensor([1,3,5,4],dtype=torch.int64))

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        base=super().forward(obs,hidden_state=hidden_state,stochastic_output=False)
        prior=height_basis(obs['policy'])@self.height_leg_coefficients
        output=torch.cat((base[...,:4]+prior,base[...,4:]),dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self):
        return _HeightInference(self)

    def as_onnx(self,verbose=False):
        return _HeightInference(self)


class _HeightInference(torch.nn.Module):
    is_recurrent=False

    def __init__(self,model):
        super().__init__()
        self.base=_SeparatedInference(model)
        self.height_leg_coefficients=torch.nn.Parameter(model.height_leg_coefficients.detach().clone())

    def forward(self,frame:torch.Tensor)->torch.Tensor:
        base=self.base(frame)
        prior=height_basis(frame)@self.height_leg_coefficients
        return torch.cat((base[...,:4]+prior,base[...,4:]),dim=-1)

    @torch.jit.export
    def reset(self)->None:
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1,35),)

    @property
    def input_names(self):
        return ['obs']

    @property
    def output_names(self):
        return ['actions']
