"""Explicit causal motion target smoothing in the existing 35D actor contract."""
from copy import deepcopy
import math

import torch

from .gap_entry_handoff import handoff_entry_gate
from .paired_recovery import wrap
from .periodic_entry_actor import PeriodicEntryGapActor, exact_state_equal
from .robust_getup import tilt_gate

VERSION = 'motion-leg-damping-v1'
MODULE = 'src/wheeled_algo/motion_target_damping.py'


def validate_alpha(alpha):
    if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not .05<=alpha<=1.:
        raise ValueError('Motion leg alpha must be finite within[.05,1]')


def motion_damping_gate(obs,stance_aware=True,shape_aware=True):
    moving = torch.maximum(obs[:,:1].abs()/.12,obs[:,2:3].abs()/.25).clamp(0.,1.)
    recovery = torch.maximum(handoff_entry_gate(obs),tilt_gate(obs,stance_aware,shape_aware))
    upright = ((-obs[:,9:10]-.9)/.06).clamp(0.,1.)
    return moving*upright*(1.-recovery)


def damp_motion_targets(action,obs,alpha,action_bound,stance_aware=True,shape_aware=True):
    """Use only the previous applied action; wheels and inactive paths are exact."""
    validate_alpha(alpha)
    if alpha==1.:
        return action
    gate = motion_damping_gate(obs,stance_aware,shape_aware)
    current,previous = action[:,:4]*.25,obs[:,22:26]*.25
    proposed = previous+alpha*wrap(current-previous)
    error = wrap(proposed-obs[:,10:14])
    magnitude = error.abs()
    shaped = error.sign()*(.06+.06*torch.tanh((magnitude-.06)/.06))
    # Limit the proposed endpoint before blending: zero gate remains continuous
    # and cannot apply the existing soft limiter twice to an inactive target.
    error = torch.where(magnitude>.06,shaped,error)
    proposed = obs[:,10:14]+error
    blended = current+gate*wrap(proposed-current)
    legs = blended/.25
    legs = torch.maximum(torch.minimum(legs,action_bound[:4]),-action_bound[:4])
    return torch.cat((torch.where(gate>0.,legs,action[:,:4]),action[:,4:]),-1)


class MotionTargetDampedGapActor(PeriodicEntryGapActor):
    def __init__(self,*args,motion_leg_alpha,**kwargs):
        super().__init__(*args,**kwargs)
        validate_alpha(motion_leg_alpha)
        if self.final_target_winding_version!='paired-v1' or not self.cyclic_leg_targets:
            raise ValueError('Motion damping requires paired cyclic final targets')
        self.motion_leg_alpha = float(motion_leg_alpha)

    def forward(self,obs):
        old = super().forward(obs)
        if self.motion_leg_alpha==1.:
            return old
        result = damp_motion_targets(old,obs,self.motion_leg_alpha,self.action_bound,
                                     self.stance_aware_recovery,self.recovery_shape_aware)
        gate = motion_damping_gate(obs,self.stance_aware_recovery,self.recovery_shape_aware)
        legs = self.finalize_leg_pairs(result[:,:4])
        return torch.cat((torch.where(gate>0.,legs,old[:,:4]),old[:,4:]),-1)


def validate_motion_damping_checkpoint(state,source):
    """No weights, optimizer, clock, RNG or unrelated options may change."""
    info,prior = state['infos'],source['infos']
    alpha = info['settings'].get('motion_leg_damping_alpha')
    validate_alpha(alpha)
    record = info.get('motion_leg_damping_migration',{})
    if (prior['settings'].get('motion_leg_damping_version') is not None
            or prior['settings'].get('gap_entry_encoder_version')!='periodic-chart-v1'
            or info['settings'].get('motion_leg_damping_version')!=VERSION
            or record.get('schema')!='motion-leg-damping-migration-v1'
            or record.get('new_PPO')!=0
            or record.get('learning_clock')!=info['completed_updates']
            or info['completed_updates']!=prior['completed_updates']):
        raise ValueError('Motion target damping requires an explicit zero-update periodic-actor migration')
    if not exact_state_equal({k:v for k,v in state.items() if k!='infos'},
                             {k:v for k,v in source.items() if k!='infos'}):
        raise ValueError('Motion damping migration changed weights, optimizer or saved state')
    expected = deepcopy(prior)
    expected['settings']['motion_leg_damping_version'] = VERSION
    expected['settings']['motion_leg_damping_alpha'] = alpha
    expected['source_sha256'][MODULE] = info['source_sha256'].get(MODULE)
    expected['motion_leg_damping_migration'] = record
    expected['hardware_ready'] = False
    if not exact_state_equal(info,expected):
        raise ValueError('Motion damping migration changed unrelated checkpoint metadata')
