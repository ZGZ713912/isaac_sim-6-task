"""Unadopted fresh-training reward candidate; fixed acceptance is unchanged."""
import math
import torch


def long_tail_scores(height,gravity,velocity,omega,target_height):
    if (gravity.shape!=velocity.shape or omega.shape!=gravity.shape or gravity.shape!=(height.numel(),3)
            or target_height.shape!=height.shape or any(not torch.isfinite(x).all()
                for x in (height,gravity,velocity,omega,target_height))):
        raise ValueError('Need finite synchronized physical state and requested height')
    upright=((1.-gravity[:,2])*.5).clamp(0.,1.)
    attitude=upright*torch.exp(-gravity[:,:2].square().sum(-1)/math.sin(math.radians(5.))**2)
    height_score=torch.exp(-((height-target_height)/.015).square())
    normalized_speed=velocity[:,:2].square().sum(-1)/.03**2+(omega[:,2]/.2).square()
    stationary=1./(1.+normalized_speed)
    angular_cost=.1*torch.log1p((omega[:,:2]/.5).square()).sum(-1)
    return 12.*attitude*height_score*stationary-angular_cost


def joint_long_tail_scores(height,gravity,velocity,omega,target_height):
    """Preserve a joint positive signal when tilt/height also leave the narrow peak."""
    if (gravity.shape!=velocity.shape or omega.shape!=gravity.shape or gravity.shape!=(height.numel(),3)
            or target_height.shape!=height.shape or any(not torch.isfinite(x).all()
                for x in (height,gravity,velocity,omega,target_height))):
        raise ValueError('Need finite synchronized physical state and requested height')
    upright=((1.-gravity[:,2])*.5).clamp(0.,1.)
    error=(gravity[:,:2].square().sum(-1)/math.sin(math.radians(5.))**2
        +((height-target_height)/.015).square()+velocity[:,:2].square().sum(-1)/.03**2
        +(omega[:,2]/.2).square())
    angular_cost=.1*torch.log1p((omega[:,:2]/.5).square()).sum(-1)
    return 12.*upright/(1.+error)-angular_cost
