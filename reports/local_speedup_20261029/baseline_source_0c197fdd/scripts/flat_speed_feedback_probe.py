"""Zero-PPO hypotheses for separate yaw feedback and already-rolling reverse."""
import torch


def coupled_actions(obs,actor,arm):
    rolling=(obs[:,20:21]-obs[:,21:22])*.3
    gyro=obs[:,6:7]*2.
    command_gate=torch.maximum(obs[:,:1].abs()/.75,obs[:,2:3].abs()/1.5)
    motion_gate=torch.maximum(rolling.abs()/.75,gyro.abs()/1.5)
    gate=(torch.maximum(command_gate,motion_gate)-1.).clamp(0.,1.)
    height=((obs[:,3:4]/5.-.32)/.1).clamp(0.,1.)
    aligned=rolling*obs[:,:1].sign()
    progress=((aligned-1.3)/1.).clamp(0.,1.)
    strength=torch.where(arm<1.5,0.,torch.where(arm<2.5,-.04,-.08))
    bound=torch.where(arm<2.5,.015,.025)
    angle=torch.maximum(torch.minimum(strength*(obs[:,:1]-rolling),bound),-bound)
    angle=angle*height*progress*gate*(obs[:,:1]<-1.5)
    gx,gz=obs[:,7:8],obs[:,9:10]
    reference=torch.cat((obs[:,:7],gx*angle.cos()-gz*angle.sin(),obs[:,8:9],
        gx*angle.sin()+gz*angle.cos(),obs[:,10:]),-1)
    action=actor(reference)
    # Equal raw increments on the oppositely signed wheel axes change yaw;
    # their forward/backward difference stays unchanged before motor limits.
    delta=(-.1*(obs[:,2:3]-gyro)*(.4373/2.)/.6).clamp(-.08,.08)*gate
    delta=torch.where(arm>0.,delta,torch.zeros_like(delta))
    candidate=torch.cat((action[:,:4],action[:,4:]+delta),-1)
    return torch.where(arm<=0.,actor(obs),candidate)
