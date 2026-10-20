"""Probe a reachable braking reference before stationary feedback captures."""
import torch

from flat_speed_feedback_probe import coupled_actions
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor


def braking_actions(obs,actor,arm,base_arm):
    rolling=(obs[:,20:21]-obs[:,21:22])*.3
    low_request=(obs[:,:1].abs()<=.75)&(obs[:,2:3].abs()<=1.5)
    decelerating=(rolling*(obs[:,:1]-rolling)<0.)&low_request
    margin=.3
    local=rolling+(obs[:,:1]-rolling).clamp(-margin,margin)
    blend=((rolling.abs()-.3)/.5).clamp(0.,1.)
    goal=obs[:,:1]+blend*(local-obs[:,:1])
    goal=torch.where((arm>0.)&decelerating,goal,obs[:,:1])
    query=torch.cat((goal,obs[:,1:]),-1)
    action=coupled_actions(query,actor,torch.where(arm>0.,float(base_arm),0.))
    moving=LowDemandFeedbackActor.feedback_actions(actor,query)
    correction=moving-actor.feedback_actions(query)
    state_gate=((rolling.abs()-.03)/.12).clamp(0.,1.)
    # For a low query v10's internal reference is the query itself. Replace
    # only its prematurely static feedback; the original policy still balances.
    eligible=low_request&(goal.abs()<=.75)&(arm>0.)
    action=action+torch.where(eligible,state_gate,torch.zeros_like(state_gate))*correction
    # The earlier common-wheel prior was harmful during launch. This version
    # acts only on the measured steady reverse deficit after aligned rolling.
    progress=((rolling*obs[:,:1].sign()-1.5)/.7).clamp(0.,1.)
    height=((obs[:,3:4]/5.-.32)/.1).clamp(0.,1.)
    gain=torch.where(arm<1.5,0.,torch.where(arm<2.5,.03,.06))
    bound=torch.where(arm<2.5,.03,.05)
    delta=gain*(obs[:,:1]-rolling)/.6
    delta=torch.maximum(torch.minimum(delta,bound),-bound)
    delta=delta*progress*height*(obs[:,:1]<-1.5)
    action=torch.cat((action[:,:4],action[:,4:]+torch.cat((delta,-delta),-1)),-1)
    return action
