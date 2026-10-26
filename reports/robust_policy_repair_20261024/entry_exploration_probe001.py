"""Private sampled-policy diagnostic; does not modify deterministic actor code."""
import math
import torch
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

def alter_entry_std(output,obs,ratio):
    if ratio not in (1.,.25):
        raise ValueError('Only the declared control1 and entry-leg quarter-std probe are supported')
    if ratio==1.:return output
    entry=handoff_entry_gate(obs)
    factor=1.-(1.-ratio)*entry
    log_std=torch.cat((output[:,1,:4]+factor.log(),output[:,1,4:]),-1)
    return torch.stack((output[:,0],log_std),-2)
