"""Policy-only gyro-release counterfactual; observer and sampling stay native."""
from contextlib import contextmanager
from types import FunctionType
import math
import torch
from wheeled_algo import robust_getup,quiet_output_actor
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

VERSION='policy-quiet-gyro-stance-authority-probe-v2'
NATIVE_GATE=robust_getup.quiet_gate

def probe_gate(obs,floor):
    if type(floor) not in (int,float) or not math.isfinite(floor) or not 0.<=floor<=1.:
        raise ValueError('Invalid retained quiet authority')
    base=NATIVE_GATE(obs)
    if floor==0.:return base
    request=torch.maximum(obs[:,:1].abs()/.12,obs[:,2:3].abs()/.25)
    tilt=((-obs[:,9:10]-.94)/.04).clamp(0.,1.)
    gyro=(obs[:,4:6].norm(dim=-1,keepdim=True)*2.-.4).clamp(0.,1.)
    recovery=torch.maximum(handoff_entry_gate(obs),robust_getup.tilt_gate(obs,True,True))
    stance=(1.-recovery/.05).clamp(0.,1.)
    added=floor*(1.-request).clamp(0.,1.)*tilt*gyro*stance
    return torch.where(added>0.,base+added,base)

@contextmanager
def policy_authority(floor):
    # Clone only the two policy method globals. Do not replace the shared
    # quiet_gate function used by the standing observer or native distribution.
    original=[(cls,cls.forward) for cls in (robust_getup.RobustGetupActor,quiet_output_actor.QuietOutputGetupActor)]
    gate=lambda obs:probe_gate(obs,floor)
    try:
        for cls,method in original:
            globals_copy=dict(method.__globals__,quiet_gate=gate)
            clone=FunctionType(method.__code__,globals_copy,method.__name__,method.__defaults__,method.__closure__)
            clone.__kwdefaults__=method.__kwdefaults__
            cls.forward=clone
        yield
    finally:
        for cls,method in original:cls.forward=method
