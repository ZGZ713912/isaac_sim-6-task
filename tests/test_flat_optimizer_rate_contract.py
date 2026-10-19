"""An explicit learning-rate migration must retain the full parent Adam state."""
from copy import deepcopy

import pytest
import torch

from wheeled_algo.flat_optimizer_rates import apply_flat_optimizer_rates as apply_rates


def optimizer(actor_lr=1e-4,critic_lr=5e-4):
    actor=torch.nn.Parameter(torch.ones(2))
    critic=torch.nn.Parameter(torch.ones(3))
    opt=torch.optim.Adam([dict(params=[actor],name="actor",lr=actor_lr),
        dict(params=[critic],name="critic",lr=critic_lr)])
    (actor.square().sum()+critic.square().sum()).backward()
    opt.step()
    return opt


def exact(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def test_loading_the_parent_overrides_constructor_rates_then_declared_change_retains_moments():
    parent=optimizer()
    resumed=optimizer(3e-4)
    resumed.load_state_dict(parent.state_dict())
    assert resumed.param_groups[0]["lr"]==1e-4
    before=deepcopy(resumed.state_dict())
    receipt=apply_rates(resumed,3e-4,5e-4,parent_rates=(1e-4,5e-4))
    expected=deepcopy(before)
    expected["param_groups"][0]["lr"]=3e-4
    assert exact(resumed.state_dict(),expected)
    assert receipt["declared_actor_rate_change"]
    assert receipt["before"]==dict(actor=1e-4,critic=5e-4)
    assert receipt["after"]==dict(actor=3e-4,critic=5e-4)


def test_same_plan_is_unchanged():
    opt=optimizer();before=deepcopy(opt.state_dict())
    receipt=apply_rates(opt,1e-4,5e-4)
    assert exact(before,opt.state_dict())
    assert not receipt["declared_actor_rate_change"]


@pytest.mark.parametrize("actor,critic,parent",[(3e-4,5e-4,None),(3e-4,1e-3,(1e-4,5e-4)),
    (3e-4,5e-4,(2e-4,5e-4)),(3e-4,5e-4,(1e-4,)),(float("nan"),5e-4,(1e-4,5e-4)),
    (float("inf"),5e-4,(1e-4,5e-4)),(-1e-4,5e-4,(1e-4,5e-4)),(True,5e-4,None),
    (3e-4,0,(1e-4,5e-4)),(3e-4,5e-4,(float("nan"),5e-4))])
def test_undeclared_or_invalid_rates_fail_before_mutation(actor,critic,parent):
    opt=optimizer();before=deepcopy(opt.state_dict())
    with pytest.raises(ValueError):
        apply_rates(opt,actor,critic,parent_rates=parent)
    assert exact(before,opt.state_dict())


def test_parameter_group_ownership_is_checked():
    opt=optimizer();opt.param_groups[0]["name"]="unknown"
    before=deepcopy(opt.state_dict())
    with pytest.raises(ValueError):
        apply_rates(opt,3e-4,5e-4,parent_rates=(1e-4,5e-4))
    assert exact(before,opt.state_dict())
