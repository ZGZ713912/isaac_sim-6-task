"""Serialized opt-in filter agrees with the diagnostic and keeps parameter keys."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter
from wheeled_algo.quiet_output_actor import QuietOutputGetupActor
from wheeled_algo.robust_getup import RobustGetupActor


@pytest.mark.parametrize("alpha", [.1, .25, .5, 1.])
def test_constructor_preserves_state_names_and_diagnostic_function(alpha):
    torch.manual_seed(74)
    parent = nn.Linear(35, 6)
    base = RobustGetupActor(parent, leg_action_clip=16., cyclic_leg_targets=True,
                           smooth_knee_targets=True, stance_aware_recovery=True, recovery_shape_aware=True)
    candidate = QuietOutputGetupActor(deepcopy(parent), quiet_output_wheel_alpha=alpha,
        leg_action_clip=16., cyclic_leg_targets=True, smooth_knee_targets=True,
        stance_aware_recovery=True, recovery_shape_aware=True)
    candidate.load_state_dict(base.state_dict(), strict=True)
    assert candidate.state_dict().keys() == base.state_dict().keys()
    assert [(k, p.shape) for k, p in candidate.named_parameters()] == [(k, p.shape) for k, p in base.named_parameters()]
    obs = torch.randn(500, 35) * .1
    obs[:250, :3] = 0.
    obs[:250, 4:6] = 0.
    obs[:250, 9] = -1.
    expected = QuietWheelOutputFilter(base, alpha)(obs)
    assert torch.equal(candidate(obs), expected)
    candidate(obs).sum().backward()
    assert torch.isfinite(candidate.quiet[-1].weight.grad).all()
    assert candidate.quiet[-1].weight.grad.abs().sum() > 0.
