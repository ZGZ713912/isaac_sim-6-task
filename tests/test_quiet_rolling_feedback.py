"""Zero migration, physical geometry ownership, and trainable motor coordination."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor


def actor(enabled):
    return RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=True,
        quiet_native_height_anchors=True, quiet_velocity_feedback=enabled,
        quiet_velocity_feedback_mode="rolling_velocity" if enabled else "all")


def states():
    x = torch.zeros(192, 35)
    x[:, 3] = (torch.tensor([.23, .305, .42]) * 5.).repeat(64)
    x[:, 9] = -1.
    x[:, 20] = .05
    x[:, 21] = -.05
    return x


def test_zero_rolling_feedback_preserves_rng_existing_weights_and_actions():
    torch.manual_seed(501)
    old = actor(False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(501)
    new = actor(True)
    assert torch.equal(rng, torch.get_rng_state())
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(n.startswith(("quiet_velocity.", "rolling_velocity.")) for n in missing.missing_keys)
    assert len([n for n in missing.missing_keys if n.startswith("quiet_velocity.")]) == 3
    assert torch.equal(old(states()), new(states()))
    new.rolling_velocity.verify_frozen_geometry()


def learned():
    old, new = actor(False), actor(True)
    new.load_state_dict(old.state_dict(), strict=False)
    with torch.no_grad():
        for head in new.quiet_velocity.values():
            head.weight.fill_(.3)
    return old, new


def test_correction_has_same_leg_common_rotation_and_no_constant_bias():
    old, new = learned()
    x = states()
    delta = new(x) - old(x)
    assert delta.abs().max() > 0.
    torch.testing.assert_close(delta[:, 0], delta[:, 1], rtol=0., atol=2e-7)
    torch.testing.assert_close(delta[:, 2], delta[:, 3], rtol=0., atol=2e-7)
    x[:, 16:22] = 0.
    assert torch.equal(new(x), old(x))


@pytest.mark.parametrize("index,value", [(0, .5), (2, 1.), (9, 1.), (4, 1.)])
def test_inactive_motion_recovery_or_disturbance_remains_exact(index, value):
    old, new = learned()
    x = states()
    x[:, index] = value
    assert torch.equal(old(x), new(x))


@pytest.mark.parametrize("name,height", [("mid", .305), ("low", .23), ("high", .42)])
def test_only_twelve_feedback_weights_can_learn_and_height_gradients_are_local(name, height):
    model = actor(True)
    model.freeze_except_quiet_velocity()
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(params) == 3 and sum(p.numel() for p in params.values()) == 12
    x = states()
    x[:, 3] = torch.tensor(height) * 5.
    model(x).sum().backward()
    for other, head in model.quiet_velocity.items():
        assert head.weight.grad is not None
        assert bool(head.weight.grad.abs().max() > 0.) == (other == name)
    assert all(p.grad is None for n,p in model.named_parameters() if n not in params)


def test_geometry_is_frozen_and_tampered_checkpoint_buffers_rejected():
    model = actor(True)
    state = deepcopy(model.state_dict())
    state["rolling_velocity.nominal"][0] += .01
    model.load_state_dict(state)
    with pytest.raises(ValueError, match="geometry buffers changed"):
        model.rolling_velocity.verify_frozen_geometry()


def test_invalid_rolling_geometry_and_previous_action_input_cannot_add_correction():
    old, new = learned()
    x = states()
    x[:, 11] = 3.
    assert torch.equal(old(x), new(x))
    x = states()
    delta = new(x) - old(x)
    x[:, 22:28] = .2
    torch.testing.assert_close(new(x) - old(x), delta, rtol=0., atol=2e-7)
