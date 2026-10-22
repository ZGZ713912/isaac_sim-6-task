"""New equilibrium feedback must retain knee geometry and inactive behavior."""
import pytest
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor


def actor(enabled):
    return RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=True,
        quiet_native_height_anchors=True, quiet_velocity_feedback=enabled,
        quiet_velocity_feedback_mode="coupled_state" if enabled else "all")


def states():
    x = torch.randn(192, 35) * .03
    x[:, 0] = x[:, 2] = 0.
    x[:, 9] = -1.
    x[:, 3] = (torch.tensor([.23, .305, .42]) * 5.).repeat(64)
    return x


def test_zero_state_feedback_preserves_rng_old_weights_and_full_actions():
    torch.manual_seed(415)
    old = actor(False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(415)
    new = actor(True)
    assert torch.equal(rng, torch.get_rng_state())
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing.missing_keys) == {
        f"quiet_velocity.{name}.{field}" for name in ("mid", "low", "high") for field in ("weight", "bias")}
    assert not missing.unexpected_keys
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    x = states()
    assert torch.equal(old(x), new(x))


def learned_pair():
    old, new = actor(False), actor(True)
    for head in (old.quiet, old.quiet_low, old.quiet_high):
        nn.init.normal_(head[-1].weight, std=.1)
    new.load_state_dict(old.state_dict(), strict=False)
    with torch.no_grad():
        for head in new.quiet_velocity.values():
            head.weight.normal_(std=.1)
            head.bias.normal_(std=.02)
    return old, new


def test_learned_common_leg_correction_preserves_each_paired_knee_angle():
    old, new = learned_pair()
    x = states()
    delta = new(x) - old(x)
    assert delta.abs().max() > 0.
    torch.testing.assert_close(delta[:, 0], delta[:, 1], rtol=0., atol=3e-7)
    torch.testing.assert_close(delta[:, 2], delta[:, 3], rtol=0., atol=3e-7)


@pytest.mark.parametrize("kind", ["moving", "turning", "fallen", "fast_gyro"])
def test_new_state_feedback_preserves_inactive_control_exactly(kind):
    old, new = learned_pair()
    x = states()
    x[:, {"moving": 0, "turning": 2, "fallen": 9, "fast_gyro": 4}[kind]] = 1.
    assert torch.equal(old(x), new(x))


def test_new_feedback_has_no_direct_previous_action_recurrence_and_can_hold_offset():
    old, new = learned_pair()
    x = states()
    delta = new(x) - old(x)
    changed = x.clone()
    changed[:, 22:28] += .2
    torch.testing.assert_close(new(changed) - old(changed), delta, rtol=0., atol=3e-7)
    with torch.no_grad():
        for head in new.quiet_velocity.values():
            head.weight.zero_()
            head.bias.fill_(.05)
    x[:, 16:22] = 0.
    assert (new(x) - old(x)).abs().max() > 0.


@pytest.mark.parametrize("name,height", [("mid", .305), ("low", .23), ("high", .42)])
def test_only_selected_height_head_receives_gradients_and_old_actor_is_frozen(name, height):
    model = actor(True)
    model.freeze_except_quiet_velocity()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in trainable.values()) == 204
    assert len(trainable) == 6
    x = states()
    x[:, 3] = torch.tensor(height) * 5.
    model(x).sum().backward()
    for other, head in model.quiet_velocity.items():
        for parameter in head.parameters():
            assert parameter.grad is not None
            assert bool(parameter.grad.abs().max() > 0.) == (other == name)
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)
