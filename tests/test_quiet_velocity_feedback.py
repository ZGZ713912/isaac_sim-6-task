"""Velocity-only learning must retain old feedback and protect motion/recovery."""
import pytest
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor


def states(count=768):
    obs = torch.randn(count, 35) * .1
    obs[:, 0] = obs[:, 2] = 0.
    obs[:, 9] = -1.
    obs[:, 3] = (torch.tensor([.23, .305, .42]) * 5.).repeat(count // 3)
    return obs


def actor(enabled):
    return RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=True,
                           quiet_native_height_anchors=True, quiet_velocity_feedback=enabled)


def test_zero_velocity_heads_preserve_rng_weights_and_nonzero_existing_feedback():
    torch.manual_seed(37)
    old = actor(False)
    rng = torch.get_rng_state()
    torch.manual_seed(37)
    new = actor(True)
    assert torch.equal(rng, torch.get_rng_state())
    for head in (old.quiet, old.quiet_low, old.quiet_high):
        nn.init.normal_(head[-1].weight, std=.1)
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing.missing_keys) == {f"quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    assert not missing.unexpected_keys
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    x = states()
    assert torch.equal(old(x), new(x))


def test_learned_velocity_path_has_no_constant_bias_and_preserves_inactive_regions():
    old, new = actor(False), actor(True)
    new.load_state_dict(old.state_dict(), strict=False)
    with torch.no_grad():
        for head in new.quiet_velocity.values():
            head.weight.normal_(std=.2)
    x = states()
    x[:, 16:22] = 0.
    assert torch.equal(old(x), new(x))
    for change in ("moving", "fallen"):
        x = states()
        x[:, 0 if change == "moving" else 9] = 1.
        assert torch.equal(old(x), new(x))


@pytest.mark.parametrize("name,height", [("mid", .305), ("low", .23), ("high", .42)])
def test_native_height_endpoint_learns_only_its_own_velocity_head(name, height):
    model = actor(True)
    x = states()
    x[:, 3] = torch.tensor(height) * 5.
    x[:, 16:22] = .03
    model(x).sum().backward()
    for other, head in model.quiet_velocity.items():
        assert head.weight.grad is not None
        if other == name:
            assert head.weight.grad.abs().max() > 0
        else:
            assert torch.count_nonzero(head.weight.grad) == 0


def test_velocity_feedback_retains_strict_unified_initialization_and_requires_experts():
    model = actor(True)
    legacy = {k: v.clone() for k, v in model.state_dict().items()
              if not k.startswith(("quiet.", "quiet_low.", "quiet_high.", "quiet_velocity."))
              and k not in ("quiet_alpha", "quiet_bound")}
    model.initialize_from_unified(legacy)
    del legacy["recovery.0.bias"]
    with pytest.raises(ValueError, match="Unexpected unified-to-robust"):
        model.initialize_from_unified(legacy)
    with pytest.raises(ValueError, match="three-height"):
        RobustGetupActor(nn.Linear(35, 6), quiet_velocity_feedback=True)


def test_velocity_only_learning_protects_all_existing_motion_recovery_and_quiet_weights():
    model = actor(True)
    model.freeze_except_quiet_velocity()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert set(trainable) == {f"quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    assert sum(p.numel() for p in trainable.values()) == 108
    model(states()).square().sum().backward()
    assert all(p.grad is None for name, p in model.named_parameters() if name not in trainable)
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in trainable.values())
    with pytest.raises(ValueError, match="migrated quiet velocity"):
        actor(False).freeze_except_quiet_velocity()
