"""Position ABI isolation, learned six-motor coordination and matched noise."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor, ZeroStandingPosition
from wheeled_algo.standing_displacement import observation_contract, validate_observation_contract
from wheeled_tasks.chassis.robust_getup import RobustDomain, settings


def actor(enabled):
    return RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=True,
        quiet_native_height_anchors=True, quiet_velocity_feedback=enabled,
        quiet_velocity_feedback_mode="standing_displacement" if enabled else "all")


def states():
    x = torch.randn(192, 35) * .03
    x[:, :3] = 0.
    x[:, 3] = (torch.tensor([.23, .305, .42])*5.).repeat(64)
    x[:, 7:10] = torch.tensor([0., 0., -1.])
    x[:, 14:16] = 0.
    return x


def test_zero_feedback_ignores_new_slots_in_all_legacy_functions_without_rng_draws():
    torch.manual_seed(5601)
    old = actor(False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(5601)
    new = actor(True)
    assert torch.equal(rng, torch.get_rng_state())
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert set(missing.missing_keys) == {f"quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    x = states()
    for index, value in ((0, 0.), (0, 2.), (9, 1.)):
        x[:, index] = value
        expected = old(x)
        changed = x.clone()
        changed[:, 14:16] = torch.rand(192, 2)*10.-5.
        assert torch.equal(new(changed), expected)


def learned():
    old, new = actor(False), actor(True)
    new.load_state_dict(old.state_dict(), strict=False)
    with torch.no_grad():
        for head in new.quiet_velocity.values():
            head.weight.copy_(torch.tensor([[.3, .3], [.3, .3], [.4, .1], [-.1, -.4]]))
    return old, new


def test_learned_position_correction_preserves_same_leg_shape_and_is_zero_at_reference():
    old, new = learned()
    x = states()
    assert torch.equal(old(x), new(x))
    altered = x.clone()
    altered[:, 14:16] = .4
    difference = new(altered)-old(x)
    assert difference.abs().max() > 0.
    torch.testing.assert_close(difference[:, 0], difference[:, 1], rtol=0., atol=2e-7)
    torch.testing.assert_close(difference[:, 2], difference[:, 3], rtol=0., atol=2e-7)
    assert bool((difference.abs() <= .5*new.quiet_bound+2e-7).all())
    for index, value in ((0, .5), (2, 1.), (9, 1.), (4, 1.)):
        before, after = x.clone(), altered.clone()
        before[:, index] = after[:, index] = value
        assert torch.equal(new(after), old(before))


@pytest.mark.parametrize("name,height", [("mid", .305), ("low", .23), ("high", .42)])
def test_only_twenty_four_position_values_learn_with_local_height_gradients(name, height):
    model = actor(True)
    model.freeze_except_quiet_velocity()
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(parameters) == 3 and sum(p.numel() for p in parameters.values()) == 24
    x = states()
    x[:, 3] = torch.tensor(height)*5.
    x[:, 14:16] = .5
    model(x).sum().backward()
    for other, head in model.quiet_velocity.items():
        assert bool(head.weight.grad.abs().max() > 0.) == (other == name)
    assert all(p.grad is None for n,p in model.named_parameters() if n not in parameters)


def test_changed_abi_requires_the_complete_versioned_contract():
    cfg = dict(quiet_velocity_feedback=True, quiet_velocity_feedback_mode="standing_displacement",
               standing_position_observation=observation_contract())
    assert validate_observation_contract(cfg)
    for key in observation_contract():
        damaged = deepcopy(cfg)
        del damaged["standing_position_observation"][key]
        with pytest.raises(ValueError, match="exact versioned"):
            validate_observation_contract(damaged)
    for change in ({"quiet_velocity_feedback": False}, {"quiet_velocity_feedback_mode": "all"},
                   {"standing_position_observation": None}):
        with pytest.raises(ValueError, match="Standing displacement"):
            validate_observation_contract(dict(cfg, **change))
    assert not validate_observation_contract({})


def test_wheel_angle_noise_uses_existing_draws_and_same_delay_without_other_domain_changes():
    a, b = [RobustDomain(24, "cpu", settings(domain_scale=.35, **extra), 5601, .001)
            for extra in ({}, {"quiet_velocity_feedback_mode": "standing_displacement"})]
    ids = torch.arange(24)
    source = torch.zeros(24, 18)
    source[:, 5] = -1.
    source[:, 10:12] = torch.tensor([21., -22.])
    a.reset(ids, source)
    b.reset(ids, source)
    for name in ("enabled", "strength", "kp", "kd", "spring", "torque_delay", "sensor_delay", "motor_alpha", "bias"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert torch.equal(a.rng.get_state(), b.rng.get_state())
    keep = list(range(10))+list(range(12, 18))
    for tick in range(4):
        first, second = a.observe(source, tick), b.observe(source, tick)
        assert torch.equal(first[:, keep], second[:, keep])
        assert torch.equal(a.rng.get_state(), b.rng.get_state())
        assert torch.equal(b.observe(source+3., tick), second)
    assert b.sigma[:, 10:12].max() > 0.
    assert a.sigma[:, 10:12].max() == 0.


def test_explicit_parent_comparison_keeps_its_original_zero_slots():
    model = nn.Linear(35, 6)
    x = states()
    expected = model(x)
    x[:, 14:16] = 4.
    assert not torch.equal(model(x), expected)
    assert torch.equal(ZeroStandingPosition(model)(x), expected)
