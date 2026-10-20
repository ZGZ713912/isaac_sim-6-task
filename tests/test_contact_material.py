import pytest
import torch

from wheeled_tasks.chassis.contact_material import ContactMaterialDomain, combine


def config(ground=(.2, .3)):
    return {
        "enabled": True,
        "combine_mode": "average",
        "ground_static_friction_range": list(ground),
        "ground_dynamic_friction_range": list(ground),
        "wheel_static_friction": .1,
        "wheel_dynamic_friction": .1,
        "body_static_friction": .25,
        "body_dynamic_friction": .25,
    }


def test_low_grip_domain_samples_ground_and_keeps_wheel_material_fixed():
    domain = ContactMaterialDomain(4096, config(), 191019)
    assert float(domain.ground_static.min()) >= .2
    assert float(domain.ground_static.max()) <= .3
    static, dynamic = domain.effective(domain.ground_static)
    assert torch.allclose(static, (domain.ground_static + .1) / 2.)
    assert torch.equal(static, dynamic)
    report = domain.summary()
    assert report["wheel_static_friction"] == .1
    assert report["effective_wheel_static_min_max"][1] <= .2


def test_randomized_training_band_is_reproducible_and_separate_from_wheel():
    a = ContactMaterialDomain(128, config((.3, .4)), 191019)
    b = ContactMaterialDomain(128, config((.3, .4)), 191019)
    assert torch.equal(a.ground_static, b.ground_static)
    assert float(a.ground_static.min()) >= .3
    assert float(a.ground_static.max()) <= .4
    assert a.wheel_static == .1


@pytest.mark.parametrize("mode, expected", [
    ("average", .2), ("min", .1), ("multiply", .03), ("max", .3)])
def test_physx_combine_modes_are_explicit(mode, expected):
    result = combine(torch.tensor(.3), torch.tensor(.1), mode)
    assert float(result) == pytest.approx(expected)


def test_dynamic_friction_cannot_exceed_static():
    bad = config()
    bad["wheel_dynamic_friction"] = .2
    with pytest.raises(ValueError, match="Dynamic friction"):
        ContactMaterialDomain(2, bad, 1)


def test_second_stage_mixture_contains_01_low_and_higher_friction_ground():
    cfg = config((.1, .4))
    cfg["ground_sampling_bands"] = [
        {"range": [.1, .1], "fraction": 1 / 3},
        {"range": [.2, .3], "fraction": 1 / 3},
        {"range": [.3, .4], "fraction": 1 / 3}]
    domain = ContactMaterialDomain(4096, cfg, 191019)
    x = domain.ground_static
    assert (x == .1).any() and ((x >= .2) & (x <= .3)).any() and (x > .3).any()
    assert ((x == .1) | ((x >= .2) & (x <= .4))).all()
    assert torch.equal(x, domain.ground_dynamic)
    assert domain.wheel_static == .1
    domain.reorder(list(reversed(range(len(x)))))
    torch.testing.assert_close(domain.ground_static, x.flip(0))


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(typo=.1),
    lambda c: c.update(ground_sampling_bands=[{"range": [.1, .1], "fraction": 1.}]),
    lambda c: c.update(ground_sampling_bands=[{"range": [.2, .3], "fraction": .2}]),
])
def test_material_typos_and_invalid_band_mixtures_fail_closed(mutate):
    cfg = config()
    mutate(cfg)
    with pytest.raises(ValueError):
        ContactMaterialDomain(8, cfg, 1)
