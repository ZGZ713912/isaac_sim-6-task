"""Effective contact ranges must survive material combination and trial order."""
import numpy as np
import pytest

from wheeled_tasks.chassis.traditional_contact import TraditionalContactDomain


def test_effective_random_friction_and_zero_roller_friction():
    domain = TraditionalContactDomain((.2, .4), 42)
    samples = np.array([domain.sample(case) for case in range(200)])
    assert samples.min() >= .2 and samples.max() <= .4 and samples.std() > .04
    assert domain.combine_mode == "multiply"
    for mu in samples:
        assert domain.body_coefficient("L_link3", mu)*domain.ground_coefficient == mu
        assert domain.body_coefficient("R_link3", mu)*domain.ground_coefficient == mu
        assert domain.body_coefficient("base_link", mu)*domain.ground_coefficient == 0.
    assert domain.body_coefficient("L_link1", .3) == .8


def test_candidate_order_and_early_termination_do_not_change_case_draw():
    a = TraditionalContactDomain((.2, .4), 42)
    b = TraditionalContactDomain((.2, .4), 42)
    reference = {case:a.sample(case) for case in range(6)}
    a.sample(100)
    for case in (5, 2, 0, 4, 1, 3):
        assert a.sample(case) == b.sample(case) == reference[case]
    assert TraditionalContactDomain((.2, .4), 43).sample(0) != reference[0]


@pytest.mark.parametrize("band", [[.4,.2], [-.1,.3], [.2,1.1], [.2,float("nan")], [.3]])
def test_invalid_friction_domain_is_rejected(band):
    with pytest.raises(ValueError):
        TraditionalContactDomain(band, 42)
