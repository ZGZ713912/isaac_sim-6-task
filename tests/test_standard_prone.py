"""Physical40deg guidance, unconstrained standing legs and true completion."""
import json
import math
from pathlib import Path

import torch

from wheeled_tasks.chassis.standard_prone import physical_knee_degrees, route_potential, stable, support_error, valid_disabled_start

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "contracts/wheel_leg_standard_prone_v1.json").read_text())


def test_passive_included_angle_respects_opposing_joint_signs_and_model_zero():
    lower = -.08616
    at_lower = torch.tensor([[lower, -lower]])
    torch.testing.assert_close(physical_knee_degrees(at_lower, lower), torch.full((1, 2), 40.))
    at_upper = at_lower + torch.tensor([[math.radians(67.), -math.radians(67.)]])
    torch.testing.assert_close(physical_knee_degrees(at_upper, lower), torch.full((1, 2), 107.))


def potential(knee, alignment=.2, standing=False):
    gravity = torch.tensor([[0., 0., -1.]])
    height = .305 if standing else SPEC["belly_height_m"]
    return route_potential(gravity, torch.tensor([[knee, knee]]), torch.tensor([alignment]), torch.tensor([height]), SPEC)


def test_fold_and_under_com_improvements_receive_dense_credit_from_extended_legs():
    assert potential(107.)[0] < potential(85.)[0] < potential(65.)[0] < potential(43.)[0]
    assert potential(65., .3)[0] < potential(65., .2)[0] < potential(65., .1)[0]
    assert potential(107.)[1]["fold_score"].item() > .01


def test_reward_near_stop_does_not_prefer_driving_through_the40deg_stop():
    torch.testing.assert_close(potential(40.)[0], potential(43.)[0])
    assert potential(39.)[0] < potential(40.)[0]


def test_folding_guidance_stays_on_for_belly_grounded_upright_body_but_fades_at_standing_height():
    assert potential(40.)[0] > potential(100.)[0]
    torch.testing.assert_close(potential(40., standing=True)[0], potential(100., standing=True)[0])
    assert not potential(40., standing=True)[1]["folding_guidance_weight"].any()


def test_horizontal_alignment_above_the_com_does_not_count_as_support_below_it():
    com = torch.tensor([[0., 0., .3], [0., 0., .3]])
    wheel = torch.tensor([[0., 0., .1], [0., 0., .5]])
    error = support_error(wheel, com, SPEC["support_min_below_com_m"])
    assert error[0] == 0. and error[1] > .2


def test_raising_from_folded_belly_to_unfolded_standing_increases_route_potential():
    assert potential(40., .02)[0] < potential(100., .02, standing=True)[0]


def test_upright_attitude_without_height_contact_and_quiet_motion_cannot_complete():
    gravity = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    height = torch.tensor([.305, .23, .305, .305])
    velocity = torch.zeros(4, 3)
    velocity[3, 0] = .2
    contact = torch.ones(4, 2, dtype=torch.bool)
    contact[2, 0] = False
    result = stable(gravity, height, torch.zeros(4, 3), velocity, contact, torch.zeros(4, 4), SPEC)
    assert result.tolist() == [True, False, False, False]


def test_disabled_start_rejects_front_inverted_standing_and_body_not_grounded():
    gravity = torch.tensor([[0., 0., -1.], [1., 0., 0.], [0., 0., 1.], [0., 0., -1.], [0., 0., -1.]])
    height = torch.tensor([.173, .173, .173, .305, .173])
    force = torch.tensor([100., 100., 100., 100., 0.])
    assert valid_disabled_start(gravity, height, force, SPEC).tolist() == [True, False, False, False, False]
