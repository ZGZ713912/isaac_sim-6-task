import math

import pytest
import torch

from wheeled_tasks.chassis.grounded_standing_reset import cylinder_clearance, grounded_pose
from wheeled_tasks.chassis.standing_contact_reward import contract, support_density, validate_contract


def test_grounding_corrects_both_initial_gaps_without_changing_xy():
    root = torch.tensor([[2., 3., .3, 0., 0., 0., 1.]], dtype=torch.float64)
    wheels = torch.tensor([[[2., 3.2, .068], [2., 2.8, .070]]], dtype=torch.float64)
    quat = torch.tensor([[[math.sqrt(.5), 0., 0., math.sqrt(.5)]]], dtype=torch.float64).expand(1, 2, 4)
    result, roll, _ = grounded_pose(root, wheels, quat, torch.zeros(1, dtype=torch.float64))
    relative = wheels - root[:, None, :3]
    axes = root.new_tensor([[[0., -1., 0.], [0., -1., 0.]]])
    final = cylinder_clearance(relative, axes, result[:, 2], root.new_zeros(1), root.new_tensor([[1., 0., 0.]]), roll)
    torch.testing.assert_close(final, root.new_full((1, 2), -.00005), atol=1e-10, rtol=0.)
    assert torch.equal(result[:, :2], root[:, :2]) and abs(float(roll)) < .01
    torch.testing.assert_close(result[:, 3:].norm(dim=-1), root.new_ones(1))


def test_support_cost_detects_a_single_bad_ms_and_worst_wheel():
    force = torch.zeros(5, 2, 2, 3)
    force[..., 2] = 70.
    clearance = torch.full((5, 2, 2), -.00005)
    assert torch.equal(support_density(force, clearance), torch.zeros(2))
    force[1, 0, 1, 2] = 0.
    torch.testing.assert_close(support_density(force, clearance), torch.tensor([.2, 0.]))
    clearance[2, 1, 0] = .0002
    torch.testing.assert_close(support_density(force, clearance), torch.tensor([.2, math.log(2.) / 5.]))


def test_adding_lateral_force_does_not_hide_missing_vertical_support():
    force = torch.zeros(5, 1, 2, 3)
    force[..., 0] = 500.
    assert support_density(force, torch.zeros(5, 1, 2)).item() == 1.


def test_bad_reward_units_and_nonfinite_contacts_are_rejected():
    for weight in (True, float('nan'), -1., 17.):
        with pytest.raises(ValueError):
            contract(weight)
    with pytest.raises(ValueError):
        support_density(torch.full((5, 1, 2, 3), float('nan')), torch.zeros(5, 1, 2))


def test_training_contract_rejects_changed_objective_reset_and_sources():
    value = contract(8., True)
    validate_contract(value, value['module_sha256'])
    with pytest.raises(ValueError):
        validate_contract(dict(value, force_margin_n=2.), value['module_sha256'])
    with pytest.raises(ValueError):
        validate_contract(value, dict(value['module_sha256'], **{next(iter(value['module_sha256'])): 'wrong'}))
    with pytest.raises(ValueError):
        contract(8., 'true')
