"""Independent interface/physics semantics; native behavior is tested separately."""
import math

import pytest
import torch
from torch import nn

from wheeled_algo.unified_getup import UnifiedGetupActor, recovery_gate
from wheeled_tasks.chassis.unified_getup import classify_gravity, joint_tracking_score, project_commands, settings, push_window


def test_rmcs_projection_keeps_yaw_and_reduces_translation():
    command = torch.tensor([[3., 4., .305], [-3., -4., .305], [1., 1., .305], [0., 4., .305]])
    result = project_commands(command, settings())
    torch.testing.assert_close(result[:, 1:], command[:, 1:])
    torch.testing.assert_close(result[:2, 0], torch.tensor([2.06 / 4, -2.06 / 4]))
    torch.testing.assert_close(result[2:], command[2:])


def test_joint_tracking_requires_both_axes():
    score = joint_tracking_score(torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]))
    assert score[0] == 1 and score[1] < .1 and score[2] < .2


def test_actual_settled_coverage_does_not_use_spawn_labels():
    gravity = torch.tensor([[0., 0., -1.], [1., 0., 0.], [-1., 0., 0.],
                            [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [.58, .58, .58]])
    assert classify_gravity(gravity).tolist() == list(range(7))


@pytest.mark.parametrize("gravity", [[1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.]])
def test_side_and_inverted_states_have_trainable_six_axis_authority(gravity):
    parent = nn.Linear(35, 6)
    actor = UnifiedGetupActor(parent)
    obs = torch.zeros(1, 35)
    obs[0, 7:10] = torch.tensor(gravity)
    assert recovery_gate(obs).item() == 1.
    actor(obs).sum().backward()
    assert actor.recovery[-1].bias.grad.abs().min() > 0
    assert all(p.grad is None for p in actor.parent.parameters())


def test_upright_straight_and_spin_retain_clipped_parent():
    parent = nn.Linear(35, 6)
    actor = UnifiedGetupActor(parent)
    obs = torch.zeros(4, 35)
    obs[:, 9] = -1.
    obs[:, 0] = torch.tensor([0., 1., -1., 0.])
    obs[:, 2] = torch.tensor([0., 0., 0., 1.])
    expected = torch.minimum(torch.maximum(parent(obs), -actor.motion_bound), actor.motion_bound)
    torch.testing.assert_close(actor(obs), expected, rtol=0, atol=0)


def test_arc_head_reads_both_commands_without_winner_selection():
    actor = UnifiedGetupActor(nn.Linear(35, 6))
    obs = torch.zeros(4, 35)
    obs[:, 9] = -1.
    obs[:, 0] = torch.tensor([1., 1., -1., -1.])
    obs[:, 2] = torch.tensor([1., -1., 1., -1.])
    actor(obs).sum().backward()
    assert actor.arc[-1].weight.grad.abs().sum() > 0


def test_no_privileged_recovery_phase_input_or_script_switch():
    cfg = settings()
    assert cfg["fallback_script"] is False and cfg["actor_dim"] == 35
    assert cfg["motor_control"] == "six_axis_RL_after_zero_effort_settling"
    with pytest.raises(ValueError):
        settings(fallback_script=True)
    with pytest.raises(ValueError):
        settings(passive_max_seconds=20.)


def test_push_does_not_apply_during_passive_or_after_termination():
    elapsed = torch.tensor([4.9, 5., 5.1, 5.1, 5.1, 5.2])
    selected = torch.ones(6, dtype=torch.bool)
    passive = torch.tensor([False, False, True, False, False, False])
    finished = torch.tensor([False, False, False, True, False, False])
    assert push_window(elapsed, selected, passive, finished, settings()).tolist() == [False, True, False, False, True, False]
