import math

import pytest
import torch
from torch import nn

from wheeled_algo.gap_entry_handoff import HandoffGapGetupActor
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.upright_target_limit import UprightTargetLimitedGapActor


def pair():
    torch.manual_seed(27)
    old = HandoffGapGetupActor(nn.Linear(35, 6), upright_entry_recovery=True,
        canonical_recovery=True, smooth_knee_targets=True, cyclic_leg_targets=True,
        recovery_leg_step=.3)
    new = UprightTargetLimitedGapActor(nn.Linear(35, 6), upright_entry_recovery=True,
        canonical_recovery=True, smooth_knee_targets=True, cyclic_leg_targets=True,
        recovery_leg_step=.3)
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


@pytest.mark.parametrize("gravity_z", [-1., -.85, -.7])
def test_final_position_error_remains_bounded_across_every_entry_phase(gravity_z):
    old, new = pair()
    obs = torch.randn(256, 35)*.1
    obs[:, 9] = gravity_z
    obs[:, 10:14] = (torch.rand(256, 4)*2.-1.)*math.pi
    obs[:, 22:26] = obs[:, 10:14]/.25
    before, after = old(obs), new(obs)
    assert (wrap(after[:, :4]*.25-obs[:, 10:14]).abs() <= .120002).all()
    assert torch.equal(before[:, 4:], after[:, 4:])
    after.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_large_handoff_jump_is_bounded_but_small_correction_is_exact(monkeypatch):
    _, new = pair()
    obs = torch.zeros(3, 35)
    obs[:, 9] = -1.
    obs[2, 10:14] = math.pi-.02
    target = torch.zeros(3, 6)
    target[0, :4] = torch.tensor([.025, -.04, .06, -.06])/.25
    target[1, :4] = torch.tensor([.8, -.7, .6, -.5])/.25
    target[2, :4] = (-math.pi+.08)/.25
    target[:, 4:] = torch.tensor([2., -3.])
    monkeypatch.setattr(HandoffGapGetupActor, "forward", lambda self, x: target)
    actual = new(obs)
    assert torch.equal(actual[0], target[0])
    assert wrap(actual[1:, :4]*.25-obs[1:, 10:14]).abs().max() <= .120002
    assert torch.equal(actual[:, 4:], target[:, 4:])


@pytest.mark.parametrize("gravity_z", [1., 0., -.5])
def test_side_and_inverted_getup_outputs_are_exactly_preserved(gravity_z):
    old, new = pair()
    obs = torch.randn(16, 35)*.2
    obs[:, 9] = gravity_z
    assert torch.equal(old(obs), new(obs))


def test_resume_preserves_the_final_target_function_in_task_metadata(tmp_path):
    from types import SimpleNamespace
    from train_robust_getup import prepare
    parent = tmp_path/"parent.pt"
    torch.save({"infos": {"settings": {"gap_entry_handoff_version":"orbit-v2",
        "gap_upright_target_limit_version":"error-v1", "gap_robustness_version":"gap-v1",
        "gap_severity":.6, "gap_actuator_timing_profile":"native",
        "native_origin_mode":"isolated_overlap_v1"}}},parent)
    bank = tmp_path/"bank.json"
    bank.write_text("{}")
    args = SimpleNamespace(output=tmp_path/"run",bank=bank,resume=parent,no_domains=False,
        curriculum_updates=128,quiet_leg_alpha=.25,quiet_wheel_alpha=.65,domain_scale=.35,
        gap_robustness=False,num_envs=32,rollout_steps=96,mode="train",friction="high",
        friction_min=.3,seed=123)
    _,task = prepare(args)
    assert task["gap_upright_target_limit_version"]=="error-v1"
