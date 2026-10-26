"""A larger fallen knee request must preserve entry, motion and hard limits."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from wheeled_algo.gap_entry_handoff import HandoffGapGetupActor
from wheeled_algo.paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap
from wheeled_algo.robust_getup import RobustGetupActor


def model(step=.15, actor=RobustGetupActor):
    extra = dict(upright_entry_recovery=True) if actor is HandoffGapGetupActor else {}
    return actor(nn.Linear(35, 6), smooth_knee_targets=True, recovery_leg_step=.3,
                 recovery_knee_step=step, **extra)


def observation(count=32):
    obs = torch.zeros(count, 35)
    obs[:, 3] = .305 * 5.
    obs[:, 9] = 1.
    delta = obs.new_tensor([b[1] for b in DELTA_BOUNDS])
    difference = obs.new_tensor([1., -1.]) * delta - obs.new_tensor(NOMINAL_DELTA)
    obs[:, 10:14] = torch.stack((-.5 * difference, .5 * difference), -1).flatten()
    obs[:, 22:26] = obs[:, 10:14] / .25
    return obs


def requested_delta(action):
    pair = (action[:, :4] * .25).unflatten(-1, (2, 2))
    return action.new_tensor([1., -1.]) * (action.new_tensor(NOMINAL_DELTA)
        + wrap(pair[..., 1] - pair[..., 0]))


@pytest.mark.parametrize("bad", [0., .049, .451, float("nan"), float("inf"), True])
def test_invalid_knee_request_cannot_create_an_actor(bad):
    with pytest.raises(ValueError, match="Knee request"):
        model(bad)


def test_larger_knee_lead_folds_more_without_changing_common_rotation_or_wheels():
    torch.manual_seed(671)
    old = model()
    with torch.no_grad():
        for p in old.recovery.parameters():
            p.zero_()
    new = deepcopy(old)
    new.recovery_knee_step = .3
    obs = observation()
    a, b = old(obs), new(obs)
    assert (requested_delta(b) < requested_delta(a)).all()
    for result in (a, b):
        delta = requested_delta(result)
        bounds = result.new_tensor(DELTA_BOUNDS)
        assert ((delta >= bounds[:, 0]) & (delta <= bounds[:, 1])).all()
    torch.testing.assert_close(a[:, :4].unflatten(-1, (2, 2)).mean(-1),
                               b[:, :4].unflatten(-1, (2, 2)).mean(-1), rtol=0, atol=1e-7)
    assert torch.equal(a[:, 4:], b[:, 4:])
    b.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_upright_orbit_entry_keeps_its_original_knee_request():
    torch.manual_seed(672)
    old = model(actor=HandoffGapGetupActor)
    new = deepcopy(old)
    new.recovery_knee_step = .3
    obs = observation()
    obs[:, 9] = -1.
    obs[:, 10:14] += obs.new_tensor([1., 1., -1., -1.])
    obs[:, 22:26] = obs[:, 10:14] / .25
    assert torch.equal(old(obs), new(obs))
    obs[:, 0] = .5 * 2.
    obs[:, 10:14] = obs[:, 22:26] = 0.
    assert torch.equal(old(obs), new(obs))


def test_resume_retains_explicit_knee_request_in_task_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_robust_getup import prepare
    parent = tmp_path / "parent.pt"
    torch.save({"infos": {"settings": {"recovery_knee_step": .3}}}, parent)
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    args = SimpleNamespace(output=tmp_path / "run", bank=bank, resume=parent, no_domains=False,
        curriculum_updates=128, quiet_leg_alpha=.25, quiet_wheel_alpha=.65, domain_scale=.35,
        gap_robustness=False, num_envs=32, rollout_steps=96, mode="train", friction="high",
        friction_min=.3, seed=123)
    _, task = prepare(args)
    assert task["recovery_knee_step"] == .3
