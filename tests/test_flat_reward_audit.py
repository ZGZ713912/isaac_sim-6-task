"""Payment denominators must exclude later episodes and retain terminal costs."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flat_reward_audit import FlatRewardAudit


def test_first_episode_and_post_warmup_payments_have_exact_denominators():
    audit = FlatRewardAudit(["a", "a", "b"], "cpu", .02, 1.)
    components = dict(progress=torch.tensor([1., 2., 3.]), effort=torch.tensor([-.5, -1., -1.5]))
    data = dict(reward=torch.tensor([.01, .02, .03]), episode_ticks=torch.tensor([1, 51, 51]),
                physical_phase=torch.tensor([0, 2, 4]))
    before = deepcopy((components, data)); rng = torch.get_rng_state().clone()
    audit.observe(components, data, torch.tensor([True, True, True]))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[0][k], v) for k, v in components.items())
    assert all(torch.equal(before[1][k], v) for k, v in data.items())
    audit.observe(dict(progress=torch.tensor([float("nan"), 20., 30.]),
        effort=torch.tensor([float("nan"), -10., -15.])),
        dict(reward=torch.tensor([float("nan"), .16, .3]), episode_ticks=torch.tensor([2, 52, 52]),
             physical_phase=torch.tensor([0, 0, 0])), torch.tensor([False, True, True]))
    whole = audit.report()["whole_episode"]["a"]
    late = audit.report()["post_warmup"]["a"]
    assert whole["frames"] == 3 and late["frames"] == 2
    assert whole["component_reward_per_policy_step"]["progress"] == pytest.approx(23 * .02 / 3)
    assert late["component_reward_per_sim_second"]["progress"] == pytest.approx(11.)
    assert whole["residual_reward_per_policy_step"] == pytest.approx(-.04 / 3, abs=1e-8)
    assert late["actual_reward_per_policy_step"] == pytest.approx(.09)
    assert late["physical_phase_fraction"]["2"] == .5


def test_reward_density_uses_policy_dt_once_at_either_physics_clock():
    audit = FlatRewardAudit(["a"], "cpu", .02, 0.)
    for _ in range(50):
        audit.observe(dict(cost=torch.tensor([-2.])),
            dict(reward=torch.tensor([-.04]), episode_ticks=torch.tensor([1]), physical_phase=torch.tensor([0])),
            torch.tensor([True]))
    row = audit.report()["whole_episode"]["a"]
    assert row["observed_sim_seconds"] == 1.
    assert row["component_reward_per_sim_second"]["cost"] == pytest.approx(-2., abs=1e-12)
    assert row["component_reward_per_policy_step"]["cost"] == pytest.approx(-.04, abs=1e-12)


def test_changed_term_set_is_rejected():
    audit = FlatRewardAudit(["a"], "cpu", .02, 0.)
    data = dict(reward=torch.tensor([0.]), episode_ticks=torch.tensor([1]), physical_phase=torch.tensor([0]))
    audit.observe(dict(a=torch.tensor([0.])), data, torch.tensor([True]))
    with pytest.raises(ValueError, match="terms changed"):
        audit.observe(dict(b=torch.tensor([0.])), data, torch.tensor([True]))
