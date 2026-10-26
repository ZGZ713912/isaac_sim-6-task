"""Candidate-only CPU tests; native production files are still frozen."""
from copy import deepcopy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
import torch
from torch import nn
from entry_optimizer_candidate001 import append_entry_optimizer_state, freeze_except_entry, prepare_entry_optimizer_parameters
from recovery_support_candidate001 import recovery_support_cost


def test_support_cost_scope_and_units():
    clearance = torch.tensor([[.45, .45], [.45, .45], [.45, .45], [0., -.001], [.45, .45]])
    gz = torch.tensor([-1., 1., -.7, -1., -1.])
    active = torch.tensor([True, True, True, True, False])
    assert torch.equal(recovery_support_cost(clearance, gz, active), torch.zeros(5))
    actual = recovery_support_cost(clearance, gz, active, 1.)
    assert torch.allclose(actual, torch.tensor([3., 0., 0., 0., 0.]))
    assert torch.equal(actual, recovery_support_cost(clearance.flip(-1), gz, active, 1.))


def test_support_cost_monotonic_clearance_and_upright_blend():
    clearance = torch.tensor([[0., 0.], [.15, .15], [.30, .30], [.45, .45]])
    result = recovery_support_cost(clearance, torch.full((4,), -1.), torch.ones(4, dtype=torch.bool), 1.)
    assert torch.allclose(result, torch.tensor([0., 1., 2., 3.]))
    middle = recovery_support_cost(torch.tensor([[.15, .15]]), torch.tensor([-.8]), torch.tensor([True]), 1.)
    assert torch.allclose(middle, torch.tensor([.5]))


@pytest.mark.parametrize("weight", [-1., 2.01, float("nan"), float("inf")])
def test_support_rejects_invalid_weight(weight):
    with pytest.raises(ValueError):
        recovery_support_cost(torch.zeros(1, 2), torch.zeros(1), torch.ones(1, dtype=torch.bool), weight)


def test_optimizer_slot_migration_native_roundtrip_and_frozen_step():
    torch.manual_seed(42)
    old_actor, critic = nn.Linear(3, 2), nn.Linear(3, 1)
    optimizer = torch.optim.Adam([dict(params=old_actor.parameters(), name="actor", lr=1e-3),
                                  dict(params=critic.parameters(), name="critic", lr=2e-3)])
    (old_actor(torch.ones(4, 3)).sum() + critic(torch.ones(4, 3)).sum()).backward()
    optimizer.step()
    before = deepcopy(optimizer.state_dict())
    migrated = append_entry_optimizer_state(before, 2)
    entry = nn.Linear(3, 2)
    native = torch.optim.Adam([dict(params=[*old_actor.parameters(), *entry.parameters()], name="actor", lr=1e-3),
                               dict(params=critic.parameters(), name="critic", lr=2e-3)])
    native.load_state_dict(migrated)
    actual = native.state_dict()
    assert actual["param_groups"] == migrated["param_groups"]
    assert actual["state"].keys() == migrated["state"].keys()
    for key, value in before["state"].items():
        mapped = key if key < 2 else key + 2
        for name, tensor in value.items():
            assert torch.equal(tensor, actual["state"][mapped][name])
    for key, value in migrated["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, actual["state"][key][name])
    old_weights = deepcopy(old_actor.state_dict())
    old_actor.requires_grad_(False)
    native.zero_grad(set_to_none=True)
    entry(torch.ones(4, 3)).sum().backward()
    native.step()
    assert all(torch.equal(v, old_actor.state_dict()[k]) for k, v in old_weights.items())
    assert len(native.state[entry.weight]) == 3
    assert before["param_groups"][0]["params"] == [0, 1]


def test_entry_gradient_scope_and_default():
    distribution = nn.Module()
    distribution.mean = nn.Module()
    distribution.mean.recovery = nn.Linear(3, 2)
    distribution.mean.entry_recovery = nn.Linear(3, 2).requires_grad_(False)
    distribution.log_std = nn.Parameter(torch.ones(2))
    mean = distribution.mean
    prepare_entry_optimizer_parameters(mean, {})
    assert not mean.entry_recovery.weight.requires_grad
    prepare_entry_optimizer_parameters(mean, dict(gap_entry_optimizer_layout="append-v1", gap_entry_recovery_split_version="frozen-v1"))
    assert mean.entry_recovery.weight.requires_grad
    freeze_except_entry(distribution)
    assert all(p.requires_grad == name.startswith("mean.entry_recovery.") for name, p in distribution.named_parameters())
    with pytest.raises(ValueError):
        prepare_entry_optimizer_parameters(mean, dict(gap_entry_optimizer_layout="bad"))


@pytest.mark.parametrize("count", [0, -1, True, .5])
def test_optimizer_layout_rejects_bad_count(count):
    with pytest.raises(ValueError):
        append_entry_optimizer_state({}, count)
