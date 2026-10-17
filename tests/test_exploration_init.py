"""Real native Gaussian/Adam checks without Isaac or a training job."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import GaussianDistribution

from wheeled_algo.exploration_init import initialize_policy_std, install_axis_std_logging, validate_policy_std

STD = [.05] * 4 + [.5] * 2


def algorithm():
    actor = torch.nn.Module()
    actor.mean = torch.nn.Linear(35, 6)
    actor.distribution = GaussianDistribution(6, init_std=.2)
    return SimpleNamespace(actor=actor, optimizer=torch.optim.Adam(actor.parameters(), lr=1e-4))


@pytest.mark.parametrize("values", [[], [.2] * 5, [.2] * 7, [.2] * 5 + [True],
    [.2] * 5 + [0.], [.2] * 5 + [-.1], [.2] * 5 + [float("nan")],
    [.2] * 5 + [float("inf")], [.2] * 5 + [".2"]])
def test_invalid_axis_contract_rejected(values):
    with pytest.raises(ValueError, match="six finite positive"):
        validate_policy_std(values)


def test_initialization_keeps_network_rng_optimizer_identity_and_native_sampling():
    alg = algorithm()
    parameter = alg.actor.distribution.std_param
    mean_before = deepcopy(alg.actor.mean.state_dict())
    rng_before = torch.get_rng_state().clone()
    initialize_policy_std(alg, STD)
    assert alg.actor.distribution.std_param is parameter
    assert any(p is parameter for g in alg.optimizer.param_groups for p in g["params"])
    assert not alg.optimizer.state
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(v, alg.actor.mean.state_dict()[k]) for k, v in mean_before.items())
    dist = alg.actor.distribution
    mean = alg.actor.mean(torch.zeros(16, 35))
    dist.update(mean)
    sample = dist.sample()
    expected = torch.distributions.Normal(mean, torch.tensor(STD))
    torch.testing.assert_close(dist.log_prob(sample), expected.log_prob(sample).sum(-1))
    assert bool((sample != mean).any())
    loss = -dist.log_prob(mean.detach() + .1).mean()
    loss.backward()
    assert torch.isfinite(parameter.grad).all() and bool((parameter.grad != 0).all())
    before = parameter.detach().clone()
    alg.optimizer.step()
    assert not torch.equal(parameter, before)
    with pytest.raises(ValueError, match="unused optimizer"):
        initialize_policy_std(alg, STD)


def test_exact_checkpoint_load_overrides_fresh_std_and_restores_adam():
    parent = algorithm()
    initialize_policy_std(parent, STD)
    dist = parent.actor.distribution
    dist.update(parent.actor.mean(torch.zeros(4, 35)))
    (-dist.log_prob(torch.ones(4, 6) * .1).mean()).backward()
    parent.optimizer.step()
    actor_state, optimizer_state = deepcopy(parent.actor.state_dict()), deepcopy(parent.optimizer.state_dict())
    child = algorithm()
    initialize_policy_std(child, STD)
    child.actor.load_state_dict(actor_state)
    child.optimizer.load_state_dict(optimizer_state)
    assert all(torch.equal(v, child.actor.state_dict()[k]) for k, v in actor_state.items())
    assert torch.equal(child.actor.distribution.std_param, parent.actor.distribution.std_param)
    assert not torch.equal(child.actor.distribution.std_param, torch.tensor(STD))
    for key, state in optimizer_state["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(child.optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0)


@pytest.mark.parametrize("writer_enabled", [False, True])
def test_axis_logging_uses_native_writer_step_and_policy_order(writer_enabled):
    events = []
    writer = SimpleNamespace(add_scalar=lambda *args: events.append(args)) if writer_enabled else None
    native_calls = []
    logger = SimpleNamespace(writer=writer, log=lambda **kwargs: native_calls.append(kwargs) or "native")
    install_axis_std_logging(logger)
    kwargs = dict(it=128, action_std=torch.tensor([[1., 2., 3., 4., 5., 6.]]))
    assert logger.log(**kwargs) == "native"
    assert native_calls == [kwargs]
    assert len(events) == (6 if writer_enabled else 0)
    if writer_enabled:
        assert events[0] == ("Policy/std/left_leg1", 1., 128)
        assert events[4] == ("Policy/std/left_wheel", 5., 128)
