from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.evaluation_interventions import neutralize_evaluation_bias


@pytest.mark.parametrize("component", [None, "policy_encoder", "policy_gyro", "pd_encoder"])
def test_bias_counterfactual_changes_only_declared_reset_slice_and_no_rng(component):
    generator = torch.Generator().manual_seed(213)
    policy = torch.randn(12, 18, generator=generator)
    pd = torch.randn(12, 6, generator=generator)
    noise = torch.randn(12, 18, generator=generator)
    history = torch.randn(3, 12, 18, generator=generator)
    env = SimpleNamespace(domain=SimpleNamespace(bias=policy.clone(), sigma=noise.clone(),
                          sensor_history=history.clone(), rng=generator),
                          getup_feedback=SimpleNamespace(bias=pd.clone()))
    rng, global_rng = generator.get_state().clone(), torch.get_rng_state().clone()
    result = neutralize_evaluation_bias(env, component)
    expected_policy, expected_pd = policy.clone(), pd.clone()
    if component == "pd_encoder":
        expected_pd.zero_()
        expected_max = pd.abs().max()
    elif component == "policy_encoder":
        expected_policy[:, 6:10] = 0.
        expected_max = policy[:, 6:10].abs().max()
    elif component == "policy_gyro":
        expected_policy[:, :3] = 0.
        expected_max = policy[:, :3].abs().max()
    assert torch.equal(env.domain.bias, expected_policy)
    assert torch.equal(env.getup_feedback.bias, expected_pd)
    assert torch.equal(env.domain.sigma, noise)
    assert torch.equal(env.domain.sensor_history, history)
    assert torch.equal(generator.get_state(), rng)
    assert torch.equal(torch.get_rng_state(), global_rng)
    if component is None:
        assert result is None
    else:
        assert result["component"] == component
        assert result["before_max_abs"] == float(expected_max)
        assert result["after_max_abs"] == 0.


def test_invalid_intervention_rejects_before_accessing_or_mutating_environment():
    with pytest.raises(ValueError, match="Unknown evaluation bias"):
        neutralize_evaluation_bias(object(), "all")
