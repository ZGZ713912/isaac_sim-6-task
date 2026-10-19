"""Changing privileged inputs must preserve the actor and learned value function."""
from copy import deepcopy

import pytest
import torch

from wheeled_algo.critic_input_extension import extend_flat_critic


def checkpoint():
    generator = torch.Generator().manual_seed(87)
    w = torch.randn(256, 81, generator=generator, dtype=torch.float64)
    return dict(actor_state_dict={"weights": torch.randn(7, generator=generator)},
        critic_state_dict={"mlp.0.weight": w, "mlp.0.bias": torch.randn(256, generator=generator, dtype=torch.float64)},
        optimizer_state_dict=dict(param_groups=[dict(name="actor", params=[0]), dict(name="critic", params=[1, 2])],
            state={0:dict(step=torch.tensor(216.), exp_avg=torch.ones(7), exp_avg_sq=torch.ones(7)),
                   1:dict(step=torch.tensor(256.), exp_avg=w.clone(), exp_avg_sq=w.square()),
                   2:dict(step=torch.tensor(256.), exp_avg=torch.ones(256), exp_avg_sq=torch.ones(256))}))


def test_zero_extension_preserves_actor_adam_old_columns_and_value_function():
    source = checkpoint()
    migrated = extend_flat_critic(source)
    assert source["critic_state_dict"]["mlp.0.weight"].shape == (256, 81)
    actual = migrated["critic_state_dict"]["mlp.0.weight"]
    assert torch.equal(actual[:, :81], source["critic_state_dict"]["mlp.0.weight"])
    assert torch.count_nonzero(actual[:, 81:]) == 0
    for key, value in source["actor_state_dict"].items():
        assert torch.equal(migrated["actor_state_dict"][key], value)
    for index in (0, 2):
        assert all(torch.equal(v, migrated["optimizer_state_dict"]["state"][index][k])
                   for k, v in source["optimizer_state_dict"]["state"][index].items())
    for name in ("exp_avg", "exp_avg_sq"):
        new = migrated["optimizer_state_dict"]["state"][1][name]
        assert torch.equal(new[:, :81], source["optimizer_state_dict"]["state"][1][name])
        assert torch.count_nonzero(new[:, 81:]) == 0
    assert migrated["optimizer_state_dict"]["state"][1]["step"] == 256
    inputs = torch.randn(12, 81, dtype=torch.float64)
    extra = torch.randn(12, 2, dtype=torch.float64) * 100
    old = torch.nn.functional.linear(inputs, source["critic_state_dict"]["mlp.0.weight"], source["critic_state_dict"]["mlp.0.bias"])
    new = torch.nn.functional.linear(torch.cat((inputs, extra), -1), actual, migrated["critic_state_dict"]["mlp.0.bias"])
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fault", ("width", "moments"))
def test_an_unrecognized_value_or_optimizer_layout_is_rejected(fault):
    source = checkpoint()
    if fault == "width":
        source["critic_state_dict"]["mlp.0.weight"] = torch.zeros(256, 82)
    else:
        source["optimizer_state_dict"]["state"][1]["exp_avg"] = torch.zeros(256, 80)
    with pytest.raises(ValueError):
        extend_flat_critic(source)
