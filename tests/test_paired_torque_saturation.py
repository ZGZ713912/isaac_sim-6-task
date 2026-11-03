"""Leg differential authority, old-contract isolation and physical cap checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train_prone_motion import configuration
from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation
from wheeled_tasks.chassis.paired_torque_saturation import CONTRACT, allocate, install, validate
from wheeled_tasks.chassis.prone_motion import validate as validate_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_shape_priority_v1.json").read_text())
LIMITS = dict(minimum=-.14, maximum=1.43, margin=.03)


def encoder(relative=.5):
    return torch.tensor([[0., relative, 0., -relative]], dtype=torch.float64)


def test_independent_clipping_destroys_differential_but_shape_priority_retains_it():
    requested = torch.tensor([[-200., -100., 150., 300.]], dtype=torch.float64)
    assert torch.equal(requested.clamp(-40., 40.), requested.new_tensor([[-40., -40., 40., 40.]]))
    limited, projected = allocate(requested, encoder(), LIMITS)
    assert torch.equal(limited, requested.new_tensor([[-40., 40., -40., 40.]]))
    assert torch.equal(limited, projected)


def test_subcap_total_pd_plus_spring_is_bitwise_unchanged():
    requested = torch.tensor([[0., 40., -39., -10.]], dtype=torch.float64)
    limited, projected = allocate(requested, encoder(), LIMITS)
    assert torch.equal(limited, requested) and torch.equal(projected, requested)
    request = torch.randn((1000, 4), generator=torch.Generator().manual_seed(108)).clamp(-1., 1.)*39.
    assert torch.equal(allocate(request, encoder().float().repeat(1000, 1), LIMITS)[0], request)


@pytest.mark.parametrize("relative,torque,expected", [
    (1.42, [-20., 40., 20., -40.], [10., 10., -10., -10.]),
    (-.13, [40., -20., -40., 20.], [10., 10., -10., -10.]),
    (1.42, [40., -20., -40., 20.], [40., -20., -40., 20.]),
    (-.13, [-20., 40., 20., -40.], [-20., 40., 20., -40.])])
def test_existing_outward_projection_keeps_common_torque_and_inward_authority(relative, torque, expected):
    request = torch.tensor([torque], dtype=torch.float64)
    limited, projected = allocate(request, encoder(relative), LIMITS)
    assert torch.equal(limited, request)
    assert torch.equal(projected, request.new_tensor([expected]))
    assert torch.equal(projected.reshape(1, 2, 2).sum(-1), request.reshape(1, 2, 2).sum(-1))


def test_many_requests_maximize_shape_authority_within_caps_and_do_not_couple_legs():
    rng = torch.Generator().manual_seed(107)
    request = torch.randn((10000, 4), generator=rng, dtype=torch.float64)*1000.
    limited, projected = allocate(request, encoder().repeat(10000, 1), LIMITS)
    assert limited.abs().max() <= 40.+1e-12 and projected.abs().max() <= 40.+1e-12
    pair = limited.reshape(-1, 2, 2); source = request.reshape(-1, 2, 2)
    torch.testing.assert_close(pair[..., 0]-pair[..., 1], (source[..., 0]-source[..., 1]).clamp(-80., 80.))
    old = source.clamp(-40., 40.)
    assert ((source[..., 0]-source[..., 1]-(pair[..., 0]-pair[..., 1])).abs()
        <= (source[..., 0]-source[..., 1]-(old[..., 0]-old[..., 1])).abs()+1e-10).all()
    changed = request.clone();changed[:, 2:] *= 3.
    assert torch.equal(allocate(changed, encoder().repeat(10000, 1), LIMITS)[0][:, :2], limited[:, :2])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_request_is_not_hidden(value):
    request = torch.tensor([[value, 100., 50., 25.]])
    limited, projected = allocate(request, encoder().float(), LIMITS)
    assert not torch.isfinite(limited).all() and not torch.isfinite(projected).all()


def test_explicit_plan_keeps_absolute_targets_reward_plant_and_record_only_closure():
    old = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
    restored = deepcopy(PLAN);assert restored.pop("paired_torque_saturation") == CONTRACT
    assert restored == old
    assert validate_plan(PLAN) == PLAN and validate_plan(old) == old
    for field, value in [("total_motor_limit_nm", 50.), ("target_semantics", "relative"), ("mode", "clip")]:
        invalid = deepcopy(CONTRACT);invalid[field] = value
        with pytest.raises(ValueError, match="explicit paired"):
            validate(invalid)


def test_installed_adapter_sums_unclipped_pd_and_holding_once_and_keeps_old_model(tmp_path):
    cfg, _, _ = configuration(tmp_path, PLAN)
    model = GasSpringCompensation(cfg["v5_control"]["gas_spring_compensation"], root=ROOT,
        device="cpu", dtype=torch.float64)
    old_function = model.motor_efforts
    calls = []
    def counted(pd, q):
        calls.append((pd.clone(), q.clone()))
        return old_function(pd, q)
    model.motor_efforts = counted
    env = SimpleNamespace(plan=PLAN, spring_feedforward=model,
        startup_report={"encoder_spring_feedforward": {}})
    install(env, CONTRACT)
    q = encoder(.7)
    holding = model.evaluate(q)[0]
    desired = q.new_tensor([[-200., -100., 150., 300.]])
    actual = model.motor_efforts(desired-holding, q)
    assert len(calls) == 1
    torch.testing.assert_close(model.last["requested_effort_nm"], desired, atol=1e-12, rtol=0.)
    torch.testing.assert_close(actual, q.new_tensor([[-40., 40., -40., 40.]]), atol=1e-12, rtol=0.)
    assert env.startup_report["paired_torque_saturation"] == CONTRACT
    # A separately constructed legacy model still executes the authenticated clamp.
    legacy = GasSpringCompensation(cfg["v5_control"]["gas_spring_compensation"], root=ROOT,
        device="cpu", dtype=torch.float64)
    torch.testing.assert_close(legacy.motor_efforts(desired-holding, q), desired.clamp(-40., 40.))
    with pytest.raises(RuntimeError, match="already"):
        install(env, CONTRACT)
