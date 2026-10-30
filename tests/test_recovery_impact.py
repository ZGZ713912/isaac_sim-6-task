"""Physical-time impact history, part caps and passive/active isolation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.recovery_impact import RecoveryImpact, GROUPS, validate

CONFIG = json.loads((Path(__file__).resolve().parents[1]/"contracts/wheel_leg_prone_motion_v5.json").read_text())["recovery_impact"]
NAMES = ["base_link","M_link1","L_link1","L_link2","L_link3"]


def monitor(config=None, dt=.001):
    return RecoveryImpact(2,"cpu",NAMES,config or CONFIG,dt,torch.tensor([20.,20.]))


def force(n,body=0):
    a = torch.zeros(2,len(NAMES),3); a[:,body,2] = n
    return a


def test_one_millisecond_peak_survives_policy_boundary_and_window_spans_it():
    m = monitor(); m.begin_interval()
    for _ in range(10): m.observe(force(100.),torch.tensor([True,True]))
    m.observe(force(2000.),torch.tensor([True,True]))
    assert m.force_peak[0,0] == 2000.
    m.begin_interval()
    m.observe(force(0.),torch.tensor([True,True]))
    assert m.force_peak[0,0] == 0.
    assert float(m.impulse_peak[0,0]) == pytest.approx(3.,abs=1e-5)
    assert m.episode_force_peak[0,0] == 2000.


def test_impulse_accumulates_per_body_before_group_reduction():
    m = monitor()
    for _ in range(10): m.observe(force(100.,2),torch.tensor([True,True]))
    for _ in range(10): m.observe(force(100.,3),torch.tensor([True,True]))
    assert float(m.impulse_peak[0,2]) == pytest.approx(1.,abs=1e-5)


def test_fixed_physical_window_agrees_under_timestep_refinement():
    coarse,fine = monitor(),monitor(dt=.0005)
    for _ in range(20): coarse.observe(force(123.),torch.tensor([True,True]))
    for _ in range(40): fine.observe(force(123.),torch.tensor([True,True]))
    assert torch.allclose(coarse.impulse_peak,fine.impulse_peak,atol=1e-5)


def test_passive_impacts_cannot_poison_active_limits_and_selective_reset_is_local():
    c = deepcopy(CONFIG); c["impulse_limits_ns"]["chassis"] = .5
    m = monitor(c)
    for _ in range(30): m.observe(force(10000.),torch.tensor([False,True]))
    assert not m.violation[0] and m.violation[1]
    m.reset(torch.tensor([0])); m.begin_interval()
    m.observe(force(100.),torch.tensor([True,True]))
    assert not m.violation[0] and m.violation[1]
    assert float(m.impulse_peak[0,0]) == pytest.approx(.1)


@pytest.mark.parametrize("group,body",[("chassis",0),("gimbal",1),("legs",2),("wheels",4)])
def test_each_part_including_wheels_has_an_independent_force_cap(group,body):
    c = deepcopy(CONFIG); c["force_limits_n"][group] = 200.
    m = monitor(c); m.observe(force(201.,body),torch.tensor([True,False]))
    assert m.violation.tolist() == [True,False]


def test_unmeasured_limits_are_not_invented_and_invalid_limits_fail_closed():
    assert all(v is None for v in CONFIG["force_limits_n"].values())
    c = deepcopy(CONFIG); c["force_limits_n"]["gimbal"] = float("nan")
    with pytest.raises(ValueError): validate(c)
    c = deepcopy(CONFIG); c["window_seconds"] = .0015
    with pytest.raises(ValueError): validate(c)
