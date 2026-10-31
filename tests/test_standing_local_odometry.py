"""Standing displacement, sensor-only capture and locomotion isolation."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.standing_local_odometry import LocalStandingOdometry, observation_contract
from wheeled_tasks.chassis import prone_motion as task

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = observation_contract()
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())


def setup(count=1, device="cpu"):
    observer = LocalStandingOdometry(count, device, .01, CONTRACT)
    nominal = observer.coordinate.geometry.nominal
    q = torch.cat((nominal, nominal.new_zeros(2)))[None].repeat(count, 1).double()
    return observer, q


def feed(observer, q, tick, *, commands=None, gravity=None, velocity=None, enabled=None, domain=None):
    n, device = len(q), q.device
    frame = torch.zeros(n, 35, device=device)
    frame[:, 3] = .305*5.
    frame[:, 9] = -1.
    frame[:, 28] = 1.
    frame[:, 10:14] = (q[:, :4]-observer.coordinate.geometry.nominal).float()
    if commands is not None:
        frame[:, :3] = torch.as_tensor(commands, device=device)
    if gravity is not None:
        frame[:, 7:10] = torch.as_tensor(gravity, device=device)
    if velocity is not None:
        frame[:, 16:22] = torch.as_tensor(velocity, device=device)*.1
    if enabled is None:
        enabled = torch.ones(n, dtype=torch.bool, device=device)
    original = frame.clone()
    frame[:, 14:16] = observer.local_wheel_angles(frame, q[:, 4:], enabled, tick)
    if domain is not None:
        frame[:, 4:22] = domain.observe(frame[:, 4:22], tick)
        original[:, 4:22] = frame[:, 4:22]
    result = observer.update(frame, observer.coordinate.geometry.nominal, enabled, tick)
    keep = [i for i in range(35) if i not in (14, 15, 31)]
    torch.testing.assert_close(original[:, keep], result[:, keep], atol=0., rtol=0., equal_nan=True)
    return result


def capture(observer, q, tick=0):
    for i in range(20):
        frame = feed(observer, q, tick+i)
    assert observer.valid.all() and frame[:, 31].eq(1.).all()
    assert not frame[:, 14:16].any()
    return tick+20


@pytest.mark.parametrize("offset", [0., 1e8, -1e8])
def test_local_double_origin_preserves_small_displacement_after_long_motion(offset):
    observer, q = setup()
    q[:, 4:] = torch.tensor([offset, -offset], dtype=torch.float64)
    tick = capture(observer, q)
    q[:, 4:] += q.new_tensor([.5, -.5])
    frame = feed(observer, q, tick, velocity=[0., 0., 0., 0., 50., -50.])
    for t in range(tick+1, tick+81):
        frame = feed(observer, q, t)
    torch.testing.assert_close(frame[:, 14:16]/20., torch.full((1, 2), .03), atol=2e-6, rtol=0.)
    assert observer.valid.all() and not observer.source_fault.any()


def test_reference_retains_sign_through_drift_and_filter_is_idempotent():
    observer, q = setup()
    tick = capture(observer, q)
    anchor = observer.anchor.clone()
    q[:, 4:] += q.new_tensor([.1, -.1])
    a = feed(observer, q, tick, velocity=[0., 0., 0., 0., 10., -10.])
    b = feed(observer, q, tick, velocity=[0., 0., 0., 0., 10., -10.])
    assert torch.equal(a, b) and (a[:, 14:16] > 0).all()
    assert torch.equal(observer.anchor, anchor)
    q[:, 4:] -= q.new_tensor([.2, -.2])
    feed(observer, q, tick+1, velocity=[0., 0., 0., 0., -20., 20.])
    for t in range(tick+2, tick+82):
        result = feed(observer, q, t)
    assert (result[:, 14:16] < 0).all() and torch.equal(observer.anchor, anchor)


@pytest.mark.parametrize("command", [[2.,0.,0.],[-2.,0.,0.],[0.,0.,10.47],[0.,0.,-10.47],
                                      [2.,0.,.6],[0.,.1,0.]])
def test_motion_and_spin_disable_features_and_discard_old_standing_reference(command):
    observer, q = setup()
    tick = capture(observer, q)
    q[:, 4:] += 10000.
    result = feed(observer, q, tick, commands=command)
    assert not result[:, [14, 15, 31]].any() and not observer.reference_active.any()
    assert not observer.source_fault.any()
    capture(observer, q, tick+1)


def test_bad_wheel_count_latches_observation_fault_without_reanchoring():
    observer, q = setup()
    tick = capture(observer, q)
    anchor = observer.anchor.clone()
    # One missed motor revolution at the actual 15.62 reduction.
    q[:, 4] += math.tau/15.62
    result = feed(observer, q, tick)
    assert observer.source_fault.all() and not result[:, [14, 15, 31]].any()
    for t in range(tick+1, tick+30):
        feed(observer, q, t)
    assert observer.source_fault.all() and torch.equal(observer.anchor, anchor)
    feed(observer, q, tick+30, commands=[1.,0.,0.])
    capture(observer, q, tick+31)


def test_invalid_imu_preserves_reference_but_severe_tilt_requires_new_capture():
    observer, q = setup()
    tick = capture(observer, q)
    anchor = observer.anchor.clone()
    result = feed(observer, q, tick, gravity=[float("nan"),0.,-1.])
    assert not result[:, [14,15,31]].any() and torch.equal(observer.anchor, anchor)
    assert observer.reference_active.all()
    feed(observer, q, tick+1)
    assert observer.valid.all()
    result = feed(observer, q, tick+2, gravity=[0.,0.,1.])
    assert not observer.reference_active.any() and not result[:, [14,15,31]].any()
    capture(observer, q, tick+3)


def test_disabled_and_selected_episode_resets_clear_all_reference_history():
    observer, q = setup(2)
    tick = capture(observer, q)
    observer.reset(torch.tensor([0]))
    result = feed(observer, q, tick)
    assert observer.valid.tolist() == [False, True]
    result = feed(observer, q, tick+1, enabled=torch.zeros(2,dtype=torch.bool))
    assert not result[:, [14,15,31]].any()
    assert not observer.source_active.any() and not observer.reference_active.any()


def test_folded_prone_and_fast_wheels_cannot_capture_a_standing_reference():
    observer, q = setup()
    bank = json.loads((ROOT/"reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())
    geometry = json.loads((ROOT/PLAN["prone_geometry_contract"]).read_text())
    pose = next(p for p in bank["poses"] if p["name"] == geometry["reset_pose_name"])
    row = pose["full_joint_positions"]
    folded = q.clone()
    folded[:, :4] = q.new_tensor([row[k] for k in ("L_joint1","LL_joint1","R_joint1","RR_joint1")])
    for tick in range(40):
        result = feed(observer, folded, tick)
        assert not result[:, [14,15,31]].any()
    for tick in range(40,80):
        result = feed(observer, q, tick, velocity=[0.,0.,0.,0.,33.,-33.])
        assert not result[:, [14,15,31]].any()
    capture(observer, q, 80)


def test_wheel_position_uses_the_same_delayed_sensor_frame_as_legs_and_imu():
    from wheeled_tasks.chassis.robust_getup import RobustDomain, settings
    domain = RobustDomain(1,"cpu",settings(domains_enabled=False, sensor_policy_dt=.01),13,.001)
    domain.sensor_delay.fill_(2)
    observer,q = setup()
    for tick in range(25):
        result = feed(observer,q,tick,domain=domain)
    assert observer.valid.all()
    q[:,4:] += q.new_tensor([.1,-.1])
    for tick in (25,26):
        result = feed(observer,q,tick,domain=domain,velocity=[0.,0.,0.,0.,10.,-10.])
        assert not result[:,14:16].any()
    result = feed(observer,q,27,domain=domain)
    assert (result[:,14:16] > 0).all()


def test_change_height_compensates_geometry_without_moving_the_reference():
    observer, q = setup()
    tick = capture(observer, q)
    anchor = observer.anchor.clone()
    before = observer.coordinate(q.float(), torch.tensor([[0.,0.,-1.]]))[0]
    bank = json.loads((ROOT/"reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())
    for name in ("stand_low", "stand_high", "stand"):
        row = bank["poses"][bank["diagnostic_height_starts"]["mapping"][name]]["full_joint_positions"]
        end = q.clone()
        end[:, :4] = q.new_tensor([row[k] for k in ("L_joint1","LL_joint1","R_joint1","RR_joint1")])
        # Roll the wheels by exactly the geometric change so body translation is zero.
        geom = observer.coordinate(end.float(), torch.tensor([[0.,0.,-1.]]))[0]
        end[:, 4:] += ((before-geom)/q.new_tensor([.06,-.06]))
        start = q.clone()
        for i in range(1, 101):
            current = start+(end-start)*i/100.
            feed(observer, current, tick, velocity=((end-start)/1.).float())
            tick += 1
        q = end
        for _ in range(80):
            result = feed(observer, q, tick)
            tick += 1
        assert observer.valid.all() and torch.equal(observer.anchor, anchor)
        torch.testing.assert_close(result[:, 14:16], torch.zeros(1,2), atol=3e-5, rtol=0.)


def reward(command, rising, distance, tail):
    count = len(distance)
    return task.reward_terms(torch.tensor([[0.,0.,-1.]]).repeat(count,1), torch.full((count,),.305),
        torch.zeros(count,3), torch.zeros(count,3), torch.tensor([command]).repeat(count,1),
        torch.zeros(count,6),torch.zeros(count,6),torch.zeros(count),torch.zeros(count),torch.tensor(distance),
        torch.full((count,),rising,dtype=torch.bool),torch.ones(count),position_tail=tail)


def test_only_zero_command_standing_gets_a_non_saturating_far_position_cost():
    distance = [.0,.01,.03,.05,.09,.23,.65,.81]
    old = reward([0.,0.,.305],False,distance,None)
    new = reward([0.,0.,.305],False,distance,PLAN["quiet_position_tail"])
    torch.testing.assert_close(old["quiet_position"][:5],new["quiet_position"][:5],atol=0.,rtol=0.)
    assert (torch.diff(new["quiet_position"][1:]) < 0).all()
    for k in old:
        if k != "quiet_position":
            torch.testing.assert_close(old[k],new[k],atol=0.,rtol=0.)
    for command,rising in (([2.,0.,.305],False),([-2.,0.,.305],False),([0.,10.,.305],False),
                            ([.02,0.,.305],False),([0.,0.,.305],True)):
        a,b = [reward(command,rising,distance,t) for t in (None,PLAN["quiet_position_tail"])]
        for k in a:
            torch.testing.assert_close(a[k],b[k],atol=0.,rtol=0.)


def test_new_task_preserves_old_recipe_and_rejects_silent_observation_changes():
    old = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v3.json").read_text())
    legacy = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_standard_20261031.json").read_text())
    for k in old:
        if k not in ("schema", "reference", "gas_spring_model", "gas_spring_feedforward", "wheel_model"):
            assert old[k] == legacy[k]
            if k != "evaluation":
                assert legacy[k] == PLAN[k]
    assert PLAN["wheel_model"]["source_model"] == old["wheel_model"]["model"]
    assert task.validate(PLAN) == PLAN
    for field in ("limit_m","validity_slot","geometry_table_sha256"):
        bad = deepcopy(PLAN)
        bad["standing_odometry"][field] = None
        with pytest.raises(ValueError):
            task.validate(bad)
    old["standing_odometry"] = CONTRACT
    with pytest.raises(ValueError,match="v4"):
        task.validate(old)
