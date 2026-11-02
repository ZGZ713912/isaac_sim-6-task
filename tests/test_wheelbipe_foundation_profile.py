from copy import deepcopy
from types import SimpleNamespace as S

import pytest
import torch

from wheeled_tasks.wheelbipe_port.training_profile import (
    apply_phase, behavior_gate, configure_profile, electrical_reward_terms,
    phase_for_update, specification,
)


def fake_cfg():
    return S(commands=S(ranges=S(),special_modes={k:S(iteration_start=0) for k in ("dash","spin_low","spin_mid")}),
        rewards={"track_lin_vel_xy":1.,"joint_torque":-.0001},events=S(reset_base=S(params={})))


def test_profile_keeps_physical_model_and_legacy_objective_separate():
    cfg = fake_cfg()
    original = deepcopy(cfg.rewards)
    configure_profile(cfg,"legacy")
    assert cfg.rewards==original and cfg.port_gas_feedforward
    configure_profile(cfg,"foundation-v1")
    assert not cfg.port_gas_feedforward and cfg.port_reward_motor_only
    assert cfg.commands.ranges.lin_vel_x==(-.5,.5)
    assert cfg.commands.rel_standing_envs==.5
    assert cfg.commands.heading_command is False
    assert cfg.rewards["joint_torque"]==original["joint_torque"]
    assert cfg.commands.special_modes["dash"].iteration_start==12000


@pytest.mark.parametrize("update,start",[(0,0),(1999,0),(2000,2000),(5999,2000),(6000,6000),(11999,6000),(12000,12000),(20000,12000)])
def test_course_uses_completed_update_clock_and_restores_same_phase(update,start):
    cfg = fake_cfg()
    configure_profile(cfg,"foundation-v1")
    assert apply_phase(cfg,update)["start_update"]==start
    assert phase_for_update(update)["start_update"]==start
    assert cfg.commands.ranges.lin_vel_x[1]==phase_for_update(update)["vx_max_m_s"]


def test_new_objective_distinguishes_drift_and_wrong_direction():
    spec = specification()
    # Actual old009 mid-standing and forward diagnostic speeds.
    def velocity_payment(v,command):
        err=torch.tensor(command-v)
        return 3.*torch.exp(-err.square()/.09)-err.square()-(abs(v) if command==0. else 0.)
    assert velocity_payment(0.,0.)-velocity_payment(-.191952914,0.)>1.2
    assert velocity_payment(.5,.5)-velocity_payment(-.144421414,.5)>3.
    assert set(spec["required_behavior_reviews"]).issubset(spec["review_updates"])
    assert spec["review_gate"]["closed_chain_distance_gate"] is None


def test_electrical_cost_excludes_dissipative_and_spring_effort_and_braking_power():
    motor=torch.tensor([[1.,2.,3.,4.,5.,-6.]])
    velocity=torch.tensor([[1.,1.,-2.,1.,1.,-3.]])
    old=torch.zeros_like(motor)
    terms=electrical_reward_terms(motor,velocity,old,old)
    assert terms["joint_torque"].item()==91.
    assert terms["wheel_power"].item()==18.  # left wheel is braking
    assert terms["torque_smoothness_leg"].item()==46.
    assert terms["torque_smoothness_wheel"].item()==45.
    assert terms["torque_rate"].item()==91.


def passing_receipt():
    return dict(status="complete",ppo_updates=0,arguments=dict(seed=111,vehicle="port",control="actor",training_profile="foundation-v1"),
        cases=[dict(case=name,envs=32,duration_s=8.,resets=0,command=[vx,0.,.305],
                    windows={"tail_half":dict(vx_m_s=dict(mean=actual,abs_mean=abs(actual)),
                        tilt_deg=dict(mean=1.),height_error_m=dict(abs_mean=.005))})
               for name,vx,actual in (("stand_mid",0.,.01),("forward",.5,.44),("reverse",-.5,-.45))])


def test_gate_pass_does_not_qualify_any_long_term_capability():
    result=behavior_gate(passing_receipt())
    assert result["passed"] and result["qualification"] is False


@pytest.mark.parametrize("failure",["wrong_direction","height_error","tilt","reset","missing","partial","wrong_seed","nan","wrong_command"])
def test_gate_rejects_old_failure_modes_and_incomplete_evidence(failure):
    receipt=passing_receipt()
    row=receipt["cases"][0]
    if failure=="wrong_direction": receipt["cases"][1]["windows"]["tail_half"]["vx_m_s"]["mean"]=-.14
    if failure=="height_error": row["windows"]["tail_half"]["height_error_m"]["abs_mean"]=.04
    if failure=="tilt": row["windows"]["tail_half"]["tilt_deg"]["mean"]=6.
    if failure=="reset": row["resets"]=1
    if failure=="missing": receipt["cases"].pop()
    if failure=="partial": receipt["status"]="sampling"
    if failure=="wrong_seed": receipt["arguments"]["seed"]=112
    if failure=="nan": row["windows"]["tail_half"]["vx_m_s"]["abs_mean"]=float("nan")
    if failure=="wrong_command": row["command"][0]=.1
    assert not behavior_gate(receipt)["passed"]


def test_user_accepted_drift_is_reported_without_new_rejection_cutoff():
    receipt=passing_receipt()
    receipt["cases"][0]["windows"]["tail_half"]["vx_m_s"] = dict(mean=.192,abs_mean=.192)
    assert behavior_gate(receipt)["passed"]
    assert specification()["reward_overrides"]["stand_still_lin_vel"]==-1.
