"""Physical traces exercise takeoff, contact hysteresis and one-shot payment."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.jump_motion import (
    JumpMotion, validate, ballistic_push, BASE, AIRBORNE, JUMP,
    WAIT, CROUCH, PUSH, FLIGHT, LAND, COMPLETE, FAILED,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_jump_motion_v1.json").read_text())
CFG = PLAN["jump_training"]


def machine(mode=JUMP, dt=.005, count=1):
    result = JumpMotion(count, "cpu", dt, CFG, torch.full((count,), 20.))
    result.reset(torch.arange(count), torch.full((count,), mode), torch.zeros(count), torch.full((count,), .42))
    return result


def sample(jump, *, z=.305, vz=0., clear=0., force=98.1, vx=0., omega=0., nonwheel=0.):
    n = jump.count
    clearance = torch.as_tensor(clear, dtype=torch.float).expand(n, 2)
    jump.observe(height=torch.full((n,), z), velocity=torch.tensor([vx, 0., vz]).expand(n, 3),
        omega=torch.full((n, 3), omega), gravity=torch.tensor([0., 0., -1.]).expand(n, 3),
        wheel_clearance=clearance, wheel_force_z=torch.full((n, 2), force),
        nonwheel_force=torch.full((n,), nonwheel), enabled=torch.ones(n, dtype=torch.bool))


def hold(jump, seconds, **kwargs):
    for _ in range(round(seconds/jump.dt)):
        sample(jump, **kwargs)


def push(jump):
    hold(jump, .1)
    assert jump.phase.item() == CROUCH
    hold(jump, .35, z=.23)
    assert jump.phase.item() == PUSH


def flight(jump):
    push(jump)
    hold(jump, .015, z=.42, vz=1., clear=.04, force=0.)
    assert jump.phase.item() == FLIGHT


def successful_landing(jump):
    hold(jump, .1, z=.42, vz=-.2, clear=.04, force=0.)
    hold(jump, .01)
    assert jump.phase.item() == LAND
    hold(jump, .3)
    assert jump.phase.item() == COMPLETE


@pytest.mark.parametrize("dt", [.001, .005])
def test_real_flight_landing_and_contiguous_hold_are_required_once(dt):
    jump = machine(dt=dt)
    jump.begin_interval()
    flight(jump)
    assert jump.total[1:3].tolist() == [1, 1]
    successful_landing(jump)
    assert jump.event_bonus().item() == 9.
    assert jump.summary()["jump_successes"] == 1
    jump.begin_interval()
    hold(jump, .5)
    assert jump.event_bonus().item() == 0. and jump.summary()["jump_successes"] == 1


def test_aerial_birth_and_drop_cannot_count_as_active_liftoff():
    jump = machine(AIRBORNE)
    jump.begin_interval()
    successful_landing(jump)
    assert jump.summary() == dict(airborne_births=1, jump_requests=0, liftoffs=0,
        landings=1, airborne_successes=1, jump_successes=0)


@pytest.mark.parametrize("clear,vz", [([.02, 0.], 1.), ([0., 0.], 1.), ([.02, .02], -.2)])
def test_force_loss_one_wheel_and_falling_are_not_self_propelled_flight(clear, vz):
    jump = machine()
    push(jump)
    hold(jump, .6, clear=clear, vz=vz, force=0.)
    assert jump.phase.item() == FAILED and jump.summary()["liftoffs"] == 0


def test_liftoff_duration_uses_contiguous_physical_samples():
    jump = machine()
    push(jump)
    sample(jump, z=.4, vz=1., clear=.02, force=0.)
    sample(jump)
    hold(jump, .01, z=.4, vz=1., clear=.02, force=0.)
    assert jump.phase.item() == PUSH
    sample(jump, z=.4, vz=1., clear=.02, force=0.)
    assert jump.phase.item() == FLIGHT


def test_landing_success_requires_full_stable_hold_and_requested_apex():
    jump = machine()
    flight(jump)
    hold(jump, .1, clear=.02, force=0.)
    hold(jump, .01)
    hold(jump, .2)
    sample(jump, vx=.3)
    hold(jump, .2)
    assert jump.summary()["jump_successes"] == 0
    hold(jump, .1)
    assert jump.summary()["jump_successes"] == 1
    second = machine()
    second.target_apex.fill_(.55)
    flight(second)
    hold(second, .1, clear=.02, force=0.)
    hold(second, .01)
    hold(second, .4)
    assert second.summary()["jump_successes"] == 0


def test_mechanical_failure_in_success_interval_cannot_be_qualified_or_paid():
    jump = machine()
    flight(jump)
    successful_landing(jump)
    jump.reject_failed_successes(torch.tensor([True]))
    assert not jump.success_event.any() and jump.summary()["jump_successes"] == 0
    assert jump.phase.item() == FAILED


def test_no_stance_request_times_out_instead_of_collecting_standing_reward_forever():
    jump = machine()
    hold(jump, 1., force=0.)
    assert jump.phase.item() == FAILED and jump.total[1] == 0


def test_base_lanes_and_partial_reset_preserve_other_episode_and_global_counters():
    jump = machine(count=3)
    jump.mode[0] = BASE
    hold(jump, .1)
    assert jump.phase.tolist() == [WAIT, CROUCH, CROUCH]
    previous = jump.phase[2].clone(), jump.total.clone()
    jump.reset(torch.tensor([1]), torch.tensor([AIRBORNE]), torch.tensor([0.]), torch.tensor([.42]))
    assert torch.equal(jump.total, previous[1]) and jump.phase[2] == previous[0]


def test_actor_context_is_idempotent_and_contains_only_commands():
    jump = machine(AIRBORNE)
    first = jump.context()
    jump.peak_height.fill_(999.)
    jump.release_vz.fill_(999.)
    jump.wheel_force_peak.fill_(999.)
    assert first.shape == (1, 6) and torch.equal(first, jump.context())


def test_ballistic_reference_has_consistent_release_velocity_and_apex():
    apex = torch.tensor([.40, .50])
    duration, velocity, acceleration = ballistic_push(.23, .36, apex)
    assert torch.allclose(.23+.5*acceleration*duration.square(), torch.full((2,), .36))
    assert torch.allclose(.36+velocity.square()/(2.*9.81), apex)


def test_flight_reward_does_not_penalize_root_height_or_vertical_ballistic_velocity():
    jump = machine(AIRBORNE)
    jump.begin_interval()
    state = dict(height=torch.tensor([.7]), gravity=torch.tensor([[0., 0., -1.]]),
        omega=torch.zeros(1, 3), velocity=torch.tensor([[0., 0., -2.]]), torque=torch.zeros(1, 6),
        leg_speed_square=torch.zeros(1))
    terms = jump.rewards(state, torch.tensor([[[0., 0., -.25], [0., 0., -.25]]]), torch.tensor([-2.]), torch.zeros(1))
    assert terms["jump_height_error"].item() == terms["jump_push_shortfall"].item() == 0.
    assert all(torch.isfinite(x).all() for x in terms.values())


@pytest.mark.parametrize("key,value", [("external_takeoff_assistance", True), ("liftoff_hold_seconds", 0.),
    ("request_delay_s", [2., 1.]), ("motor_owner", "script"), ("roles_per10", ["jump"]*10)])
def test_invalid_contracts_fail_before_native_startup(key, value):
    bad = deepcopy(CFG)
    bad[key] = value
    with pytest.raises(ValueError):
        validate(bad)


def test_jump_task_cli_reaches_the_same_authenticated_plant_and_fresh_learner(tmp_path):
    from train_prone_motion import configuration, learner_config, source_hashes
    from wheeled_tasks.chassis.prone_motion_cli import train_parser, training_options
    parser = train_parser()
    args = parser.parse_args(["--task", "WheelLeg-JumpMotion-v1", "--num-envs", "80", "--max-iterations", "20",
        "--num-steps-per-env", "24", "--domain-randomization", "off"])
    path, plan, output = training_options(args, ROOT)
    config, settings, bank = configuration(tmp_path, plan)
    assert plan["jump_training"] == CFG and output.parent.name == "jump_motion"
    assert config["actor_frame_dim"] == 35 and config["action_dim"] == 6
    assert settings["sensor_policy_dt"] == .01 and config["policy_dt"] == .01
    assert config["physics_dt"] == .005 and config["position_iterations"] == 64 and config["velocity_iterations"] == 32
    agent = learner_config(plan)
    assert not agent["resume"] and agent["num_steps_per_env"] == 24
    hashes = source_hashes(path, plan)
    assert all(name in hashes for name in ("src/wheeled_tasks/chassis/jump_motion.py",
        "src/wheeled_tasks/chassis/jump_motion_env.py", "scripts/probe_jump_motion.py", "scripts/train_jump_motion.sh"))
