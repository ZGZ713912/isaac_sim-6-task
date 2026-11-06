"""Regressions for the observed idle-standing jump-reward loophole."""
import torch

from wheeled_tasks.wheelbipe_port.mixed_curriculum import JumpObserver, TaskSchedule, quota_counts
from wheeled_tasks.wheelbipe_port.training_profile import configure_profile, phase_for_update, specification
from test_wheelbipe_foundation_profile import fake_cfg


def setup():
    spec = specification("foundation-jump-v2")
    schedule = TaskSchedule(100, "cpu", 119, spec)
    observer = JumpObserver(schedule, .005, torch.full((100,), 20.))
    return schedule, observer


def payment(observer, age, height, vz):
    return observer.reward_terms(age, height, vz, torch.zeros(100, 2, 3), .02)


def test_waiting_after_missed_push_cannot_collect_jump_height_reward():
    s, o = setup()
    age = s.delay+1.2
    result = payment(o, age, torch.full((100,), .305), torch.zeros(100))
    jump = s.mask("jump")
    assert (result["jump_height"][jump] < 0).all()
    assert (result["jump_push_velocity"][jump] == 0).all()
    assert (result["jump_air_placement"][jump] == 0).all()
    assert not o.flown.any() and not o.landed.any()


def test_following_crouch_and_push_outscores_idle_at_the_same_clock():
    s, o = setup()
    jump = s.mask("jump")
    for dt in (.3, .45, .55):
        age = s.delay+dt
        ref = s.reference(age)
        tracked = payment(o, age, ref["height"], ref["velocity"])
        idle = payment(o, age, torch.full((100,), .305), torch.zeros(100))
        assert (sum(tracked.values())[jump] > sum(idle.values())[jump]).all()
    age = s.delay+.45
    h = s.reference(age)["height"]
    up = payment(o, age, h, torch.full((100,), .2))["jump_push_velocity"]
    down = payment(o, age, h, torch.full((100,), -.2))["jump_push_velocity"]
    assert (up[jump] > down[jump]).all()


def test_recovery_height_payment_requires_a_real_landing():
    s, o = setup()
    age = s.delay+1.2
    idle = payment(o, age, torch.full((100,), .305), torch.zeros(100))["jump_height"]
    o.flown.fill_(True)
    in_air = payment(o, age, torch.full((100,), .305), torch.zeros(100))["jump_height"]
    o.landed.fill_(True)
    landed = payment(o, age, torch.full((100,), .305), torch.zeros(100))["jump_height"]
    jump = s.mask("jump")
    assert (idle[jump] < 0).all() and (in_air[jump] == 0).all() and (landed[jump] > 0).all()


def test_ordinary_reward_weights_and_explicit_task_coverage_are_preserved():
    a, b = fake_cfg(), fake_cfg()
    configure_profile(a, "foundation-v2")
    configure_profile(b, "foundation-jump-v2")
    assert all(b.rewards[k] == v for k, v in a.rewards.items())
    counts = quota_counts(1024, b.port_mixed_spec)
    assert counts == dict(stand=154, forward=154, reverse=154, height=102,
                         turn=51, dash=51, spin=51, jump=205, airborne=102)
    assert b.port_gas_feedforward is False


def test_forward_and_reverse_jumps_ramp_after_initial_stationary_preparation():
    s, _ = setup()
    phase = dict(phase_for_update(20000, s.spec), completed_update=20000)
    cmd, _ = s.update_commands(torch.zeros(100), phase)
    assert (cmd[s.mask("jump")] == 0).all()
    s.reset(torch.arange(100), 12000)
    cmd, _ = s.update_commands(torch.zeros(100), phase)
    jump_speed = cmd[s.mask("jump"), 0]
    assert (jump_speed > 0).any() and (jump_speed < 0).any() and (jump_speed == 0).any()
    assert jump_speed.abs().max() == .5


def test_jump_command_is_time_only_and_no_contact_can_advance_it():
    s, o = setup()
    age = s.delay+.3
    before = s.reference(age)
    o.flown.fill_(True)
    o.landed.fill_(True)
    o.failed.fill_(True)
    after = s.reference(age)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert (before["height"][s.mask("jump")] < .26).all()


def test_landing_while_tracking_a_moving_command_does_not_require_stopping():
    s, o = setup()
    n = len(s.role)
    velocity = torch.zeros(n, 3)
    velocity[:, 0] = .5
    gravity = torch.zeros(n, 3)
    gravity[:, 2] = -1.
    o.flown.fill_(True)
    o.peak_height.fill_(.6)
    for _ in range(51):
        o.observe(torch.full((n,), 2.), torch.full((n,), .305), velocity, gravity,
            torch.zeros(n, 2), torch.full((n, 2), 100.), torch.zeros(n),
            horizontal_speed_error=torch.zeros(n))
    assert o.succeeded[s.mask("jump")].all()
    o.reset(torch.arange(n))
    o.flown.fill_(True)
    o.peak_height.fill_(.6)
    for _ in range(51):
        o.observe(torch.full((n,), 2.), torch.full((n,), .305), velocity, gravity,
            torch.zeros(n, 2), torch.full((n, 2), 100.), torch.zeros(n),
            horizontal_speed_error=torch.full((n,), .5))
    assert not o.succeeded.any()
