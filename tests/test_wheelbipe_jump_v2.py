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


def test_v3_air_reward_is_relative_to_release_and_pays_reachable_tuck():
    spec = specification("foundation-jump-v3")
    schedule = TaskSchedule(100, "cpu", 119, spec)
    observer = JumpObserver(schedule, .005, torch.full((100,), 20.))
    observer.flown.fill_(True)
    age = schedule.delay + 1.0
    release = torch.zeros(100, 2, 3)
    release[:, :, 2] = .38
    no_tuck = observer.reward_terms(age, torch.full((100,), .45), torch.zeros(100),
                                    release, .02)["jump_air_placement"]
    selected = schedule.mask("jump") | schedule.mask("airborne")
    assert torch.allclose(observer.release_leg_length[selected], torch.full((int(selected.sum()),), .38))
    modest = release.clone()
    modest[:, :, 2] = .34
    modest_reward = observer.reward_terms(age, torch.full((100,), .45), torch.zeros(100),
                                          modest, .02)["jump_air_placement"]
    assert (modest_reward[schedule.mask("jump")] >
            no_tuck[schedule.mask("jump")]).all()
    assert (modest_reward[schedule.mask("jump")] > .5).all()


def test_v3_descent_reward_prefers_partial_reextension_before_landing():
    spec = specification("foundation-jump-v3")
    schedule = TaskSchedule(100, "cpu", 119, spec)
    observer = JumpObserver(schedule, .005, torch.full((100,), 20.))
    observer.flown.fill_(True)
    observer.release_leg_length.fill_(.38)
    observer.release_leg_lengths.fill_(.38)
    observer.air_time.fill_(.3)  # past the explicit tuck-hold phase
    age = schedule.delay + 1.0
    extended = torch.zeros(100, 2, 3)
    extended[:, :, 2] = .35  # .03 m residual tuck, the descent target
    extended_reward = observer.reward_terms(age, torch.full((100,), .45), torch.full((100,), -.2),
                                            extended, .02)["jump_air_placement"]
    extended_extension = observer.last_air_extension.clone()
    still_tucked = extended.clone()
    still_tucked[:, :, 2] = .30
    tucked_reward = observer.reward_terms(age, torch.full((100,), .45), torch.full((100,), -.2),
                                          still_tucked, .02)["jump_air_placement"]
    jump = schedule.mask("jump")
    assert (extended_reward[jump] > tucked_reward[jump]).all()
    assert (extended_extension[jump] > 0.9).all()


def test_v3_landing_buffer_tracks_a_reward_only_quadratic_reference():
    spec = specification("foundation-jump-v3")
    schedule = TaskSchedule(100, "cpu", 119, spec)
    observer = JumpObserver(schedule, .005, torch.full((100,), 20.))
    count = len(schedule.role)
    upward = torch.zeros(count, 3)
    upward[:, 2] = .6
    falling = torch.zeros(count, 3)
    falling[:, 2] = -.4
    gravity = torch.zeros(count, 3)
    gravity[:, 2] = -1.
    airborne_clearance = torch.full((count, 2), .03)
    no_contact_force = torch.zeros(count, 2)
    contact_clearance = torch.zeros(count, 2)
    contact_force = torch.full((count, 2), 100.)
    release_legs = torch.full((count, 2), .38)

    # Establish flight, then hold contact long enough to begin the 20 ms
    # post-contact reference.  The observer must not write any simulation
    # state for this reward-only trajectory.
    for _ in range(10):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .6), upward, gravity,
                         airborne_clearance, no_contact_force, torch.zeros(count),
                         leg_lengths=release_legs)
    for _ in range(4):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .32), falling, gravity,
                         contact_clearance, contact_force, torch.zeros(count),
                         leg_lengths=release_legs)
    selected = schedule.mask("jump") | schedule.mask("airborne")
    assert observer.landing_buffer_active[selected].all()
    assert observer.landing_buffer_used[selected].all()
    assert torch.allclose(observer.landing_buffer_time[selected], torch.zeros(int(selected.sum())))

    wheel_body = torch.zeros(count, 2, 3)
    initial_v0 = 2.*(.24-.32)/.3
    terms = observer.reward_terms(torch.full((count,), 2.), torch.full((count,), .32),
                                  torch.full((count,), initial_v0), wheel_body, .02)
    assert (observer.last_landing_buffer_height[selected] > .99).all()
    assert (observer.last_landing_buffer_velocity[selected] > .99).all()
    assert (terms["jump_height"][selected] > 0.).all()

    # Halfway through the trajectory, the reference is h=.26 m and vz=-.2667
    # m/s for h0=.32 m, target=.24 m and end velocity 0.
    observer.landing_buffer_time.fill_(.15)
    halfway_vz = initial_v0 + ((0.-initial_v0)/.3)*.15
    halfway_h = .32 + initial_v0*.15 + .5*((0.-initial_v0)/.3)*(.15**2)
    observer.reward_terms(torch.full((count,), 2.), torch.full((count,), halfway_h),
                          torch.full((count,), halfway_vz), wheel_body, .02)
    assert (observer.last_landing_buffer_height[selected] > .99).all()
    assert (observer.last_landing_buffer_velocity[selected] > .99).all()

    for _ in range(60):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .30), falling, gravity,
                         contact_clearance, contact_force, torch.zeros(count),
                         leg_lengths=release_legs)
    assert not observer.landing_buffer_active.any()


def test_v3_active_success_requires_measured_bilateral_tuck_but_drop_does_not():
    spec = specification("foundation-jump-v3")
    schedule = TaskSchedule(100, "cpu", 119, spec)
    observer = JumpObserver(schedule, .005, torch.full((100,), 20.))
    count = len(schedule.role)
    velocity = torch.zeros(count, 3)
    velocity[:, 2] = .6
    gravity = torch.zeros(count, 3)
    gravity[:, 2] = -1.
    air_clearance = torch.full((count, 2), .03)
    air_force = torch.zeros(count, 2)
    leg_release = torch.full((count, 2), .38)
    # Establish measured liftoff and release geometry for the active lanes.
    for _ in range(8):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .6), velocity, gravity,
                         air_clearance, air_force, torch.zeros(count), leg_lengths=leg_release)
    # A landing without a bilateral tuck cannot pass the v3 active-jump gate.
    contact_clearance = torch.zeros(count, 2)
    contact_force = torch.full((count, 2), 100.)
    calm_velocity = torch.zeros_like(velocity)
    for _ in range(51):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .305), calm_velocity, gravity,
                         contact_clearance, contact_force, torch.zeros(count), leg_lengths=leg_release)
    jump = schedule.mask("jump")
    airborne = schedule.mask("airborne")
    assert not observer.succeeded[jump].any()
    assert observer.succeeded[airborne].all()

    observer.reset(torch.arange(count))
    for _ in range(8):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .6), velocity, gravity,
                         air_clearance, air_force, torch.zeros(count), leg_lengths=leg_release)
    tucked = torch.full((count, 2), .33)
    for _ in range(10):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .6), velocity, gravity,
                         air_clearance, air_force, torch.zeros(count), leg_lengths=tucked)
    for _ in range(51):
        observer.observe(torch.full((count,), 2.), torch.full((count,), .305), calm_velocity, gravity,
                         contact_clearance, contact_force, torch.zeros(count), leg_lengths=leg_release)
    assert observer.tuck_seen[jump].all()
    assert observer.succeeded[jump].all()


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
