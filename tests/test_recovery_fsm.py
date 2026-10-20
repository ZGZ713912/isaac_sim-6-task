"""Reference/state-machine oracles; these do not certify physical self-righting."""
import math

import pytest
import torch

from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase as P, RecoveryRoute as R


NOMINAL = torch.tensor([.1, .7, 0., -.1, -.7, 0.])
FOLD = torch.tensor([.2, .5, 0., -.2, -.5, 0.])
THRUST = torch.tensor([-.1, .9, 0., .1, -.9, 0.])
SUPPORT = torch.tensor([0., .85, 0., 0., -.85, 0.])
SIGNS = [1., 1., -1., -1.]


def make(count=1, config=None, q=None, active=True, fold=FOLD):
    fsm = RecoveryFSM(count, "cpu", config or {}, NOMINAL, fold, THRUST, SUPPORT, SIGNS)
    if q is None:
        q = NOMINAL.repeat(count, 1)
    fsm.reset(torch.arange(count), q, active)
    return fsm


def update(fsm, *, pitch=0., gravity=None, height=.305, support=True, body_clear=True, gyro=None, dq=None, q=None):
    n = fsm.count
    if gravity is None:
        angle = torch.as_tensor(pitch).expand(n)
        gravity = torch.stack((angle.sin(), torch.zeros(n), -angle.cos()), -1)
    return fsm.update(fsm.script_targets.clone() if q is None else q,
        torch.zeros(n, 6) if dq is None else dq, gravity,
        torch.zeros(n, 3) if gyro is None else gyro,
        torch.as_tensor(height, dtype=torch.float32).expand(n),
        torch.as_tensor(support, dtype=torch.bool).expand(n),
        torch.as_tensor(body_clear, dtype=torch.bool).expand(n))


def until_phase(fsm, target, limit=1601, **kwargs):
    for _ in range(limit):
        if fsm.phase.eq(int(target)).all():
            return
        update(fsm, **kwargs)
    pytest.fail(f"Did not reach {target}; phase={fsm.phase.tolist()}")


def test_pose_based_routes_use_only_gravity_and_estimated_height():
    fsm = make(6)
    a = math.radians(120.)
    gravity = torch.tensor([[0., 0., -1.], [math.sin(.6), 0., -math.cos(.6)],
        [math.sin(a), 0., -math.cos(a)], [-math.sin(a), 0., -math.cos(a)],
        [0., 1., 0.], [0., -1., 0.]])
    update(fsm, gravity=gravity, height=torch.tensor([.305, .15, .1, .1, .1, .1]), support=False)
    assert fsm.route.tolist() == [R.UPRIGHT, R.PLANT, R.ORBIT_POSITIVE, R.ORBIT_NEGATIVE, R.SIDE, R.SIDE]
    assert fsm.phase.tolist() == [P.PREPARE, P.FOLD, P.FOLD, P.FOLD, P.FOLD, P.FOLD]


@pytest.mark.parametrize("direction", [-1., 1.])
def test_contact_extension_preserves_the_completed_orbit_in_both_directions(direction):
    fsm = make(config={"thrust_preserve_orbit_winding": True,
        "orbit_thrust_min_angle_rad": .35, "orbit_thrust_tilt_max_rad": math.radians(115.),
        "thrust_speed_rad_s": .6})
    fsm.phase[:] = P.ORBIT
    fsm.route[:] = R.ORBIT_POSITIVE if direction > 0 else R.ORBIT_NEGATIVE
    reference = fsm.fold + direction * (2 * math.tau + 3.8) / fsm.axis_signs
    fsm.script_targets[:, fsm.LEG_IDS] = reference
    fsm.orbit_start[:] = reference - direction * 3.8 / fsm.axis_signs
    fsm.orbit_angle[:] = 3.8
    q = fsm.script_targets.clone()
    fsm.continuous_q.copy_(q)
    fsm.last_raw_q.copy_(q)
    update(fsm, q=q, pitch=direction * math.radians(105.), height=.24)
    assert fsm.phase.item() == P.THRUST
    contact = fsm.script_targets[:, fsm.LEG_IDS].clone()
    aligned = fsm.thrust - direction * math.radians(105.) / fsm.axis_signs
    expected = contact + fsm.paired_delta(aligned[None], contact)
    torch.testing.assert_close(fsm.thrust_anchor, expected, atol=1e-6, rtol=0.)
    # Contact selects a gravity-aligned extension on the existing winding;
    # the next tick retains the slow extension bound.
    update(fsm, pitch=direction * math.radians(105.), height=.24)
    step = fsm.script_targets[:, fsm.LEG_IDS] - contact
    assert step.abs().max() <= .6 * .005 + 1e-6
    assert fsm.script_targets[:, [0, 1]].sign().eq(direction).all()


def test_fold_reference_is_synchronized_and_speed_bounded():
    folded = NOMINAL.clone()
    folded[list(fsm_ids := RecoveryFSM.LEG_IDS)] += torch.tensor([1., 2., -1., -2.])
    fsm = make(fold=folded)
    before = fsm.script_targets[:, fsm_ids].clone()
    update(fsm, pitch=.6, height=.1, support=False)
    step = fsm.script_targets[:, fsm_ids] - before
    torch.testing.assert_close(step, torch.tensor([[.0025, .005, -.0025, -.005]]), atol=1e-7, rtol=0.)
    assert fsm.script_targets[:, fsm.WHEEL_IDS].eq(0.).all()


def test_hip_selects_auxiliary_winding_instead_of_independent_shortest_arcs():
    goal = torch.tensor([[3.1, 5., -3.1, -5.]])
    reference = torch.tensor([[-3.1, -1.2, 3.1, 1.2]])
    delta = RecoveryFSM.paired_delta(goal, reference)
    raw = (goal - reference).reshape(1, 2, 2)
    result = delta.reshape(1, 2, 2)
    torch.testing.assert_close(result[..., 1] - result[..., 0], raw[..., 1] - raw[..., 0])
    assert delta.abs().max() < .2


@pytest.mark.parametrize("direction", [-1, 1])
def test_intentional_full_revolution_preserves_both_pairs_and_encoder_unwrap(direction):
    fsm = make()
    kwargs = dict(pitch=direction * math.radians(120.), height=.1, support=False)
    until_phase(fsm, P.ORBIT, **kwargs)
    start = fsm.script_targets[:, fsm.LEG_IDS].clone()
    previous = start.clone()
    for _ in range(400):
        # Motor sensors may wrap at pi while the desired rotation stays continuous.
        raw = torch.atan2(fsm.script_targets.sin(), fsm.script_targets.cos())
        update(fsm, q=raw, **kwargs)
        current = fsm.script_targets[:, fsm.LEG_IDS]
        physical_step = (current - previous) * torch.tensor(SIGNS)
        assert (physical_step * direction >= -1e-6).all()
        assert physical_step.abs().max() <= .02001
        physical = (current - start) * torch.tensor(SIGNS)
        torch.testing.assert_close(physical, physical[:, :1].expand_as(physical), atol=3e-5, rtol=0.)
        previous = current.clone()
        if fsm.failed.item():
            break
    assert fsm.failure_code.item() == 4
    assert (physical.abs() >= math.tau).all()
    assert fsm.continuous_q[:, fsm.LEG_IDS].abs().max() > math.pi


def test_side_swing_is_paired_opposed_and_reroutes_after_pose_changes():
    fsm = make(2)
    gravity = torch.tensor([[0., 1., 0.], [0., -1., 0.]])
    until_phase(fsm, P.SIDE, gravity=gravity, height=.1, support=False)
    start = fsm.script_targets[:, fsm.LEG_IDS].clone()
    for _ in range(15):
        update(fsm, gravity=gravity, height=.1, support=False)
    motion = (fsm.script_targets[:, fsm.LEG_IDS] - start) * torch.tensor(SIGNS)
    torch.testing.assert_close(motion[:, 0], motion[:, 1])
    torch.testing.assert_close(motion[:, 2], motion[:, 3])
    torch.testing.assert_close(motion[:, 0], -motion[:, 2])
    torch.testing.assert_close(motion[0], -motion[1])
    update(fsm, pitch=math.radians(120.), height=.1, support=False)
    assert fsm.route.eq(int(R.ORBIT_POSITIVE)).all() and fsm.phase.eq(int(P.FOLD)).all()


def test_side_without_reorientation_fails_in_bounded_time():
    fsm = make()
    until_phase(fsm, P.FAILED, gravity=torch.tensor([[0., 1., 0.]]), height=.1, support=False)
    assert fsm.failure_code.item() == 2 and fsm.age_ticks.item() < 1600


def test_plant_requires_deployable_support_before_prepare():
    fsm = make()
    until_phase(fsm, P.PLANT, pitch=.6, height=.15, support=False)
    for _ in range(100):
        update(fsm, pitch=.6, height=.15, support=False)
    assert fsm.phase.item() == P.PLANT
    until_phase(fsm, P.PREPARE, pitch=.6, height=.15, support=True)


def test_orbit_contact_enters_thrust_then_capture_then_prepare():
    fsm = make()
    until_phase(fsm, P.ORBIT, pitch=math.radians(120.), height=.1, support=False)
    until_phase(fsm, P.THRUST, pitch=math.radians(120.), height=.15, support=True)
    before = fsm.script_targets.clone()
    update(fsm, pitch=math.radians(50.), height=.3, support=True)
    assert fsm.phase.item() == P.CAPTURE
    assert (fsm.script_targets - before).abs().max() <= .02001
    update(fsm, pitch=.1, height=.305, support=True)
    assert fsm.phase.item() == P.PREPARE


@pytest.mark.parametrize("direction", [-1, 1])
def test_slow_sagittal_sequence_pauses_folded_then_rotates_selected_direction(direction):
    fsm = make(config={"sagittal_plant_before_orbit": True, "plant_requires_support": False,
        "plant_dwell_seconds": .12, "fold_speed_rad_s": .8, "stand_speed_rad_s": .6,
        "thrust_speed_rad_s": .6, "orbit_speed_rad_s": 1.2,
        "capture_speed_rad_s": .8, "orbit_thrust_min_angle_rad": .35,
        "orbit_thrust_tilt_max_rad": math.radians(80.)})
    kwargs = dict(pitch=direction * math.radians(120.), height=.1, support=False)
    until_phase(fsm, P.PLANT, **kwargs)
    assert fsm.route.item() == (R.ORBIT_POSITIVE if direction > 0 else R.ORBIT_NEGATIVE)
    compact = fsm.script_targets[:, fsm.LEG_IDS].clone()
    for _ in range(23):
        update(fsm, **kwargs)
        assert fsm.phase.item() == P.PLANT
        torch.testing.assert_close(fsm.script_targets[:, fsm.LEG_IDS], compact)
    update(fsm, **kwargs)
    assert fsm.phase.item() == P.ORBIT
    for _ in range(100):
        before = fsm.script_targets[:, fsm.LEG_IDS].clone()
        update(fsm, **kwargs)
        step = (fsm.script_targets[:, fsm.LEG_IDS] - before) * torch.tensor(SIGNS)
        assert (step * direction >= -1e-6).all()
        assert step.abs().max() <= 1.2 * .005 + 1e-6
    # A supported pose inside the declared tilt gate can stop rotation before
    # the legacy 3.7 rad threshold; standing has its own lower speed bound.
    update(fsm, pitch=direction * math.radians(60.), height=.3, support=True)
    assert fsm.phase.item() == P.THRUST and fsm.orbit_angle.item() < 3.7
    before = fsm.script_targets.clone()
    update(fsm, pitch=direction * math.radians(50.), height=.3, support=True)
    assert fsm.phase.item() == P.CAPTURE
    assert (fsm.script_targets - before).abs().max() <= .6 * .005 + 1e-6


def test_legacy_prepare_and_push_speeds_still_cover_the_original_phases():
    fsm = make(config={"prepare_speed_rad_s": .7, "push_speed_rad_s": 2.5})
    assert fsm.config["fold_speed_rad_s"] == .7
    assert fsm.config["stand_speed_rad_s"] == fsm.config["thrust_speed_rad_s"] == 2.5


def test_low_grip_capture_does_not_lower_the_extended_leg_at_a_large_body_tilt():
    fsm = make(config={"thrust_capture_tilt_max_rad": math.radians(25.)})
    fsm.phase[:] = P.THRUST
    update(fsm, pitch=math.radians(45.), height=.305)
    assert fsm.phase.item() == P.THRUST
    update(fsm, pitch=math.radians(20.), height=.305)
    assert fsm.phase.item() == P.CAPTURE


def test_slow_extension_does_not_slow_the_whole_leg_attitude_compensation():
    fsm = make(config={"stand_speed_rad_s": .5, "separate_upright_rotation_from_extension": True,
        "upright_rotation_speed_rad_s": 3.})
    fsm.phase[:] = P.PREPARE
    before = fsm.script_targets[:, fsm.LEG_IDS].clone()
    update(fsm, pitch=.4, height=.305)
    step = (fsm.script_targets[:, fsm.LEG_IDS] - before).reshape(1, 2, 2)
    rotation = step.mean(-1)
    extension = (step[:, :, 1] - step[:, :, 0]) / 2
    torch.testing.assert_close(rotation.abs().amax(), torch.tensor(3. * .005), atol=1e-6, rtol=0.)
    assert extension.abs().max() <= .5 * .005 + 1e-6
    assert extension.abs().max() > 0
    assert step.abs().max() <= (3. + .5) * .005 + 1e-6


def test_handover_requires_sensor_stability_support_and_clear_chassis():
    fsm = make(4)
    for _ in range(40):
        update(fsm, support=torch.tensor([False, True, True, True]),
            body_clear=torch.tensor([True, False, True, True]),
            gyro=torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]))
    assert fsm.phase[:3].eq(int(P.PREPARE)).all()
    assert fsm.blending[3]


def test_exact_200ms_blend_then_one_second_continuously_stable_pure_rl():
    fsm = make()
    until_phase(fsm, P.BLEND)
    assert fsm.blend.item() == 0. and fsm.motion_hold.item()
    for tick in range(1, 40):
        update(fsm)
        assert fsm.blending.item() and fsm.blend.item() == pytest.approx(tick / 40, abs=1e-6)
        assert not fsm.pure_rl.item() and fsm.stable_ticks.item() == 0
    update(fsm)
    assert fsm.pure_rl.item() and fsm.blend.item() == 1. and fsm.motion_hold.item()
    for _ in range(199):
        update(fsm)
    assert fsm.motion_hold.item() and fsm.stable_ticks.item() == 199
    update(fsm)
    assert fsm.motion_released.item() and not fsm.motion_hold.item()


def test_unstable_pure_rl_resets_stability_clock_without_restarting_script():
    fsm = make()
    until_phase(fsm, P.RL)
    for _ in range(150):
        update(fsm)
    update(fsm, support=False)
    assert fsm.pure_rl.item() and fsm.stable_ticks.item() == 0 and fsm.motion_hold.item()
    for _ in range(199):
        update(fsm)
    assert fsm.motion_hold.item()
    update(fsm)
    assert not fsm.motion_hold.item()


def test_blend_is_inside_script_budget_but_stable_rl_wait_is_separate():
    fsm = make(config={"max_script_seconds": .25})
    until_phase(fsm, P.FAILED)
    assert fsm.failure_code.item() == 6 and fsm.age_ticks.item() == 50
    fsm = make(config={"max_script_seconds": .35})
    until_phase(fsm, P.RL)
    for _ in range(300):
        update(fsm, support=False)
    assert not fsm.failed.item() and fsm.motion_hold.item()


def test_lost_upright_during_takeover_fails_and_failure_is_terminal():
    fsm = make()
    until_phase(fsm, P.BLEND)
    update(fsm, pitch=math.radians(60.))
    assert fsm.failed.item() and fsm.failure_code.item() == 5
    target = fsm.script_targets.clone()
    for _ in range(20):
        update(fsm)
    assert fsm.failed.item() and fsm.motion_hold.item() and fsm.blend.item() == 0.
    torch.testing.assert_close(fsm.script_targets, target)


def test_nonfinite_sensor_fails_without_poisoning_reference_or_other_lane():
    fsm = make(2)
    q = NOMINAL.repeat(2, 1)
    q[0, 0] = float("nan")
    update(fsm, q=q)
    assert fsm.failed.tolist() == [True, False] and fsm.failure_code[0] == 1
    assert torch.isfinite(fsm.script_targets).all()


def test_reset_is_local_inputs_are_not_mutated_and_inactive_is_pure_rl():
    fsm = make(3, active=torch.tensor([True, True, False]))
    for _ in range(5):
        update(fsm)
    before = {key: value.clone() for key, value in vars(fsm).items()
              if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 3}
    q = NOMINAL.repeat(3, 1)
    q[0] += 2 * math.pi
    original = q.clone()
    fsm.reset(torch.tensor([0]), q, torch.tensor([True, True, False]))
    for key, value in before.items():
        torch.testing.assert_close(getattr(fsm, key)[1:], value[1:])
    torch.testing.assert_close(q, original)
    assert fsm.age_ticks[0] == 0 and fsm.phase[0] == P.SELECT
    assert fsm.pure_rl[2] and fsm.blend[2] == 1. and not fsm.motion_hold[2] and not fsm.scripted[2]


@pytest.mark.parametrize("config", [{"dt": .001}, {"blend_seconds": .1}, {"stable_seconds": .9},
    {"max_script_seconds": 9.}, {"prepare_speed_rad_s": float("nan")}, {"typo": 1.},
    {"max_reroutes": 2}, {"max_reroutes": .5}, {"max_reroutes": True},
    {"reroute_tilt_rad": .1}, {"reroute_tilt_rad": math.pi}])
def test_invalid_clock_budget_or_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        make(config=config)


def test_constructor_rejects_nonfinite_pose_and_inconsistent_pair_axes():
    bad = NOMINAL.clone()
    bad[1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        make(fold=bad)
    with pytest.raises(ValueError, match="axis signs"):
        RecoveryFSM(1, "cpu", {}, NOMINAL, FOLD, THRUST, SUPPORT, [1., -1., -1., -1.])


@pytest.mark.parametrize("phase", [P.PREPARE, P.CAPTURE])
def test_transient_support_loss_does_not_reroute_and_timer_requires_continuity(phase):
    fsm = make()
    fsm.phase[:] = phase
    for _ in range(29):
        update(fsm, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.phase.item() == phase and fsm.reroute_count.item() == 0
    update(fsm, pitch=math.radians(120.), support=True, height=.1)
    assert fsm.unsupported_ticks.item() == 0
    for _ in range(29):
        update(fsm, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.phase.item() == phase and fsm.reroute_count.item() == 0


@pytest.mark.parametrize("phase", [P.PREPARE, P.CAPTURE])
def test_sustained_unsupported_pose_reselects_route_without_resetting_winding_or_target(phase):
    q = NOMINAL.repeat(1, 1)
    q[:, [0, 1]] += 2 * math.tau
    q[:, [3, 4]] -= 2 * math.tau
    fsm = make(q=q)
    fsm.phase[:] = phase
    fsm.age_ticks[:] = 500
    fsm.continuous_pitch[:] = 2 * math.tau
    fsm.pitch_initialized[:] = True
    for _ in range(29):
        update(fsm, pitch=math.radians(120.), support=False, height=.1)
    before = fsm.script_targets.clone()
    measured = fsm.continuous_q.clone()
    update(fsm, q=measured, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.phase.item() == P.FOLD and fsm.route.item() == R.ORBIT_POSITIVE
    assert fsm.reroute_count.item() == 1 and fsm.age_ticks.item() == 530
    assert fsm.phase_ticks.item() == 0 and fsm.continuous_pitch.item() > 2 * math.tau
    torch.testing.assert_close(fsm.continuous_q, measured)
    assert (fsm.script_targets - before).abs().max() <= .00501
    assert fsm.script_targets[:, [0, 1]].min() > math.tau
    assert fsm.script_targets[:, [3, 4]].max() < -math.tau


def test_reroute_is_once_only_and_does_not_refund_the_original_eight_second_budget():
    fsm = make()
    fsm.phase[:] = P.PREPARE
    fsm.age_ticks[:] = 1539  # 7.695 s already consumed.
    for _ in range(30):
        update(fsm, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.phase.item() == P.FOLD and fsm.age_ticks.item() == 1569
    assert fsm.reroute_count.item() == 1
    # Even after re-entering capture, sustained loss cannot buy another retry.
    fsm.phase[:] = P.CAPTURE
    for _ in range(30):
        update(fsm, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.phase.item() == P.CAPTURE and fsm.age_ticks.item() == 1599
    assert fsm.reroute_count.item() == 1
    update(fsm, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.failed.item() and fsm.failure_code.item() == 6 and fsm.age_ticks.item() == 1600


@pytest.mark.parametrize("phase", [P.BLEND, P.RL])
def test_takeover_never_restarts_script_on_sustained_loss(phase):
    fsm = make()
    fsm.phase[:] = phase
    for _ in range(35):
        update(fsm, pitch=math.radians(120.), support=False, height=.1)
    assert fsm.failed.item() and fsm.failure_code.item() == 5
    assert fsm.reroute_count.item() == 0


def test_reroute_accounting_is_per_environment_and_reset_is_local():
    fsm = make(3)
    fsm.phase[:] = P.PREPARE
    for _ in range(30):
        update(fsm, pitch=math.radians(120.), support=torch.tensor([False, True, False]), height=.1)
    assert fsm.reroute_count.tolist() == [1, 0, 1]
    diagnostics = fsm.diagnostics()["recovery_reroute_count"]
    fsm.reset([0], NOMINAL.repeat(3, 1), True)
    assert fsm.reroute_count.tolist() == [0, 0, 1]
    assert diagnostics.tolist() == [1, 0, 1]


def test_reroute_requires_large_tilt_and_can_be_explicitly_disabled_for_comparison():
    upright = make()
    disabled = make(config={"max_reroutes": 0})
    upright.phase[:] = disabled.phase[:] = P.PREPARE
    for _ in range(40):
        update(upright, pitch=math.radians(60.), support=False, height=.1)
        update(disabled, pitch=math.radians(120.), support=False, height=.1)
    for fsm in (upright, disabled):
        assert fsm.phase.item() == P.PREPARE and fsm.reroute_count.item() == 0
