"""Counterexamples for energy, phase ordering and population-correct jump metrics."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.wheelbipe_port.jump_objectives import JumpPhysics, descent_time, energy_shape
from wheeled_tasks.wheelbipe_port.mixed_curriculum import JumpObserver, TaskSchedule, quota_counts, validate
from wheeled_tasks.wheelbipe_port.training_profile import configure_profile, specification
from test_wheelbipe_foundation_profile import fake_cfg


def setup(dt=.005):
    spec = specification('foundation-jump-v8')
    s = TaskSchedule(100, 'cpu', 211, spec)
    o = JumpObserver(s, dt, torch.full((100,), 20.))
    return s, o


def tick(s, o, *, vz=.8, root_vz=None, com_z=.5, height=.5, clear=.15, force=0.,
         length=.38, age=None, motor=0., speed=0.):
    n = len(s.role)
    vec = lambda x: torch.full((n,), x, dtype=torch.float32)
    velocity = torch.zeros(n, 3);velocity[:, 2] = vz if root_vz is None else root_vz
    gravity = torch.zeros(n, 3);gravity[:, 2] = -1.
    wheel_force = torch.full((n, 2), force/2.)
    legs = torch.as_tensor(length).float().expand(n, 2).clone()
    physics = JumpPhysics(vec(com_z), vec(vz), wheel_force,
        torch.full((n, 4), motor), torch.full((n, 4), speed), vec(.2))
    o.observe(s.delay+.4 if age is None else age, vec(height), velocity, gravity,
        torch.full((n, 2), clear), wheel_force, vec(0.), leg_lengths=legs, physics=physics)
    return physics


def rewards(s, o, length=.30):
    wheel = torch.zeros(100, 2, 3);wheel[:, :, 2] = -length
    return o.reward_terms(s.delay+.4, torch.full((100,), .5), torch.full((100,), .8), wheel, .02)


def fly(s, o):
    tick(s, o, clear=0., force=200.)
    tick(s, o, com_z=.42)
    for _ in range(7):
        tick(s, o, com_z=.47)
    assert o.flown[s.mask('jump')].all()


def test_profile_retains_v7_mechanics_commands_quotas_and_the_40mm_gate():
    old, new = specification('foundation-jump-v7'), specification('foundation-jump-v8')
    for key in ('real2sim', 'control', 'phases', 'platform', 'arc_commands', 'observation_contract'):
        assert new[key] == old[key]
    assert quota_counts(2048, new) == quota_counts(2048, old)
    assert new['jump']['minimum_bilateral_tuck_delta_m'] == .04
    cfg = fake_cfg();configure_profile(cfg, 'foundation-jump-v8')
    assert cfg.rewards['joint_torque'] == -.0001
    assert cfg.rewards['jump_release_energy'] == cfg.rewards['jump_clearance'] == 1.
    assert specification('foundation-jump-v8', 10000)['jump']['difficulty_full_update'] == 5000


def test_more_upward_energy_remains_better_above_old_target_and_downward_energy_is_zero():
    speeds = torch.tensor([-3., 0., 1., 1.57, 2., 3., 4.])
    shaped = energy_shape(speeds, .25)
    assert torch.equal(shaped[:2], torch.zeros(2))
    assert (shaped[2:].diff() > 0.).all()
    assert (shaped < 1.).all()


def test_supported_push_is_not_landing_impact_and_held_torque_cannot_earn_energy():
    s, o = setup();jump = s.mask('jump')
    tick(s, o, vz=0., clear=0., force=2000., motor=40., speed=0.)
    terms = rewards(s, o)
    assert o.sequence.push[jump].all()
    assert (terms['jump_impact'][jump] == 0.).all()
    assert (terms['jump_push_velocity'][jump] == 0.).all()
    assert (o.sequence.values['push_work'][jump] == 0.).all()
    assert (o.sequence.values['push_saturation_time'][jump] > 0.).all()


def test_push_improvement_is_tick_independent_and_repeated_small_bounces_do_not_pay():
    payments = []
    for dt, speeds in ((.005, [0., .5, 1., 2., 3.]), (.01, [0., 1., 3.])):
        s, o = setup(dt);jump = s.mask('jump')
        for speed in speeds:
            tick(s, o, vz=speed, clear=0., force=300.)
        first = o.sequence.values['interval_push_payment'].clone()
        for speed in (0., 2., 0., 3.):
            tick(s, o, vz=speed, clear=0., force=300.)
        torch.testing.assert_close(o.sequence.values['interval_push_payment'], first)
        payments.append(first[jump][0])
    torch.testing.assert_close(payments[0], payments[1])


def test_release_energy_is_once_only_and_uses_first_air_com_not_root_or_confirmation_speed():
    s, o = setup();jump = s.mask('jump')
    tick(s, o, clear=0., force=200.)
    tick(s, o, vz=2., root_vz=10., com_z=.42)
    for _ in range(7):
        tick(s, o, vz=.8, root_vz=.1)
    torch.testing.assert_close(o.sequence.values['release_vz'][jump], torch.full((int(jump.sum()),), 2.))
    expected = 4.*energy_shape(torch.tensor(2.), .25)
    torch.testing.assert_close(o.sequence.values['interval_release_bonus'][jump], expected.expand(int(jump.sum())))
    o.begin_interval()
    for _ in range(8):
        tick(s, o)
    assert (o.sequence.values['interval_release_bonus'] == 0.).all()


def test_non_com_upward_motion_and_aborted_air_candidate_cannot_donate_energy():
    s, o = setup();jump = s.mask('jump')
    tick(s, o, clear=0., force=200.)
    for _ in range(8):
        tick(s, o, vz=0., root_vz=2.)
    assert not o.flown[jump].any()
    tick(s, o, vz=.1, force=200., clear=0.)
    for _ in range(8):
        tick(s, o, vz=.4)
    torch.testing.assert_close(o.sequence.values['release_vz'][jump], torch.full((int(jump.sum()),), .4))


def test_preparation_rebound_without_a_supported_push_is_not_active_liftoff():
    s, o = setup();jump = s.mask('jump')
    for _ in range(12):
        tick(s, o, age=s.delay+.1, vz=2.)
    assert not o.flown[jump].any()
    assert (o.sequence.values['interval_release_bonus'][jump] == 0.).all()


def test_tuck_must_be_bilateral_and_held_and_extension_waits_until_late_descent():
    s, o = setup();jump = s.mask('jump');fly(s, o)
    for _ in range(12):
        tick(s, o, length=[.29, .38])
    assert not o.tuck_seen[jump].any()
    tick(s, o, length=.30)
    assert not o.tuck_seen[jump].any()
    for _ in range(7):
        tick(s, o, length=.30)
    assert o.tuck_seen[jump].all()
    for _ in range(12):
        tick(s, o, length=.30, vz=-.1, clear=.4)
    assert o.apex_seen[jump].all() and not o.extension_started[jump].any()
    tick(s, o, length=.30, vz=-1., clear=.04)
    assert o.extension_started[jump].all()
    assert descent_time(torch.tensor(.04), torch.tensor(-1.)) < .05


def test_landing_impact_persists_within_policy_interval_and_is_not_missed():
    s, o = setup();jump = s.mask('jump');fly(s, o);o.begin_interval()
    tick(s, o, vz=-1., clear=0., force=2000.)
    tick(s, o, vz=0., clear=0., force=200.)
    assert (rewards(s, o)['jump_impact'][jump] < 0.).all()
    o.begin_interval()
    tick(s, o, vz=0., clear=0., force=200.)
    assert (rewards(s, o)['jump_impact'][jump] == 0.).all()


def test_success_requires_reextension_and_post_contact_compression_then_counts_real_episodes():
    s, o = setup();jump = s.mask('jump');fly(s, o)
    for _ in range(8):
        tick(s, o, length=.30, com_z=.6)
    tick(s, o, length=.30, vz=-1., clear=.04)
    tick(s, o, length=.34, vz=-1., clear=.03)
    assert o.sequence.values['extended'][jump].all()
    tick(s, o, length=.34, vz=-.4, clear=0., force=200., height=.305)
    for _ in range(55):
        tick(s, o, length=.34, vz=0., clear=0., force=200., height=.305)
    assert not o.succeeded[jump].any()
    tick(s, o, length=.30, vz=0., clear=0., force=200., height=.305)
    assert o.succeeded[jump].all()
    done = jump.clone()
    o.sequence.record_outcomes(o, done, torch.zeros_like(done))
    o.sequence.record_outcomes(o, done, torch.zeros_like(done))
    logs = o.sequence.metrics()
    assert logs['JumpSequence/jump/completed_episodes_total'] == int(jump.sum())
    assert logs['JumpSequence/jump/success_fraction'] == 1.
    o.reset(torch.arange(100))
    assert not o.sequence.success_ready().any()
    assert o.sequence.metrics()['JumpSequence/jump/completed_episodes_total'] == int(jump.sum())


def test_configuration_and_missing_physics_fail_closed():
    s, o = setup()
    with pytest.raises(ValueError, match='require measured COM'):
        o.observe(s.delay+1., torch.ones(100), torch.zeros(100, 3), torch.zeros(100, 3),
            torch.ones(100, 2), torch.zeros(100, 2), torch.zeros(100))
    for key, value in (('energy_height_scale_m', float('nan')), ('leg_motor_cap_nm', 41.),
                       ('urgent_extension_lead_s', .2), ('push_torque_cost_fraction', -1.)):
        spec = deepcopy(s.spec);spec['jump']['sequence_objective'][key] = value
        with pytest.raises(ValueError):
            validate(spec)


def test_reward_entry_applies_dt_once_and_one_leg_reextension_does_not_pass():
    s, o = setup();jump = s.mask('jump');fly(s, o)
    for _ in range(8):
        tick(s, o, length=.30)
    tick(s, o, length=.30, vz=-1., clear=.04)
    tick(s, o, length=[.34, .30], vz=-1., clear=.03)
    assert not o.sequence.values['extended'][jump].any()
    seq = o.sequence
    seq.values['interval_push_payment'].fill_(.05)
    seq.values['interval_release_bonus'].fill_(.7)
    seq.values['interval_clearance_payment'].fill_(.03)
    terms = rewards(s, o)
    for name, expected in (('jump_push_velocity', .05), ('jump_release_energy', .7), ('jump_clearance', .03)):
        torch.testing.assert_close(terms[name]*.02, torch.full((100,), expected))


def test_small_native_sweep_keeps_one_lane_per_other_task_and_covers_candidates():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/probe_wheelbipe_jump.py'
    module_spec = importlib.util.spec_from_file_location('jump_sweep', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    grid = module.candidates([.16, .20, .24], [.08, .12, .18])
    for count in (24, 32, 64):
        spec = module.sweep_course(specification('foundation-jump-v8'), count)
        counts = quota_counts(count, spec)
        assert all(n == 1 for task, n in counts.items() if task != 'jump')
        assert counts['jump'] >= len(grid)


def test_latched_success_followed_by_a_fall_is_not_a_successful_completed_episode():
    s, o = setup();jump = s.mask('jump')
    o.succeeded[jump] = True
    terminated = torch.zeros_like(jump)
    terminated[jump.nonzero().flatten()[0]] = True
    o.sequence.record_outcomes(o, jump, terminated)
    logs = o.sequence.metrics()
    assert logs['JumpSequence/jump/succeeded_episodes_total'] == int(jump.sum())-1
    assert logs['JumpSequence/jump/completed_episodes_total'] == int(jump.sum())
    assert logs['JumpSequence/jump/terminated_episodes_total'] == 1
