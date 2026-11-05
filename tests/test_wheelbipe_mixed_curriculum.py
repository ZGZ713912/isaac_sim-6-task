from copy import deepcopy

import pytest
import torch

from wheeled_tasks.wheelbipe_port.mixed_curriculum import (
    JumpObserver, TASKS, TaskSchedule, quota_counts, rollout_metrics, validate,
)
from wheeled_tasks.wheelbipe_port.training_profile import configure_profile, phase_for_update, specification
from test_wheelbipe_foundation_profile import fake_cfg


def course(count=100, dt=.005):
    spec = specification("foundation-jump-v1")
    schedule = TaskSchedule(count, "cpu", 119, spec)
    observer = JumpObserver(schedule, dt, torch.full((count,), 20.))
    return schedule, observer


@pytest.mark.parametrize("count", [20, 32, 256, 512, 1024])
def test_every_task_retains_quota_across_individual_resets(count):
    s, _ = course(count)
    before = s.role.clone()
    expected = quota_counts(count, s.spec)
    assert sum(expected.values()) == count
    assert {n: int(s.mask(n).sum()) for n in TASKS} == expected
    for i in range(count):
        s.reset(torch.tensor([i]), 14000)
    assert torch.equal(s.role, before)
    assert all(abs(expected[n]/count-s.spec["task_fractions"][n]) <= 1/count+1e-9 for n in TASKS)


def test_sampler_does_not_consume_the_learner_random_stream():
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    s, _ = course()
    s.update_commands(torch.zeros(100), phase_for_update(0, s.spec))
    assert torch.equal(before, torch.get_rng_state())


def test_bad_quota_and_missing_tasks_fail_before_native_start():
    spec = specification("foundation-jump-v1")
    spec["task_fractions"]["stand"] = .3
    with pytest.raises(ValueError, match="sum to one"):
        validate(spec)
    with pytest.raises(ValueError, match="too small"):
        quota_counts(19, specification("foundation-jump-v1"))


def test_forward_reverse_have_guaranteed_signs_and_low_speed_exposure():
    s, _ = course(1024)
    phase = dict(phase_for_update(16000, s.spec), completed_update=16000)
    commands, heights = s.update_commands(torch.zeros(1024), phase)
    assert (commands[s.mask("forward"), 0] >= .1).all()
    assert (commands[s.mask("reverse"), 0] <= -.1).all()
    for task in ("forward", "reverse"):
        low = (commands[s.mask(task), 0].abs() <= .5).float().mean()
        assert .35 < low < .75
    assert commands[s.mask("spin"), 2].abs().max() > 100*2*torch.pi/60
    assert (commands[s.mask("stand")] == 0).all()
    assert (heights >= .23).all() and (heights <= .42).all()


def test_context_and_references_ignore_all_physical_observer_changes():
    s, observer = course()
    age = torch.full((100,), 1.1)
    base = torch.randn(100, 7)
    before = s.context(base, age)
    reference = deepcopy(s.reference(age))
    observer.flown.fill_(True)
    observer.landed.fill_(True)
    observer.failed.fill_(True)
    assert torch.equal(before, s.context(base, age))
    assert all(torch.equal(reference[n], s.reference(age)[n]) for n in reference)
    nonjump = ~(s.mask("jump") | s.mask("airborne"))
    assert torch.equal(before[nonjump], base[nonjump])


def sample(observer, age=2., *, clear=.03, force=0., vz=.6, height=.50, nonwheel=0.):
    n = len(observer.weight)
    velocity = torch.zeros(n, 3)
    velocity[:, 2] = vz
    gravity = torch.zeros(n, 3)
    gravity[:, 2] = -1.
    observer.observe(torch.full((n,), age), torch.full((n,), height), velocity, gravity,
        torch.full((n, 2), clear), torch.full((n, 2), force), torch.full((n,), nonwheel))


def test_no_force_alone_cannot_award_liftoff_and_drop_is_not_active_jump():
    s, o = course()
    for _ in range(10):
        sample(o, clear=0., force=0.)
    assert not o.liftoff_event.any() and not o.flown.any()
    for _ in range(8):
        sample(o)
    assert o.liftoff_event[s.mask("jump")].all()
    assert not o.liftoff_event[~s.mask("jump")].any()
    assert o.flown[s.mask("airborne")].all()


def test_real_flight_landing_and_stable_hold_pay_each_event_once():
    s, o = course()
    for _ in range(8):
        sample(o)
    assert o.liftoff_event.sum() == s.mask("jump").sum()
    o.begin_interval()
    for _ in range(51):
        sample(o, clear=0., force=100., vz=0., height=.305)
    selected = s.mask("jump") | s.mask("airborne")
    assert o.landing_event[selected].all() and o.success_event[selected].all()
    terms = o.reward_terms(torch.full((100,), 2.), torch.full((100,), .305),
        torch.zeros(100), torch.zeros(100, 2, 3), .02)
    assert torch.allclose(terms["jump_events"][selected]*.02, torch.full((30,), 1.75))
    o.begin_interval()
    sample(o, clear=0., force=100., vz=0., height=.305)
    assert not o.success_event.any() and not o.landing_event.any()


def test_missing_apex_or_bad_support_cannot_award_success():
    s, o = course()
    for _ in range(8):
        sample(o, height=.39)
    for _ in range(60):
        sample(o, clear=0., force=100., vz=0., height=.305)
    assert not o.succeeded[s.mask("jump")].any()
    assert o.succeeded[s.mask("airborne")].all()
    o.reset(torch.arange(100))
    for _ in range(8):
        sample(o)
    for _ in range(60):
        sample(o, clear=0., force=0., vz=0., height=.305)
    assert not o.succeeded.any()


def test_timeout_cannot_be_evaded_or_paid_repeatedly():
    s, o = course()
    sample(o, age=4.5, clear=0., force=100., vz=0., height=.305)
    selected = s.mask("jump") | s.mask("airborne")
    assert o.timeout_event[selected].all()
    o.begin_interval()
    sample(o, age=4.6, clear=0., force=100., vz=0., height=.305)
    assert not o.timeout_event.any()


def test_jump_dense_scale_and_event_units_are_bounded_and_nonjump_payment_is_zero():
    s, o = course()
    age = s.delay+s.spec["jump"]["crouch_seconds"]+.1
    ref = s.reference(age)
    terms = o.reward_terms(age, ref["height"], ref["velocity"], torch.zeros(100, 2, 3), .02)
    assert terms["jump_height"].max() <= 2.
    assert terms["jump_push_velocity"].max() <= 3.
    for value in terms.values():
        assert torch.isfinite(value).all()
        assert (value[~(s.mask("jump") | s.mask("airborne"))] == 0).all()


def test_profile_retains_original_plant_and_base_weights():
    old, new = fake_cfg(), fake_cfg()
    configure_profile(old, "foundation-v2")
    configure_profile(new, "foundation-jump-v1")
    assert all(new.rewards[k] == v for k, v in old.rewards.items())
    assert new.port_gas_feedforward == old.port_gas_feedforward
    assert new.height_range == old.height_range
    assert new.port_mixed_spec["policy_hz"] == 50
    assert new.port_mixed_spec["physics_and_feedback_pd_hz"] == 200


def test_actual_rollout_counts_and_scale_metrics_cover_all_tasks_without_mutation():
    s, _ = course()
    adv = torch.randn(24, 100, 1)
    before = adv.clone()
    metrics = rollout_metrics(s.role, adv, adv*2, adv*3)
    assert sum(metrics["MixedRollout/"+name+"/samples"] for name in TASKS) == 2400
    assert torch.equal(adv, before)
    assert len(metrics) == len(TASKS)*4
