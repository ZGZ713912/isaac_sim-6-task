"""Controller convention checks, independent of any simulator or policy reward."""
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.controller_handoff import CONTROL_JOINT_NAMES, STANDING_POLICY_JOINT_NAMES
from wheeled_tasks.chassis.wheelbipe_interface import WheelbipeStandingInterface
from wheeled_tasks.chassis.handoff_evaluation import CompletedPhysicsClock, HandoffEvaluation, admit_nominal, contact_summary


def interface(count=1):
    nominal = torch.tensor([-.42, .1372233, 0., .42, -.1372161, 0.])
    model = {"sides": {s: {"delta_rad": [-.14, 1.43]} for s in ("left", "right")}}
    return WheelbipeStandingInterface(nominal, model, count)


def test_fresh_standing_frame_uses_named_axes_and_no_recovery_features():
    c = interface()
    q = c.nominal[None]+torch.tensor([[.1, .2, 17., .3, .4, -19.]])
    dq = torch.tensor([[1., 2., 3., 4., 5., 6.]])
    frame = c.observations(q, dq, torch.tensor([[.2, .4, .6]]), torch.tensor([[0., 0., -1.]]),
                           torch.tensor([[.7, .8, .9, .305]]))
    assert frame.shape == (1, 35)
    assert frame[0, :10].tolist() == pytest.approx([.7, .8, .9, 1.525, .1, .2, .3, 0., 0., -1.])
    assert frame[0, 10:16].tolist() == pytest.approx([.1, .3, .2, .4, 0., 0.], abs=1e-6)
    assert frame[0, 16:22].tolist() == pytest.approx([.1, .4, .2, .5, .3, .6])
    assert frame[0, 22:28].tolist() == [0.]*6
    assert frame[0, 28:].tolist() == [1., 0., 0., 0., 0., 0., 0.]


def test_raw_action_history_keeps_standing_order_and_does_not_alias_input():
    c = interface()
    action = torch.tensor([[.1, .2, .3, .4, 100., -100.]])
    original = action.clone()
    legs, wheels = c.targets(action, c.nominal[None])
    assert legs[0].tolist() == pytest.approx([-.37, .2872233, .52, .0627839], abs=1e-6)
    assert wheels.tolist() == [[90., -90.]]
    action.zero_()
    assert torch.equal(c.previous_action, original)


@pytest.mark.parametrize("turns", [(1, 1), (-2, 3), (4, -3)])
def test_common_encoder_turn_does_not_change_fixed_policy_observation(turns):
    c = interface()
    q = c.nominal.expand(1, -1).clone()
    turned = q.clone()
    turned[:, :2] += turns[0]*2*torch.pi
    turned[:, 3:5] += turns[1]*2*torch.pi
    assert torch.allclose(c.canonical_legs(turned), q, atol=2e-6, rtol=0)


def test_target_projection_preserves_pair_mean_and_declared_branch():
    c = interface(3)
    action = torch.tensor([[20., -30., -10., 30., 0., 0.], [-20., 10., 30., -20., 1., -1.], [0.]*6])
    legs, _ = c.targets(action, c.nominal.expand(3, -1))
    pair = legs.unflatten(-1, (2, 2))
    delta = (pair[..., 1]-pair[..., 0])*torch.tensor([1., -1.])
    assert bool((delta >= -.140001).all() and (delta <= 1.430001).all())
    assert bool((pair.mean(-1)-c.nominal[[0, 1, 3, 4]].reshape(2, 2).mean(-1)).abs().max() <= torch.pi)


def test_component_clipping_happens_before_scale():
    c = interface()
    f = c.observations(c.nominal[None], torch.full((1, 6), 1000.), torch.full((1, 3), 1000.),
                       torch.tensor([[0., 0., -1.]]), torch.tensor([[1000., -1000., 0., 2.]]))
    assert f[0, :5].tolist() == [100., -100., 0., 5., 50.]
    assert f[0, 16:22].tolist() == [20.]*6


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_rejected_without_overwriting_history(value):
    c = interface()
    action = torch.zeros(1, 6);action[0, 0] = value
    with pytest.raises(ValueError):
        c.targets(action, c.nominal[None])
    assert not bool(c.previous_action.any())


def test_unqualified_communication_is_rejected_before_touching_environment():
    env = SimpleNamespace(num_envs=2)
    with pytest.raises(ValueError):
        admit_nominal(env)
    assert vars(env) == {"num_envs": 2}


def test_per_body_impulse_precedes_group_reduction_and_clips_fractional_oldest_interval():
    # Switching the loaded leg must not add two different bodies' peaks.
    rows = [dict(time=t, dt=dt, contact_force=f) for t, dt, f in (
        (.009, .009, [[0, 0, 0], [0, 0, 100], [0, 0, 0]]),
        (.018, .009, [[0, 0, 0], [0, 0, 0], [0, 0, 100]]),
        (.027, .009, [[0, 0, 50], [0, 0, 0], [0, 0, 0]]))]
    r = contact_summary(rows, ["base_link", "L_link1", "R_link1"], 0., .027)
    assert r["group_positive_world_z_20ms_peak_ns"]["legs"] == pytest.approx(.9)
    assert r["group_positive_world_z_20ms_peak_ns"]["chassis"] == pytest.approx(.45)
    assert r["group_positive_world_z_20ms_peak_ns"]["wheels"] == 0.


def test_impulse_excludes_disabled_intervals_and_samples_after_handoff():
    rows = [dict(time=t, dt=.005, contact_force=[[0, 0, f]])
            for t, f in ((.005, 1000), (.010, 10), (.015, 1000))]
    r = contact_summary(rows, ["base_link"], .005, .010)
    assert r["group_positive_world_z_20ms_peak_ns"]["chassis"] == pytest.approx(.05)


def test_native_clock_commits_only_fetched_intervals_and_keeps_the_delegate():
    calls = []
    native = SimpleNamespace(simulate=lambda *a: calls.append(a), fetch_results=lambda: "fetched", marker="original")
    clock = CompletedPhysicsClock(native)
    for dt in (.001, .001, .005, .005):
        before = clock.time_s
        clock.simulate(dt, 0.)
        assert clock.time_s == before
        assert clock.fetch_results() == "fetched"
        assert clock.last_dt == dt
    assert clock.steps == 4 and clock.time_s == pytest.approx(.012)
    assert calls == [(.001, 0.), (.001, 0.), (.005, 0.), (.005, 0.)]
    assert clock.native is native and clock.marker == "original"


def test_failed_native_fetch_does_not_advance_recorded_physics_time():
    def fail():
        raise RuntimeError("solver failure")
    clock = CompletedPhysicsClock(SimpleNamespace(simulate=lambda *a: None, fetch_results=fail))
    clock.simulate(.005, 0.)
    with pytest.raises(RuntimeError, match="solver failure"):
        clock.fetch_results()
    assert clock.steps == 0 and clock.time_s == 0.
    with pytest.raises(RuntimeError, match="Uncommitted"):
        clock.simulate(.005, 0.)


def test_passive_placement_does_not_consume_the_active_recovery_budget():
    evaluator = SimpleNamespace(time=20., enabled_at=.16)
    assert not HandoffEvaluation.recovery_budget_exhausted(evaluator, 20.)
    evaluator.time = 20.16
    assert HandoffEvaluation.recovery_budget_exhausted(evaluator, 20.)


def test_standing_reference_has_no_measured_recovery_impulse():
    r = contact_summary([dict(time=.005, dt=.005, contact_force=[[0, 0, 100.]])], ['base_link'], None, None)
    assert r['measurement_available'] is False
