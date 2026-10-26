import math

import pytest
import torch
from torch import nn

from wheeled_tasks.chassis.imu_mount import ImuMount, euler_matrix
from wheeled_tasks.chassis.gap_robustness import apply_profile, rigid_profile, leg_quality
from wheeled_tasks.chassis.robust_getup import RobustDomain, settings
from wheeled_algo.gap_robust_actor import GapRobustGetupActor, entry_gate


def sensors(count=1):
    x = torch.zeros(count, 18)
    x[:, 5] = -1.
    return x


def mount(count=1):
    return ImuMount(count, "cpu", dict(fixed_degrees=[3., 3., 1.],
        wobble_degrees=[0., .5], wobble_hz=[1., 8.]), torch.Generator().manual_seed(123))


def test_fixed_mount_is_a_single_persistent_rotation_for_gyro_and_gravity():
    m = mount()
    m.fixed[:, 1] = math.radians(2.)
    x = sensors()
    x[:, 0] = .5
    first, later = m.apply(x, 0.), m.apply(x, 12.)
    assert torch.equal(first, later)
    rotation = euler_matrix(m.fixed).transpose(-1, -2)
    torch.testing.assert_close(first[:, :3], (rotation @ x[:, :3, None]).squeeze(-1))
    torch.testing.assert_close(first[:, 3:6], (rotation @ x[:, 3:6, None]).squeeze(-1))
    assert first[0, 3] == pytest.approx(math.sin(math.radians(2.)), abs=1e-7)
    assert torch.equal(first[:, 6:], x[:, 6:])


def test_moving_mount_gyro_matches_derivative_of_rotated_gravity():
    m = mount()
    m.fixed[:] = torch.tensor([[.02, -.03, .01]])
    m.amplitude[:] = torch.tensor([[.01, .008]])
    m.frequency[:] = torch.tensor([[5., 3.]])
    t, epsilon = .127, 1e-5
    x = sensors()
    result = m.apply(x, t)
    derivative = (m.apply(x, t+epsilon)[:, 3:6]-m.apply(x, t-epsilon)[:, 3:6])/(2.*epsilon)
    expected = -torch.cross(result[:, :3]*2., result[:, 3:6], dim=-1)
    torch.testing.assert_close(derivative, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(result[:, 3:6].norm(dim=-1), torch.ones(1))


def test_nominal_lanes_and_zero_mount_are_exact_and_partial_reset_is_local():
    m = mount(4)
    m.reset(torch.arange(4), torch.tensor([False, True, True, False]))
    before = {k:v.clone() for k,v in m.parameters().items()}
    x = sensors(4)
    x[:, :3] = torch.randn(4, 3)
    assert torch.equal(m.apply(x, .5)[[0, 3]], x[[0, 3]])
    m.reset(torch.tensor([1]), torch.tensor([True]))
    for k,v in m.parameters().items():
        assert torch.equal(v[[0, 2, 3]], before[k][[0, 2, 3]])


def test_mount_reset_does_not_leak_clean_imu_through_delay_history():
    cfg = apply_profile(settings(sensor_noise_scale=0., nominal_fraction=0.))
    d = RobustDomain(3, "cpu", cfg, 1, .001)
    d.reset(torch.arange(3), sensors(3))
    assert not torch.equal(d.sensor_history[0, :, 3:6], sensors(3)[:, 3:6])
    d.sensor_delay.fill_(1)
    result = d.observe(sensors(3), 0).clone()
    assert torch.equal(result, d.observe(sensors(3), 0))
    assert not torch.equal(result[:, 3:6], sensors(3)[:, 3:6])


def frame(count=1):
    x = torch.zeros(count, 35)
    x[:, 9] = -1.
    return x


def test_floor_entry_uses_orbit_not_large_standing_knee_and_handles_wrap():
    x = frame(4)
    x[0, 10:14] = torch.tensor([.815, 1.558, -.849, -1.622])
    x[1, 10:14] = torch.tensor([-.4, .4, .4, -.4])
    x[2, 10:14] = x[0, 10:14]+2.*math.pi
    x[3, 9] = 1.
    torch.testing.assert_close(entry_gate(x).flatten(), torch.tensor([1., 0., 1., 0.]))


def test_handoff_orbit_keeps_authority_at_the_observed_early_release():
    from wheeled_algo.gap_entry_handoff import handoff_entry_gate, HandoffGapGetupActor
    x = frame(4)
    x[0, 10:14] = torch.tensor([.53, 1.20, -.55, -1.15])
    x[1, 10:14] = torch.tensor([-.4, .4, .4, -.4])
    x[2, 10:14] = .29
    x[3, 9] = 1.
    assert handoff_entry_gate(x)[0] == 1.
    assert entry_gate(x)[0] < .5
    assert torch.equal(handoff_entry_gate(x)[1:], torch.zeros(3, 1))
    torch.manual_seed(17)
    old = GapRobustGetupActor(nn.Linear(35, 6), upright_entry_recovery=True)
    new = HandoffGapGetupActor(nn.Linear(35, 6), upright_entry_recovery=True)
    new.load_state_dict(old.state_dict(), strict=True)
    assert torch.equal(old(x)[1:], new(x)[1:])
    assert torch.isfinite(new(x)).all()


@pytest.mark.parametrize("handoff_version", ["orbit-v2", "invalid"])
@pytest.mark.parametrize("explicit_gap_flag", [False,True])
def test_resume_keeps_the_handoff_function_in_training_and_task_metadata(tmp_path,handoff_version,explicit_gap_flag):
    from types import SimpleNamespace
    from train_robust_getup import prepare
    parent = tmp_path/"parent.pt"
    torch.save({"infos": {"settings": {"gap_entry_handoff_version": handoff_version,
                "gap_robustness_version":"gap-v1","gap_severity":.6,
                "gap_actuator_timing_profile":"native",
                "native_origin_mode": "isolated_overlap_v1"}}}, parent)
    bank = tmp_path/"bank.json"
    bank.write_text("{}")
    args = SimpleNamespace(output=tmp_path/"run",bank=bank,resume=parent,no_domains=False,
        curriculum_updates=128,quiet_leg_alpha=.25,quiet_wheel_alpha=.65,domain_scale=.35,
        gap_robustness=explicit_gap_flag,gap_severity=.6,num_envs=32,rollout_steps=96,
        mode="train",friction="high",friction_min=.3,seed=123)
    if handoff_version=="invalid":
        with pytest.raises(ValueError,match="Unknown gap entry"):
            prepare(args)
    else:
        _,task = prepare(args)
        assert task["gap_entry_handoff_version"] == "orbit-v2"
        assert task["gap_actuator_timing_profile"]=="native"
        assert task["gap_severity"]==.6 and task["motor_tau_ms"]==[0.,0.]


def test_native_timing_profile_retains_all_sensor_mass_gain_and_friction_domains():
    stress = apply_profile(settings(),.6)
    native = apply_profile(dict(settings(),gap_actuator_timing_profile="native"),.6)
    for key,expected in dict(motor_tau_ms=[0.,0.],torque_delay_ms=[0,0],
                             pd_feedback_delay_ms=[0,0],pd_feedback_period_ms=[1,1]).items():
        assert native.pop(key)==expected
        stress.pop(key)
    assert native.pop("gap_actuator_timing_profile")=="native"
    assert native==stress


def test_entry_actor_is_bounded_differentiable_and_preserves_nonentry_output():
    torch.manual_seed(12)
    old = GapRobustGetupActor(nn.Linear(35, 6), canonical_recovery=True,
        smooth_knee_targets=True, recovery_leg_step=.3)
    new = GapRobustGetupActor(nn.Linear(35, 6), canonical_recovery=True,
        smooth_knee_targets=True, recovery_leg_step=.3, upright_entry_recovery=True)
    new.load_state_dict(old.state_dict(), strict=True)
    x = frame(3)
    x[1, 9] = 1.
    x[2, 10:14] = torch.tensor([.815, 1.558, -.849, -1.622])
    x[2, 22:26] = x[2, 10:14]/.25
    action = new(x)
    assert torch.equal(action[:2], old(x)[:2])
    assert (action[2, :4]*.25-x[2, 10:14]).abs().max() <= .120001
    action[2].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in new.recovery.parameters())


def test_entry_exploration_is_smaller_for_legs_and_has_exact_sample_likelihood():
    from wheeled_algo.gap_robust_actor import GapSensorConditionedMean
    actor = GapRobustGetupActor(nn.Linear(35, 6), upright_entry_recovery=True)
    distribution = GapSensorConditionedMean(actor)
    x = frame()
    x[:, 10:14] = torch.tensor([[.815, 1.558, -.849, -1.622]])
    parameters = distribution(x)
    torch.testing.assert_close(parameters[:, 1, :4].exp(), torch.full((1, 4), .1))
    torch.testing.assert_close(parameters[:, 1, 4:].exp(), torch.full((1, 2), .5))
    normal = torch.distributions.Normal(parameters[:, 0], parameters[:, 1].exp())
    sample = normal.sample()
    replay = distribution(x)
    assert torch.equal(normal.log_prob(sample), torch.distributions.Normal(replay[:, 0], replay[:, 1].exp()).log_prob(sample))


def test_quality_cost_targets_excess_speed_saturation_and_slew():
    cfg = apply_profile(settings())["gap_leg_quality"]
    cost = leg_quality(torch.tensor([1., 12., 12.]), torch.tensor([0., 1., 1.]),
        torch.tensor([0., 20., 20.]), torch.tensor([True, True, False]), cfg)
    assert cost[0] == 0. and cost[1] > 0. and cost[2] == 0.
    for severity in (0., -1., 1.1, float("nan")):
        with pytest.raises(ValueError):
            apply_profile(settings(), severity)
    body = rigid_profile({}, 1.)
    for key in ("base_mass_scale", "leg_mass_scale", "wheel_mass_scale"):
        assert body[key] == [.9, 1.1]
    assert body["base_com_offset_m"]["x"] == [-.015, .015]


def test_user_mass_range_scales_with_curriculum_without_changing_other_rigid_domains():
    prior = {"unrelated_setting": 123}
    body = rigid_profile(prior, .6)
    for key in ("base_mass_scale", "leg_mass_scale", "wheel_mass_scale"):
        assert body[key] == pytest.approx([.94, 1.06])
    assert body["inertia_scale"] == pytest.approx([.88, 1.12])
    assert body["base_com_offset_m"]["x"] == pytest.approx([-.009, .009])
    assert body["unrelated_setting"] == 123
    assert prior == {"unrelated_setting": 123}


def test_randomized_pd_scales_before_clipping_and_preserves_requested_torque():
    from wheeled_tasks.chassis.gap_robustness import randomized_leg_pd
    error = torch.tensor([[-.5, .5, .1, -.1]])
    velocity = torch.tensor([[0., 0., 2., -2.]])
    tau, requested = randomized_leg_pd(error, velocity, .8, 1.2, 160., 2.5)
    torch.testing.assert_close(requested, torch.tensor([[-64., 64., 6.8, -6.8]]))
    torch.testing.assert_close(tau, torch.tensor([[-40., 40., 6.8, -6.8]]))


@pytest.fixture(scope="module")
def extended_bank(tmp_path_factory):
    import json
    from pathlib import Path
    from build_robust_getup_bank import build
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("gap_bank")/"bank.json"
    build(output, True, root/"tests/data/real_start_leg_positions_20261023.json")
    return json.loads(output.read_text())


def test_extended_bank_preserves_legacy_and_closes_real_start_legs(extended_bank):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    from build_robust_getup_bank import Builder, digest
    legacy = json.loads((root/"reports/unified_robust_20261021/pose_bank002.json").read_text())
    bank = extended_bank
    seeds = json.loads((root/"tests/data/real_start_leg_positions_20261023.json").read_text())
    assert bank["poses"][:261] == legacy["poses"]
    bundle = root/"model/纯底盘_v7_flat_base107/urdf"
    builder = Builder(bundle, digest(bundle/"manifest.json"))
    actual = [p for p in bank["poses"] if p.get("real_start_reconstruction")]
    assert len(actual) == len(seeds["cases"]) == 6
    for pose, seed in zip(actual, seeds["cases"]):
        builder.validate_q(pose["full_joint_positions"])
        for name, value in zip(("L_joint1", "LL_joint1", "R_joint1", "RR_joint1"), seed["model_q_p_order_rad"][:4]):
            assert pose["full_joint_positions"][name] == pytest.approx(value, abs=1e-9)


def test_fixed_diagnostic_grid_covers_all_mount_profiles_and_real_starts(extended_bank):
    from evaluate_gap_robustness import cases
    rows = cases(extended_bank)
    assert len(rows) == 51
    assert len([c for c in rows if c["behavior_case"].startswith("real_start_")]) == 12
    for name in ("stand_low", "stand", "stand_high"):
        assert {c["imu_case"] for c in rows if c["behavior_case"] == name} == {
            "clean", "pitch_plus2", "pitch_minus2", "roll_plus2", "roll_minus2", "wobble", "combined"}
