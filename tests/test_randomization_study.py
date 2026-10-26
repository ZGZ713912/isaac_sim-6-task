"""Keep intensity changes explicit and prevent legacy overlay surprises."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.chassis.randomization_study import apply, resolve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec():
    return json.loads((ROOT / "contracts/wheel_leg_randomization_study_v1.json").read_text())


def test_zero_strength_is_nominal_without_hidden_gap_gain(spec):
    result = resolve(spec, "combined_light", 0.)
    assert result["task"]["domains_enabled"] is False
    assert result["task"]["motor_strength_range"] == [1., 1.]
    assert result["task"]["imu_mount"] is None
    assert result["dynamics_randomization"]["base_com_offset_m"]["x"] == [0., 0.]
    assert result["ground_friction_range"] == [.5, .5]


def test_noise_first_does_not_randomize_dynamics_or_timing(spec):
    result = resolve(spec, "sensor_first")
    assert result["task"]["encoder_noise_rad"] == .00025
    assert result["task"]["pd_encoder_bias_rad"] == .001
    assert result["task"]["motor_strength_range"] == [1., 1.]
    assert result["dynamics_randomization"]["inertia_scale"] == [1., 1.]
    assert result["task"]["sensor_delay_policy_ticks"] == [0, 0]
    assert result["task"]["imu_mount"] is None


def test_amplitude_scaled_once_and_couplings_reported(spec):
    result = resolve(spec, "combined_light", .5)
    assert result["task"]["motor_strength_range"] == [.975, 1.025]
    assert result["task"]["encoder_bias_rad"] == .0005
    assert result["effective_unsaturated_leg_kd_scale_bounds"] == pytest.approx([.975*.9625, 1.025*1.0375])
    assert result["policy_PD_encoder_bias_disagreement_bound_rad"] == .001
    assert result["dynamics_randomization"]["enabled_fraction"] == .5


def test_mount_velocity_is_reported_separately_from_white_noise(spec):
    result = resolve(spec, "imu_wobble_only")
    assert result["task"]["gyro_noise_rad_s"] == 0.
    assert result["imu_wobble_single_axis_gyro_peak_bound_rad_s"] == pytest.approx(.0877298169)


def test_explicit_delays_do_not_disappear_when_radius_changes(spec):
    spec["timing"]["sensor_delay_policy_ticks"] = [0, 1]
    result = resolve(spec, "nominal", 0.)
    assert result["task"]["sensor_delay_policy_ticks"] == [0, 1]
    assert result["task"]["domains_enabled"] is True


def test_no_in_place_input_changes_or_stacked_legacy_profile(spec):
    original = deepcopy(spec)
    resolved = resolve(spec)
    cfg = {"contact_material": {}}
    task = {"version": "scratch"}
    apply(cfg, task, resolved)
    assert spec == original and "randomization_study" in task
    with pytest.raises(ValueError): apply(cfg, task, resolved)
    with pytest.raises(ValueError): apply(cfg, {"gap_robustness_version": "gap-v1"}, resolved)


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -.01, 1.01, True])
def test_bad_strength_rejected(spec, strength):
    with pytest.raises(ValueError): resolve(spec, strength=strength)


@pytest.mark.parametrize("field,value", [("torque_delay_ms", [0, 5]),
    ("sensor_delay_policy_ticks", [0, .5]), ("pd_feedback_period_ms", [0, 1]),
    ("pd_feedback_delay_ms", [0, 8])])
def test_history_overflow_or_fractional_tick_rejected(spec, field, value):
    spec["timing"][field] = value
    with pytest.raises(ValueError): resolve(spec)


def test_resolved_noise_reaches_real_sampler_in_physical_units(spec):
    import torch
    from wheeled_tasks.chassis.robust_getup import RobustDomain
    from wheeled_tasks.chassis.getup_feedback import GetupFeedback
    cfg = resolve(spec, "sensor_first")["task"]
    domain = RobustDomain(512, "cpu", cfg, 261003, .001)
    ids = torch.arange(512)
    sensors = torch.zeros(512, 18)
    sensors[:, 5] = -1.
    domain.reset(ids, sensors, tick=0)
    observed = domain.observe(sensors, 0)
    enabled = domain.enabled
    assert enabled.any() and (~enabled).any()
    assert torch.equal(observed[~enabled], sensors[~enabled])
    assert torch.all(domain.strength == 1.) and torch.all(domain.kp == 1.)
    assert torch.all(domain.sensor_delay == 0) and torch.all(domain.motor_alpha == 1.)
    # Actor gyro and motor-velocity slots use0.5 and0.1 respectively.
    assert torch.allclose(domain.sigma[enabled, 0]*2., torch.tensor(.0025))
    assert torch.allclose(domain.sigma[enabled, 12:16]*10., torch.tensor(.0125))
    assert torch.allclose(domain.sigma[enabled, 16:18]*10., torch.tensor(.0625))
    feedback = GetupFeedback(512, "cpu", enabled, cfg, domain.rng)
    zero = torch.zeros(512, 6)
    feedback.reset(ids, zero, zero)
    q, dq = feedback.observe(zero, zero)
    assert torch.equal(q, feedback.bias) and torch.equal(dq, zero)
    assert not feedback.bias[~enabled].any()
    assert feedback.bias.abs().max() <= .001


def test_strength_changes_actual_draw_radius_without_changing_lane_draws(spec):
    import torch
    from wheeled_tasks.chassis.robust_getup import RobustDomain
    domains = []
    for strength in (.5, 1.):
        cfg = resolve(spec, "sensor_first", strength)["task"]
        domain = RobustDomain(512, "cpu", cfg, 261003, .001)
        domain.reset(torch.arange(512), torch.zeros(512, 18), tick=0)
        domains.append(domain)
    small, large = domains
    assert torch.equal(small.enabled, large.enabled)
    assert torch.equal(small.bias*2., large.bias)
    assert torch.equal(small.sigma*2., large.sigma)
    assert torch.equal(small.rng.get_state(), large.rng.get_state())
