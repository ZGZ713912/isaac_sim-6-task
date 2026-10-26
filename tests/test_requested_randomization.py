"""Verify the requested final ranges and actual policy/PD sensor draws."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.randomization_study import resolve, scalar_metrics
from wheeled_tasks.chassis.robust_getup import RobustDomain
from wheeled_tasks.chassis.getup_feedback import GetupFeedback

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def requested():
    return json.loads((ROOT / "contracts/wheel_leg_randomization_study_v2.json").read_text())


def test_requested_final_ranges_and_remaining_ranges(requested):
    r = resolve(requested)
    t, d = r["task"], r["dynamics_randomization"]
    assert t["motor_strength_range"] == [.84, 1.16]
    for key in ("leg_kp_scale_range", "leg_kd_scale_range", "spring_strength_range"):
        assert t[key] == [1., 1.]
    assert d["inertia_scale"] == [.94, 1.06]
    for key in ("base_mass_scale", "leg_mass_scale", "wheel_mass_scale"):
        assert d[key] == [.92, 1.08]
    assert d["base_com_offset_m"] == {"x": [-.012, .012], "y": [-.0064, .0064], "z": [-.008, .008]}
    assert t["nominal_fraction"] == .25 and d["enabled_fraction"] == .75
    assert t["imu_mount"]["fixed_degrees"] == [.8, .8, .8/3.]
    assert t["imu_mount"]["wobble_degrees"] == [0., 0.]
    assert r["imu_wobble_single_axis_gyro_peak_bound_rad_s"] == 0.
    assert t["encoder_bias_rad"] == t["pd_encoder_bias_rad"] == .004
    assert t["encoder_noise_rad"] == .00035
    assert t["gyro_noise_rad_s"] == .0035 and t["gyro_bias_rad_s"] == .00525
    assert t["gravity_noise"] == .0014 and t["velocity_noise_rad_s"] == [.0175, .0875]
    assert t["sensor_delay_policy_ticks"] == [0, 1]
    assert t["torque_delay_ms"] == t["pd_feedback_delay_ms"] == [0, 0]
    assert t["pd_feedback_period_ms"] == [1, 1] and t["motor_tau_ms"] == [0., 0.]
    assert r["ground_friction_range"] == [.3, .5]
    assert r["effective_wheel_ground_friction_range"] == [.2, .3]
    assert r["effective_unsaturated_leg_kp_scale_bounds"] == [.84, 1.16]
    assert r["effective_unsaturated_leg_kd_scale_bounds"] == [.84, 1.16]


@pytest.mark.parametrize("profile,strength", [("nominal", 1.), ("requested", 0.)])
def test_nominal_controls_remove_all_perturbations(requested, profile, strength):
    r = resolve(requested, profile, strength)
    assert r["task"]["domains_enabled"] is False
    assert r["task"]["imu_mount"] is None
    assert r["task"]["sensor_delay_policy_ticks"] == [0, 0]
    assert r["task"]["encoder_noise_rad"] == 0.
    assert r["dynamics_randomization"]["inertia_scale"] == [1., 1.]
    assert r["ground_friction_range"] == [.5, .5]


def test_actual_draws_fix_gains_and_remove_imu_motion(requested):
    r = resolve(requested)
    domain = RobustDomain(512, "cpu", r["task"], 261003, .001)
    ids = torch.arange(512)
    sensors = torch.zeros(512, 18)
    sensors[:, 5] = -1.
    domain.reset(ids, sensors, tick=0)
    assert domain.enabled.any() and (~domain.enabled).any()
    assert torch.all(domain.kp == 1.) and torch.all(domain.kd == 1.) and torch.all(domain.spring == 1.)
    assert domain.strength.min() >= .84 and domain.strength.max() <= 1.16
    assert not domain.imu_mount.amplitude.any()
    fixed = domain.imu_mount.fixed.abs().max(0).values
    assert torch.all(torch.rad2deg(fixed) <= torch.tensor([.8, .8, .8/3.]))
    early = domain.imu_mount.apply(sensors, 0.)
    late = domain.imu_mount.apply(sensors, 137.75)
    assert torch.equal(early, late) and not early[:, :3].any()
    assert torch.allclose(domain.sigma[domain.enabled, 0]*2., torch.tensor(.0035))
    assert torch.allclose(domain.sigma[domain.enabled, 12:16]*10., torch.tensor(.0175))
    assert torch.allclose(domain.sigma[domain.enabled, 16:18]*10., torch.tensor(.0875))
    assert domain.sensor_delay.min() == 0 and domain.sensor_delay.max() == 1
    feedback = GetupFeedback(512, "cpu", domain.enabled, r["task"], domain.rng)
    feedback.reset(ids, torch.zeros(512, 6), torch.zeros(512, 6))
    assert feedback.bias.abs().max() <= .004
    assert not feedback.bias[~domain.enabled].any()
    assert not torch.equal(feedback.bias[:, :4], domain.bias[:, 6:10])


def test_amplitude_scaling_keeps_requested_discrete_timing_explicit(requested):
    r = resolve(requested, strength=.5)
    assert r["task"]["motor_strength_range"] == [.92, 1.08]
    assert r["dynamics_randomization"]["inertia_scale"] == [.97, 1.03]
    assert r["task"]["sensor_delay_policy_ticks"] == [0, 1]
    assert r["task"]["imu_mount"]["fixed_degrees"] == [.4, .4, .4/3.]
    assert r["task"]["leg_kd_scale_range"] == [1., 1.]


def test_actual_inertia_sampler_keeps_mass_coupling_and_shared_mask(requested):
    from wheeled_tasks.chassis.dynamics import RigidBodyRandomization
    cfg = resolve(requested)["dynamics_randomization"]
    mass = torch.ones(512, 3)
    inertia = torch.eye(3).flatten().repeat(512, 3, 1)
    com = torch.zeros(512, 3, 7)
    com[:, :, -1] = 1.
    rigid = RigidBodyRandomization(mass, inertia, com,
        ["base_link", "L_link3", "R_link3"], cfg, 261003)
    assert rigid.enabled.any() and (~rigid.enabled).any()
    assert rigid.mass_scale.min() >= .92 and rigid.mass_scale.max() <= 1.08
    assert rigid.inertia_scale.min() >= .94 and rigid.inertia_scale.max() <= 1.06
    assert torch.all(rigid.inertia_scale[~rigid.enabled] == 1.)
    combined = rigid.mass_scale*rigid.inertia_scale
    assert combined.min() >= .8648 and combined.max() <= 1.1448
    assert torch.equal(rigid.inertias, inertia*combined[..., None])


def test_ablation_removes_only_declared_group(requested):
    base = resolve(requested)
    no_bias = resolve(requested, "without_encoder_bias")
    assert no_bias["task"]["encoder_bias_rad"] == no_bias["task"]["pd_encoder_bias_rad"] == 0.
    for field in ("motor_strength_range", "imu_mount", "encoder_noise_rad", "sensor_delay_policy_ticks"):
        assert no_bias["task"][field] == base["task"][field]
    assert no_bias["dynamics_randomization"] == base["dynamics_randomization"]
    no_delay = resolve(requested, "without_policy_delay")
    assert no_delay["task"]["sensor_delay_policy_ticks"] == [0, 0]


@pytest.mark.parametrize("edit", ["gain", "wobble", "unknown_reference", "inertia", "timing"])
def test_invalid_requested_configs_are_not_silently_ignored(requested, edit):
    if edit == "gain": requested["fixed"]["leg_kp_scale"] = .9
    if edit == "wobble": requested["fixed"]["imu_wobble"] = True
    if edit == "unknown_reference": requested["reference"]["typo"] = 1
    if edit == "inertia": requested["reference"]["independent_inertia_fraction"] = float("nan")
    if edit == "timing": requested["profiles"]["requested"]["timing"] = .5
    with pytest.raises(ValueError): resolve(requested)


def test_configuration_tags_include_physical_dynamics_and_imu_bounds(requested):
    tags = scalar_metrics(resolve(requested))
    assert tags["inertia_scale/min"] == .94 and tags["inertia_scale/max"] == 1.06
    assert tags["imu_fixed_bound_deg/roll"] == .8
    assert tags["leg_kd_scale_range/min"] == 1.
    assert tags["imu_wobble_single_axis_gyro_peak_bound_rad_s"] == 0.


def test_readonly_replacement_binds_task_mechanism_and_design(requested, tmp_path):
    file = tmp_path / "study.json"
    file.write_text(json.dumps(requested))
    module_spec = importlib.util.spec_from_file_location("readonly_domains", ROOT / "scripts/scratch_domain_evaluation.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    cfg = dict(contact_material=dict(ground_static_friction_range=[.5, .5], ground_dynamic_friction_range=[.5, .5]))
    design = dict(stages=dict(speed=dict(roles=dict(train=dict(
        contract=dict(contact_material={}, dynamics_randomization={}),
        runtime=dict(contact_material={}, modules=dict(dynamics_randomization=False)))))),
        review_basis=dict(only_runtime_override={}))
    task = dict(version="scratch", randomization_study={"old": "authenticated"})
    untouched = deepcopy(requested)
    intervention, _ = module.configure(cfg, design, task, file, "requested", 1.)
    assert intervention["replaced_study"] == {"old": "authenticated"}
    row = design["stages"]["speed"]["roles"]["train"]
    assert row["contract"]["dynamics_randomization"] == cfg["dynamics_randomization"]
    assert row["runtime"]["contact_material"] == cfg["contact_material"]
    assert row["runtime"]["modules"]["dynamics_randomization"] is True
    assert task["randomization_study"] == intervention["resolved"]
    assert requested == untouched
