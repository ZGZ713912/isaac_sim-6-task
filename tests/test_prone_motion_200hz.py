"""Physical-time and provenance regressions for the coarse training option."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from train_prone_motion import configuration, learner_config, source_hashes
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.control_contract import load_chassis_control, COARSE_POLICY_100_TIMING
from wheeled_tasks.chassis.getup_feedback import GetupFeedback
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options, LEGACY_SOLVER_OPTION_KEYS
from wheeled_tasks.chassis.robust_getup import RobustDomain, settings
from wheeled_tasks.chassis.training_wheel_response import configure

ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE = ROOT/"contracts/wheel_leg_prone_motion_v4.json"
BASE = json.loads(PLAN_FILE.read_text())


@pytest.mark.parametrize("profile,strength", [("nominal", 0.), ("requested", 1.)])
def test_cli_200hz_preserves_physics_inputs_and_policy_horizons(tmp_path, profile, strength):
    args = train_parser().parse_args(["--physics-hz", "200", "--position-iterations", "64",
        "--velocity-iterations", "32", "--num-envs", "256", "--max-iterations", "10000", "--seed", "106"])
    _, plan, _ = training_options(args, ROOT)
    assert task.validate(plan) == plan
    cfg, domain, _ = configuration(tmp_path/"coarse", plan, profile=profile, strength=strength)
    fine, _, _ = configuration(tmp_path/"fine", resolve_plan(BASE), profile=profile, strength=strength)
    control = load_chassis_control(ROOT/cfg["control_math_source"])
    recorded = load_chassis_control(ROOT/BASE["wheel_model"]["control"]["file"])
    rates = task.timing(plan)
    assert rates == control["timing"] == COARSE_POLICY_100_TIMING
    assert cfg["physics_dt"] == .005 and cfg["policy_dt"] == .01
    assert (cfg["position_iterations"], cfg["velocity_iterations"]) == (64, 32)
    assert plan["rollout_steps"]*rates["policy_dt"] == .64
    assert plan["rollout_steps"]*rates["decimation"] == 128
    assert plan["num_envs"]*plan["rollout_steps"] == 16384
    assert control["actuators"] == recorded["actuators"]
    for key in ("asset_manifest_sha256", "full_vehicle", "leg_motor_friction", "v5_control",
                "identified_wheel_response", "closure_gap_termination_m", "gimbal_motion"):
        assert cfg[key] == fine[key]
    for key in ("global_curriculum", "multi_pose_recovery", "recovery_impact", "roles_per20", "evaluation"):
        assert plan[key] == BASE[key]
    assert learner_config(plan)["algorithm"] == learner_config(BASE)["algorithm"]
    assert learner_config(plan)["max_iterations"] == 10000 and not learner_config(plan)["resume"]
    assert domain["sensor_policy_dt"] == .01
    saved = json.loads((tmp_path/"coarse/task_contract.json").read_text())
    assert (saved["physics_hz"], saved["policy_hz"]) == (200, 100)
    assert saved["simulation_timing_approximation"]["hardware_equivalence_verified"] is False
    row = json.loads((tmp_path/"coarse/config/design.json").read_text())["stages"]["speed"]["roles"]["train"]
    assert row["control"]["timing"] == rates and row["runtime"]["decimation"] == 2
    assert source_hashes(PLAN_FILE, plan)[cfg["control_math_source"]] == cfg["control_math_sha256"]


@pytest.mark.parametrize("options", [dict(physics_hz=100), dict(physics_hz=True), dict(physics_hz=200.),
    dict(physics_hz=200, position_iterations=12, velocity_iterations=6),
    dict(physics_hz=200, velocity_iterations=31), dict(physics_hz=200, recovery_impact_window_ms=3.)])
def test_invalid_coarse_clocks_or_solver_fail_before_native_launch(options):
    with pytest.raises(ValueError):
        resolve_plan(BASE, **options)


def test_historical_resolved_plans_and_explicit_fine_option_stay_readable():
    old = resolve_plan(BASE, position_iterations=12, velocity_iterations=6)
    old["runtime_cli"]["options"] = {k:v for k,v in old["runtime_cli"]["options"].items() if k in LEGACY_SOLVER_OPTION_KEYS}
    assert task.validate(old) == old
    fine = resolve_plan(BASE, physics_hz=1000)
    assert task.timing(fine) == task.timing(BASE) and fine["wheel_model"] == BASE["wheel_model"]
    assert "simulation_timing_approximation" not in fine
    legacy = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())
    with pytest.raises(ValueError):resolve_plan(legacy, physics_hz=200)
    coarse = resolve_plan(BASE, physics_hz=200)
    coarse["timing"]["decimation"] = 1
    with pytest.raises(ValueError):task.validate(coarse)


def test_coarse_derivation_rejects_changed_physical_prior_even_with_new_hash(tmp_path):
    plan = resolve_plan(BASE, physics_hz=200)
    declaration = deepcopy(plan["wheel_model"])
    for value in declaration.values():
        if isinstance(value, dict) and "file" in value:
            dst = tmp_path/value["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT/value["file"]).read_bytes())
    path = tmp_path/declaration["control"]["file"]
    bad = json.loads(path.read_text());bad["actuators"]["leg"]["kp"] = 159.
    path.write_text(json.dumps(bad))
    declaration["control"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="derivation may change only"):
        configure(tmp_path, {}, declaration, {})


@pytest.mark.parametrize("dt", [.001, .005])
@pytest.mark.parametrize("delay_ms", [0, 1, 3, 5, 7, 11])
def test_encoder_delay_retains_milliseconds_and_is_causal(dt, delay_ms):
    feedback = GetupFeedback(2, "cpu", torch.tensor([True, False]), dict(
        pd_feedback_period_ms=[1,1], pd_feedback_delay_ms=[delay_ms,delay_ms], pd_encoder_bias_rad=0.),
        torch.Generator().manual_seed(43), dt)
    feedback.reset(torch.arange(2), torch.zeros(2,6), torch.zeros(2,6))
    for tick in range(20):
        time_ms = tick*dt*1000.
        q, dq = feedback.observe(torch.full((2,6), time_ms), torch.full((2,6), -time_ms))
        torch.testing.assert_close(q[0], torch.full((6,), max(time_ms-delay_ms, 0.)), atol=2e-5, rtol=1e-6)
        torch.testing.assert_close(dq[0], -q[0])
        torch.testing.assert_close(q[1], torch.full((6,), time_ms), rtol=0., atol=0.)


def test_encoder_hold_period_rounds_up_to_native_steps():
    f = GetupFeedback(1, "cpu", torch.tensor([True]), dict(pd_feedback_period_ms=[7,7],
        pd_feedback_delay_ms=[0,0], pd_encoder_bias_rad=0.), torch.Generator().manual_seed(1), .005)
    f.reset(torch.tensor([0]), torch.zeros(1,6), torch.zeros(1,6))
    observed = [float(f.observe(torch.full((1,6), tick*5.), torch.zeros(1,6))[0][0,0]) for tick in range(5)]
    assert observed == [0., 0., 10., 10., 20.]


@pytest.mark.parametrize("dt", [.001, .005])
@pytest.mark.parametrize("delay_ms", [0, 1, 3, 7])
def test_torque_delay_and_sensor_report_retain_actual_units(dt, delay_ms):
    cfg = settings(nominal_fraction=0., motor_strength_range=[1.,1.], motor_tau_ms=[0.,0.],
        torque_delay_ms=[delay_ms,delay_ms], sensor_delay_policy_ticks=[1,1], sensor_policy_dt=.01)
    # settings() scales robustness ranges from zero; this fixture selects one
    # fixed physical delay rather than sampling that range.
    cfg.update(torque_delay_ms=[delay_ms,delay_ms], sensor_delay_policy_ticks=[1,1])
    domain = RobustDomain(2, "cpu", cfg, 43, dt)
    sensors = torch.zeros(2,18);sensors[:,5] = -1.
    domain.reset(torch.arange(2), sensors)
    for tick in range(20):
        time_ms = tick*dt*1000.
        actual = domain.apply_torque(torch.full((2,6), time_ms), torch.tensor([False, True]))
        torch.testing.assert_close(actual[0], torch.full((6,), max(time_ms-delay_ms,0.)), atol=2e-5, rtol=1e-6)
        assert torch.count_nonzero(actual[1]) == 0
    assert domain.summary()["torque_delay_ms_max"] == delay_ms
    assert math.isclose(domain.summary()["sensor_delay_ms_max"], 10.)


def test_contact_durations_are_real_ms_and_coarse_data_cannot_claim_2ms_acceptance():
    from review_standing_acceptance import support_review
    result = support_review(np.array([True,False,True]), standing_start_sample=0, sample_period_ms=5.)
    assert result["raw_missing_ms"] == result["retained_loss_ms"] == 5.
    assert result["evaluated_physical_ms"] == 15. and not result["support_pass"]
    supported = support_review(np.ones(10,dtype=bool), standing_start_sample=0, sample_period_ms=5.)
    assert supported["sampled_support_pass"] and not supported["contact_tolerance_resolved"]
    assert not supported["support_pass"]
