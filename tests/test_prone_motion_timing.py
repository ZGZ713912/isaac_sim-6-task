"""Check the new actor period against physical-time and historical contracts."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from train_prone_motion import configuration, learner_config
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.robust_getup import RobustDomain
from wheeled_tasks.chassis.training_wheel_response import configure

ROOT = Path(__file__).resolve().parents[1]
OLD = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v2.json").read_text())


def test_same_rollout_time_with_half_the_actor_samples():
    for plan in (OLD, PLAN):
        task.validate(plan)
        rates = task.timing(plan)
        assert plan["rollout_steps"]*rates["policy_dt"] == .64
        assert plan["rollout_steps"]*rates["decimation"] == 640
        assert rates["physics_dt"] == rates["pc_control_dt"] == .001
    assert PLAN["num_envs"]*PLAN["rollout_steps"] == 16384
    assert PLAN["max_updates"] == OLD["max_updates"] == 10000
    assert PLAN["domain_sequence"] == OLD["domain_sequence"]


@pytest.mark.parametrize("seconds", [.64, 1., 20., 30., 70.])
def test_gamma_and_gae_keep_their_horizons_in_seconds(seconds):
    for name in ("gamma", "lam"):
        old, new = task.discounts(OLD)[name], task.discounts(PLAN)[name]
        assert math.isclose(old**(seconds/.005), new**(seconds/.01), rel_tol=1e-12)


def test_density_integral_and_action_slope_cost_are_rate_independent():
    # Same physical constant-density reward and a linear action ramp.
    paid = []
    for plan in (OLD, PLAN):
        dt = task.timing(plan)["policy_dt"]
        delta = torch.tensor((2.*dt)**2, dtype=torch.float64)
        density = 3.-.01*task.normalized_action_delta(delta, plan)
        paid.append(density*dt*plan["rollout_steps"])
    torch.testing.assert_close(paid[0], paid[1], rtol=1e-12, atol=0.)


@pytest.mark.parametrize("field,bad", [("rollout_steps", 128), ("action_rate_reference_dt", .01),
                                      ("timing", {"physics_dt": .005}), ("discounts", {"gamma": .99})])
def test_refuse_inconsistent_timing_contract(field, bad):
    plan = deepcopy(PLAN)
    plan[field] = bad
    with pytest.raises(ValueError):
        task.validate(plan)


@pytest.mark.parametrize("profile,strength", [("nominal", 0.), ("requested", 1.)])
def test_full_configuration_and_design_use_actual_100hz(tmp_path, profile, strength):
    cfg, settings, _ = configuration(tmp_path, PLAN, profile=profile, strength=strength)
    control = load_chassis_control(ROOT/cfg["control_math_source"])
    contract = json.loads((tmp_path/"task_contract.json").read_text())
    row = json.loads((tmp_path/"config/design.json").read_text())["stages"]["speed"]["roles"]["train"]
    assert control["timing"] == row["control"]["timing"] == task.timing(PLAN)
    for name in ("physics_dt", "policy_dt"):
        assert row["runtime"][name] == row["contract"][name] == cfg[name] == control["timing"][name]
    assert row["runtime"]["decimation"] == 10
    assert contract["policy_hz"] == 100 and contract["rollout_steps"] == 64
    assert cfg["num_steps_per_env"] == 64 and cfg["ppo_minibatch_samples"] == 4096
    assert settings["sensor_policy_dt"] == .01
    assert settings["reward_gamma"] == learner_config(PLAN)["algorithm"]["gamma"]
    assert contract["policy_sensor_delay_ms"] == ([0., 10.] if strength else [0., 0.])
    assert contract["hardware_ready"] is False
    assert control["actuators"]["leg"]["kp"] == 160.
    assert control["actuators"]["leg"]["kd"] == 2.5
    recorded = load_chassis_control(ROOT/PLAN["wheel_model"]["recorded_control"]["file"])
    assert control["actuators"] == recorded["actuators"]


def test_historical_200hz_configuration_still_matches_its_delivered_control(tmp_path):
    cfg, _, _ = configuration(tmp_path, OLD)
    assert cfg["policy_dt"] == .005 and cfg["num_steps_per_env"] == 128
    assert cfg["control_math_source"] == "model/wheel_identified_1562_20261028/control.json"


def test_100hz_derivation_authenticates_its_recorded_controller(tmp_path):
    cfg, settings, _ = configuration(tmp_path, PLAN)
    declaration = deepcopy(PLAN["wheel_model"])
    declaration["recorded_control"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="input SHA.*recorded_control"):
        configure(ROOT, cfg, declaration, settings)


def test_policy_sensor_mount_clock_uses_seconds_at_100hz(tmp_path):
    _, cfg, _ = configuration(tmp_path, PLAN)
    domain = RobustDomain(2, "cpu", cfg, 1, .001)
    calls = []
    domain.imu_mount = SimpleNamespace(apply=lambda x, t: calls.append(t) or x)
    sensors = torch.zeros(2, 18)
    sensors[:, 5] = -1.
    domain.observe(sensors, 100)
    domain.observe(sensors, 100)  # Cached reads must not age or redraw noise.
    assert calls == [1.]
    domain.sensor_delay.fill_(1)
    sensors[:, 0] = 2.
    # One stale frame is exactly one10ms policy period, not a5ms report.
    assert torch.equal(domain.observe(sensors, 101)[:, 0], torch.zeros(2))


def test_new_64_step_learner_is_fresh_and_uses_100hz_discounts():
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from wheeled_algo.scratch_standing import initialize_fresh
    cfg = learner_config(PLAN, "cpu")
    cfg["multi_gpu"] = None
    obs = TensorDict(dict(policy=torch.zeros(8, 35), critic=torch.zeros(8, 83)), batch_size=[8])
    algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=6), cfg, "cpu")
    proof = initialize_fresh(algorithm, torch.zeros(6), initial_std=PLAN["initial_std"])
    assert cfg["num_steps_per_env"] == 64 and cfg["resume"] is False
    assert not algorithm.optimizer.state and proof["optimizer_states"] == 0
    assert not proof["checkpoint_loaded"] and not proof["teacher_loaded"]
    assert all(p.requires_grad for model in (algorithm.actor, algorithm.critic) for p in model.parameters())
    assert algorithm.actor(obs).shape == (8, 6)


def trace_at_100hz():
    from test_prone_motion import synthetic
    original, case = synthetic(prone=True)
    # Re-bin the same1ms physical evidence into10ms actor intervals. State and
    # completion events belong to the END of each interval.
    physical = ("wheel_force", "wheel_clearance", "motor_velocity", "motor_effort")
    arrays = {key: (value.reshape(-1, 10, *value.shape[2:]) if key in physical else value[1::2]).copy()
              for key, value in original.items()}
    for key in ("commands", "requested", "velocity"):
        arrays[key][4000:4300, 0, 0] = 2.
    arrays["position"][:, 0, 0] = np.cumsum(arrays["velocity"][:, 0, 0])*.01
    return arrays, case


@pytest.mark.parametrize("loss_ms", [2, 3])
def test_100hz_contact_excludes_birth_and_counts_raw_milliseconds(loss_ms):
    from evaluate_prone_motion import review_arrays
    arrays, case = trace_at_100hz()
    arrays["wheel_force"][100, :loss_ms, 0] = 0.
    row = review_arrays(arrays, [case], np.zeros((1, 2)), PLAN)[0]
    assert row["full_horizon"] and row["full30s_quiet_window"]
    assert row["post_event_start_sample"] == 100
    assert row["support"]["ignored_loss_ms" if loss_ms == 2 else "retained_loss_ms"] == loss_ms
    assert row["engineering_screen_pass"] == (loss_ms == 2)
    arrays["terminated"][6000, 0] = True
    assert not review_arrays(arrays, [case], np.zeros((1, 2)), PLAN)[0]["engineering_screen_pass"]


def test_100hz_review_requires_full30seconds_and_declared_substeps():
    from evaluate_prone_motion import review_arrays
    arrays, case = trace_at_100hz()
    arrays["valid"][3099:] = False  # Completion at1s, then only29.99s.
    row = review_arrays(arrays, [case], np.zeros((1, 2)), PLAN)[0]
    assert not row["full30s_quiet_window"] and not row["engineering_screen_pass"]
    with pytest.raises(ValueError, match="timing/physical"):
        review_arrays(arrays, [case], np.zeros((1, 2)), PLAN, dt=.005)
