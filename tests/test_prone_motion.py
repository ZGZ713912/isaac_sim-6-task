"""Task regressions: reward credit, command coverage and honest completion windows."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.unified_getup import project_commands
from train_prone_motion import configuration, learner_config, verify_native_probe
from evaluate_prone_motion import review_arrays, CAPABILITIES

PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())


def test_population_and_full_budget():
    assert task.validate(PLAN)["max_updates"] == 10000
    assert len(CAPABILITIES) == 13
    for key in ("max_updates", "post_completion_quiet_seconds", "forward_speed_m_s"):
        bad = deepcopy(PLAN)
        bad[key] /= 2
        with pytest.raises(ValueError):
            task.validate(bad)


@pytest.mark.parametrize("completed", [0, 999, 1000, 3000, 5000, 6000, 7000, 10000])
def test_coverage_amplitude_and_projection(completed):
    values = torch.tensor([task.target(r, completed, PLAN) for r in task.ROLES])
    effective = project_commands(values, dict(wheel_track_m=.4373, wheel_radius_m=.06,
        wheel_speed_limit_rad_s=90., lateral_acceleration_m_s2=2.06))
    assert torch.allclose(values, effective)
    if completed <= 1000:
        assert not values[:, :2].any()
    if completed >= 7000:
        assert values[2, 0] == 2 and values[3, 0] == -2
        assert (values[6:10, 0].abs() == 2).all()
        assert values[4, 1].item() == pytest.approx(100*math.tau/60)
        assert values[5, 1].item() == pytest.approx(-100*math.tau/60)
        assert set((int(torch.sign(v)), int(torch.sign(w))) for v,w in values[6:10, :2]) == {(1,1),(1,-1),(-1,1),(-1,-1)}


def test_motion_waits_for_actual_completion_and30seconds():
    times = torch.tensor([40., 40., 40., 40.])
    recovered = torch.tensor([float("nan"), 11., 8., 5.])
    targets = torch.tensor([[2., 0., .305]]*4)
    actual = task.commands(targets, times, recovered, torch.ones(4, dtype=torch.bool), PLAN)
    assert not actual[:3, 0].any()  # never completed,29s quiet,32s including startup quiet
    assert actual[3, 0] == 2.


def rewards(height, speed, yaw):
    return task.reward_terms(torch.tensor([[0.,0.,-1.]]), torch.tensor([height]),
        torch.tensor([[speed,0.,0.]]), torch.tensor([[0.,0.,yaw]]), torch.tensor([[2.,1.,.42]]),
        torch.zeros(1,6), torch.zeros(1,6), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        torch.zeros(1,dtype=torch.bool), torch.ones(1))


def test_bad_height_does_not_erase_motion_credit():
    for height in (.173, .305, .42):
        bad, good = rewards(height, 0., 0.), rewards(height, 2., 1.)
        assert good["forward"] > bad["forward"]
        assert good["yaw"] > bad["yaw"]
        assert good["forward_error"] > bad["forward_error"]
        assert good["yaw_error"] > bad["yaw_error"]
    assert rewards(.173, 0., 0.)["forward"] == rewards(.42, 0., 0.)["forward"]


def test_spin_holds_its_center_without_penalizing_translation_distance():
    common = [torch.tensor([[0.,0.,-1.]]), torch.tensor([.305]), torch.zeros(1,3), torch.zeros(1,3)]
    rest = [torch.zeros(1,6), torch.zeros(1,6), torch.zeros(1),torch.tensor([.2]),torch.tensor([.2]),
            torch.zeros(1,dtype=torch.bool),torch.ones(1)]
    spin = task.reward_terms(*common,torch.tensor([[0.,10.,.305]]),*rest)
    forward = task.reward_terms(*common,torch.tensor([[2.,0.,.305]]),*rest)
    assert spin["quiet_position"] < 0
    assert forward["quiet_position"] == 0 and forward["rise_position"] == 0


def test_rise_allows_belly_contact_but_refall_fails():
    result = task.failures(torch.tensor([[0.,0.,-1.]]*2), torch.tensor([.173,.173]), torch.tensor([100.,100.]),
        torch.tensor([.09,.09]), torch.tensor([5.,5.]), torch.tensor([True,False]), torch.tensor([False,False]), PLAN)
    assert result["refall"].tolist() == [False,True]
    assert not result["rise_excursion"].any()
    assert not result["rise_timeout"].any()


@pytest.mark.parametrize("profile,strength", [("nominal",0.), ("sensors_only",.25), ("inertia_only",.25), ("requested",1.)])
def test_config_keeps_physics_and_v2_contract(tmp_path, profile, strength):
    cfg, settings, bank = configuration(tmp_path, PLAN, profile=profile, strength=strength)
    assert bank.is_file()
    assert cfg["target_num_envs"] == 256 and cfg["record_diagnostics"]
    assert cfg["stages"][0]["vx_max"] == 2.
    gas = cfg["v5_control"]["identified_gas_spring"]
    assert gas["enabled"] and gas["right_leg_symmetry_assumed"]
    assert gas["model"]["kind"] == "effective_monotone_table"
    assert gas["outside_identified_domain"] == "endpoint_hold_unvalidated"
    assert settings["episode_seconds"] == 70
    assert settings["leg_action_clip"] == 16 and cfg["v5_control"]["action_clip"] == 16
    assert cfg["contact_material"]["wheel_static_friction"] == .7
    assert cfg["wheel_reduction"]["gear_ratio"] == 15.62
    if profile != "nominal":
        assert settings["leg_kp_scale_range"] == [1.,1.]
        assert settings["leg_kd_scale_range"] == [1.,1.]
        assert settings["spring_strength_range"] == [1.,1.]
        assert settings["imu_mount"]["wobble_degrees"] == [0.,0.]
    if profile == "requested":
        assert cfg["dynamics_randomization"]["inertia_scale"] == [.94,1.06]
        assert settings["nominal_fraction"] == .25


def test_native_probe_population_can_use_the_actual_scene_allocator(tmp_path):
    from wheeled_tasks.chassis.task import choose_scene_groups
    cfg, _, _ = configuration(tmp_path, PLAN, num_envs=80)
    assert len(choose_scene_groups(cfg["scene_groups"], 80)) == cfg["target_num_envs"]
    assert all((PLAN["roles_per20"]*4).count(role) > 0 for role in task.ROLES)


@pytest.mark.parametrize("fault", [None, "status", "check", "domain", "source", "PPO", "checkpoint", "missing"])
def test_ppo_requires_a_complete_saved_native_probe(tmp_path, fault):
    result = dict(status="native_probe_completed_unqualified", completed_updates=0, optimizer_attempts=0,
        checkpoint_reads=[], source_mismatches_at_exit=[], all_native_checks_pass=True,
        contexts=[dict(profile="nominal", strength=0.), dict(profile="requested", strength=1.)],
        native_probes=[dict(all_checks_pass=True, new_PPO=0, checks=dict(force=True)) for _ in range(2)])
    if fault == "status": result["status"] = "failed"
    if fault == "check": result["native_probes"][0]["checks"]["force"] = False
    if fault == "domain": result["contexts"].pop()
    if fault == "source": result["source_mismatches_at_exit"] = ["changed.py"]
    if fault == "PPO": result["completed_updates"] = 1
    if fault == "checkpoint": result["checkpoint_reads"] = ["old.pt"]
    if fault != "missing": (tmp_path/"progress.json").write_text(json.dumps(result))
    if fault:
        with pytest.raises((RuntimeError, FileNotFoundError)): verify_native_probe(tmp_path)
    else:
        assert verify_native_probe(tmp_path)["all_native_checks_pass"]


def synthetic(prone=False):
    n = 14000
    arrays = {k:np.zeros(shape, dtype=dtype) for k,shape,dtype in (
        ("valid",(n,1),bool), ("completed_event",(n,1),bool), ("terminated",(n,1),bool),
        ("position",(n,1,3),float), ("velocity",(n,1,3),float), ("omega",(n,1,3),float),
        ("gravity",(n,1,3),float), ("commands",(n,1,3),float), ("requested",(n,1,3),float),
        ("height",(n,1),float), ("time",(n,1),float), ("root_quaternion",(n,1,4),float),
        ("wheel_force",(n,5,1,2,3),float), ("wheel_clearance",(n,5,1,2),float),
        ("motor_velocity",(n,5,1,6),float), ("motor_effort",(n,5,1,6),float), ("gap",(n,1),float))}
    arrays["valid"][:] = True
    arrays["height"][:] = .305
    arrays["commands"][:,:,2] = arrays["requested"][:,:,2] = .305
    arrays["gravity"][:,:,2] = -1
    arrays["root_quaternion"][:,:,3] = 1
    arrays["wheel_force"][:,:,:,:,2] = 20
    arrays["time"][:,0] = (np.arange(n)+1)*.005
    case = dict(role="prone" if prone else "stand", recovery=prone, birth_height=.173 if prone else .305,
                pitch_deg=0., target=[2.,0.,.305] if prone else [0.,0.,.305])
    if prone:
        arrays["completed_event"][199,0] = True
        arrays["wheel_force"][:200] = 0  # excluded before first actual completion
    return arrays, case


def test_no_later_suffix_can_hide_refall():
    a, case = synthetic()
    assert review_arrays(a,[case],np.zeros((1,2)),PLAN)[0]["engineering_screen_pass"]
    a["terminated"][12000,0] = True
    assert not review_arrays(a,[case],np.zeros((1,2)),PLAN)[0]["engineering_screen_pass"]


def test_contact_starts_after_event_and_keeps3ms_whole():
    a, case = synthetic(prone=True)
    a["wheel_force"][200, :2,0] = 0
    row = review_arrays(a,[case],np.zeros((1,2)),PLAN)[0]
    assert row["post_event_start_sample"] == 200
    assert row["support"]["ignored_loss_ms"] == 2
    assert row["support"]["support_pass"]
    a["wheel_force"][200, :3,0] = 0
    row = review_arrays(a,[case],np.zeros((1,2)),PLAN)[0]
    assert row["support"]["retained_loss_ms"] == 3
    assert not row["support"]["support_pass"]


def test_partial_quiet_cannot_pass():
    a, case = synthetic(prone=True)
    a["valid"][6100:] = False
    row = review_arrays(a,[case],np.zeros((1,2)),PLAN)[0]
    assert not row["full30s_quiet_window"]
    assert not row["engineering_screen_pass"]


def test_native_hook_is_opt_in_and_does_not_drop_physical_guards():
    source = (ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py").read_text()
    ast.parse(source)
    assert "terminated = mechanical | gap_bad | impact_bad | boundary_bad" in source
    assert "density, progress, terminated, task_extra = self.task_transition" in source
    assert 'physical_samples.append' in source
    assert 'self.robot.write_root_link_pose_to_sim' not in source.split('def step(self, actions):',1)[1]


def test_actual_new_architecture_starts_with_empty_adam_and_six_trainable_outputs():
    from types import SimpleNamespace
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from wheeled_algo.scratch_standing import initialize_fresh
    cfg = learner_config(PLAN, "cpu")
    cfg["multi_gpu"] = None
    obs = TensorDict(dict(policy=torch.zeros(8,35), critic=torch.zeros(8,83)), batch_size=[8])
    algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8,num_actions=6),cfg,"cpu")
    proof = initialize_fresh(algorithm,torch.zeros(6),initial_std=PLAN["initial_std"])
    assert not algorithm.optimizer.state and proof["optimizer_states"] == 0
    assert not proof["checkpoint_loaded"] and not proof["teacher_loaded"]
    assert all(p.requires_grad for p in algorithm.actor.parameters())
    assert algorithm.actor(obs).shape == (8,6)
    assert cfg["actor"]["hidden_dims"] == [256,128,64]
