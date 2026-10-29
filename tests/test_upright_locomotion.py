"""Task separation, dense control credit and retained acceptance/domain scope."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.upright_locomotion import failures, resolve, reward_terms, training_case

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "contracts/wheel_leg_upright_locomotion_v1.json").read_text())


@pytest.mark.parametrize("mode", ["balance", "basic_motion"])
def test_all_births_are_upright_and_tasks_never_age_into_recovery(mode):
    spec = resolve(SPEC, mode)
    cases = [training_case(spec, lane, 17) for lane in range(100)]
    assert all(not c["recovery"] and c["pose_index"] == 17 and c["command"] == [0., 0., .305] for c in cases)
    if mode == "balance":
        assert not any(c.get("sequence") for c in cases)
    else:
        assert Counter(c["role"] for c in cases) == dict(balance=40, forward=20, reverse=20, left=10, right=10)
        for c in cases:
            if c["role"] != "balance":
                assert c["sequence"][-1] == [8., [0., 0., .305]]


def test_fall_body_contact_and_height_failure_apply_even_while_moving():
    gravity = torch.tensor([[0., 0., -1.], [math.sin(.6), 0., -math.cos(.6)], [0., 0., -1.], [0., 0., -1.]])
    flags = failures(gravity, torch.tensor([.305, .305, .305, .2]), torch.full((4,), .305),
                     torch.tensor([0., 0., 2.1, 0.]), resolve(SPEC))
    assert flags["tilt"].tolist() == [False, True, False, False]
    assert flags["nonwheel_contact"].tolist() == [False, False, True, False]
    assert flags["height"].tolist() == [False, False, False, True]


def density(speed):
    velocity = torch.tensor([[speed, 0., 0.]])
    return sum(reward_terms(torch.tensor([[0., 0., -1.]]), torch.tensor([.305]), velocity,
        torch.zeros(1, 3), torch.tensor([[0., 0., .305]]), torch.zeros(1, 6),
        torch.zeros(1), torch.zeros(1)).values()).item()


def test_velocity_credit_does_not_vanish_outside_old_3cm_per_second_gaussian():
    assert density(0.) > density(.1) > density(.2) > density(.3)
    assert density(-.2) == density(.2)
    assert density(.2) - density(.3) > .2


@pytest.mark.parametrize("change", [dict(recovery_training=True), dict(automatic_task_or_height_curriculum=True)])
def test_cannot_silently_reintroduce_self_righting_or_clock_driven_curriculum(change):
    with pytest.raises(ValueError, match="independent"):
        resolve(dict(SPEC, **change))


def prepare(tmp_path, profile=None, friction="baseline", *, prone=False, static_prior=False):
    module_spec = importlib.util.spec_from_file_location("upright_base_train", ROOT / "scripts/train_scratch_standing.py")
    base = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(base)
    args = SimpleNamespace(output=tmp_path, bank=ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json",
        precise_objective=False, standing_failures=False, wheel_velocity_increment=not prone,
        static_load_initialization=static_prior, contact_break_cost=None, leg_target_rate_cost=None,
        all_heights_after=512, num_envs=64, rollout_steps=128, seed=271027,
        randomization_study=ROOT / "contracts/wheel_leg_randomization_study_v2.json" if profile else None,
        randomization_profile=profile, randomization_strength=1., normalize_observations=True,
        desync_episodes=False, upright_task=None if prone else ROOT / "contracts/wheel_leg_upright_locomotion_v1.json",
        standard_prone_task=ROOT / "contracts/wheel_leg_standard_prone_v1.json" if prone else None,
        upright_mode="balance", upright_friction_profile=friction)
    return base.prepare(args)


def test_prone_target_range_covers_rotation_without_changing_PD_or_motor_limits(tmp_path):
    original = json.loads((ROOT / "model/gas_spring_compensation_unified_20261029/mechanism.json").read_text())["v5_control"]
    cfg, task = prepare(tmp_path, "requested", prone=True)
    assert cfg["v5_control"] == dict(original, action_clip=16.)
    assert task["standard_prone"]["leg_action_clip"] == task["leg_action_clip"] == 16.
    resolved = json.loads((tmp_path / "task_contract.json").read_text())
    assert resolved["initial_pitch_range_deg"] == [0., 0.]
    assert resolved["wheel_action_parameterization"] == "learned_absolute_velocity_v1"


@pytest.mark.parametrize("friction,expected", [("baseline", [.5, .6]), ("legacy_low_grip", [.2, .3])])
def test_requested_domains_remain_full_and_friction_summary_matches_actual_material(tmp_path, friction, expected):
    cfg, task = prepare(tmp_path, "requested", friction)
    assert cfg["contact_material"]["ground_static_friction_range"] == [.3, .5]
    assert task["upright_locomotion"]["effective_wheel_ground_friction_range"] == expected
    assert task["randomization_study"]["effective_wheel_ground_friction_range"] == expected
    assert task["motor_strength_range"] == [.84, 1.16]
    assert cfg["dynamics_randomization"]["inertia_scale"] == [.94, 1.06]
    assert task["leg_kp_scale_range"] == task["leg_kd_scale_range"] == task["spring_strength_range"] == [1., 1.]
    assert task["imu_mount"]["wobble_degrees"] == [0., 0.]
    assert task["sensor_delay_policy_ticks"] == [0, 1]
    assert task["motion_push_fraction"] == 0. and task["reward_potential_gain"] == 0.
    assert task["upright_locomotion"]["recovery_training"] is False
    contract = json.loads((tmp_path / "task_contract.json").read_text())
    assert contract["training_origin"].startswith("seeded random actor and critic; empty Adam")
    assert contract["curriculum"]["automatic_changes"] is False
    assert contract["static_load_initialization"] is None
    assert cfg["v5_control"]["gas_spring_compensation"]["enabled"] is True


def test_uncompensated_static_load_prior_is_not_reused_with_gas_compensation(tmp_path):
    with pytest.raises(ValueError,match="Static prior mechanics differ"):
        prepare(tmp_path,static_prior=True)


def test_curriculum_covers_every_permanent_requirement_without_claiming_pending_stages_are_available():
    curriculum = json.loads((ROOT / "contracts/wheel_leg_separated_curriculum_v1.json").read_text())
    requirements = json.loads((ROOT / curriculum["requirements"]).read_text())
    expected = {c["id"] for c in requirements["capabilities"]}
    assert set(curriculum["final_capability_ids"]) == expected
    assert set().union(*(set(s["capabilities"]) for s in curriculum["stages"])) == expected
    assert {s["id"] for s in curriculum["stages"] if s["implementation"] == "available"} == {"P0", "U0", "U1"}
    assert curriculum["domain_sequence"]["legacy_low_grip_required_in_final_matrix"]
    assert curriculum["final_evidence"]["unmeasured_is_pass"] is False
