"""Identification scope, signed dissipation, immutable assets and task admission."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch

from train_prone_motion import configuration
from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202, digest
from wheeled_tasks.chassis.full_vehicle_training import authenticate
from wheeled_tasks.chassis.leg_motor_friction import LegMotorFriction, declaration, load, JOINTS

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"model/dm_leg_friction_20261031"
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())


def component(count=1):
    return LegMotorFriction(count, "cpu", PLAN["leg_motor_friction"], root=ROOT)


def test_identified_plateau_is_output_torque_and_unknowns_stay_null():
    cfg, model = load(PLAN["leg_motor_friction"], ROOT)
    p = json.loads((MODEL/"parameters.json").read_text())
    assert cfg["enabled"] and model["joint_names"] == list(JOINTS)
    assert model["coulomb_nm"] == [2.019506006848738, 2.047246309400147]*2
    assert model["torque_units"] == "Nm at DM active output"
    assert model["source_parameter_sha256"] == digest(MODEL/"parameters.json")
    for name, row in p["motors"].items():
        assert row["static_breakaway_nm"] is None
        assert row["viscous_coefficient_nm_s_rad"] is None
        assert row["stribeck_velocity_rad_s"] is None
        assert row["low_speed_moving_friction_nm"] > row["coulomb_plateau_candidate_nm"]
        if name.startswith("right_"):
            assert row["symmetry_source"] == name.replace("right_", "left_")
    assert model["bias_applied"] is model["physical_static_model_enabled"] is False


@pytest.mark.parametrize("speed", [.2, .6, 1.5, 3.1417, 12.])
def test_both_actual_directions_are_dissipative_without_axis_sign_or_gear_multipliers(speed):
    friction = component(2)
    velocity = torch.tensor([[speed, -speed, -speed, speed], [-speed, speed, speed, -speed]])
    before = velocity.clone()
    effort = friction.effort(velocity)
    assert torch.equal(velocity, before)
    torch.testing.assert_close(effort.abs(), friction.coulomb.expand_as(effort), rtol=0., atol=0.)
    assert (effort*velocity < 0.).all()
    # Right-side model axes mirror left-side axes. Actual velocity already
    # supplies that sign; applying the API encoder reversal again would inject energy.
    torch.testing.assert_close(effort[0, :2], -effort[0, 2:], rtol=0., atol=0.)
    assert friction.native_effort_limit > 40.+float(effort.abs().max())-1e-6


def test_rest_has_no_stiction_or_bias_and_regularization_is_continuous():
    friction = component(3)
    assert torch.equal(friction.effort(torch.zeros(3, 4)), torch.zeros(3, 4))
    velocities = torch.tensor([[1e-7]*4, [.025]*4, [.05]*4])
    effort = friction.effort(velocities)
    assert effort[0].abs().max() < 5e-6
    torch.testing.assert_close(effort[1], effort[2]*.5)
    assert not friction.summary()["physical_static_model_enabled"]


def test_numerical_passive_only_decay_never_adds_energy():
    # This is an integration sanity check with a declared artificial inertia,
    # not a claim to have identified the installed leg's effective inertia.
    friction = component(2)
    velocity = torch.tensor([[.2, 1.5, -.6, -3.1], [-.2, -1.5, .6, 3.1]])
    for _ in range(200):
        before = velocity.clone()
        velocity = velocity+.001*friction.effort(velocity)/.05
        assert (velocity.abs() <= before.abs()+1e-8).all()
        assert (velocity*before >= 0.).all()


@pytest.mark.parametrize("field,value", [("bias_applied", True), ("controller_compensation", True),
    ("static_breakaway_nm", 2.3), ("viscous_coefficient_nm_s_rad", 0.), ("stribeck_velocity_rad_s", .2),
    ("coulomb_nm", [2.02, 2.05, 2.02, 2.05]), ("physical_static_model_enabled", True)])
def test_rehashed_unidentified_or_changed_physics_is_rejected(tmp_path, field, value):
    shutil.copytree(MODEL, tmp_path/"model")
    path = tmp_path/"model/model.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load(declaration(tmp_path, "model/model.json"), tmp_path)


def test_parameter_tampering_is_detected_even_when_model_is_unchanged(tmp_path):
    shutil.copytree(MODEL, tmp_path/"model")
    cfg = declaration(tmp_path, "model/model.json")
    with (tmp_path/"model/parameters.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="source changed"):
        load(cfg, tmp_path)


def test_default_asset_and_training_share_friction_and_cannot_silently_disable_it(tmp_path):
    asset = FullRobotV202()
    assert asset.leg_friction(1).enabled
    cfg, _, _ = configuration(tmp_path, PLAN, num_envs=80, device="cpu", performance_backend="eager")
    assert cfg["leg_motor_friction"] == asset.runtime["leg_motor_friction"] == PLAN["leg_motor_friction"]
    authenticate(cfg, ROOT)
    for altered in (None, {"enabled": False}):
        bad = deepcopy(cfg)
        if altered is None:
            bad.pop("leg_motor_friction")
        else:
            bad["leg_motor_friction"] = altered
        with pytest.raises(ValueError, match="leg moving friction"):
            authenticate(bad, ROOT)


def test_old_asset_and_live_run_dependencies_remain_byte_identical():
    target = ROOT/"model/wheel_leg_v2.0.2/friction_v1"
    migration = json.loads((target/"migration.json").read_text())
    for key in ("unchanged_parent_manifest", "unchanged_training_manifest", "previous_plan"):
        row = migration[key]
        assert digest(ROOT/row["file"]) == row["sha256"]
    old = FullRobotV202(ROOT/"model/wheel_leg_v2.0.2/isaac")
    assert not old.leg_friction(1).enabled
    old_plan = json.loads((target/"previous_plan.json").read_text())
    old_cfg = json.loads((ROOT/old_plan["mechanism"]["file"]).read_text())
    assert digest(ROOT/old_plan["mechanism"]["file"]) == old_plan["mechanism"]["sha256"]
    assert authenticate(old_cfg, ROOT)["leg_motor_friction"] is None


def test_component_does_not_consume_the_policy_rng():
    torch.manual_seed(403)
    expected = torch.rand(8)
    torch.manual_seed(403)
    friction = component()
    friction.effort(torch.ones(1, 4))
    torch.testing.assert_close(torch.rand(8), expected, rtol=0., atol=0.)
