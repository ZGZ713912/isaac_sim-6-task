"""A local fresh run must retain the reward/plant and reject foreign lineage."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from wheeled_tasks.chassis import local_scratch
from wheeled_tasks.chassis.control_contract import load_chassis_control, validate_chassis_control
from wheeled_tasks.chassis.preflight import SubstepControlAudit, design_preflight
from wheeled_tasks.chassis.task import choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle():
    (ROOT / "runs").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="scratch-test-", dir=ROOT / "runs") as directory:
        target = Path(directory) / "config"
        file = local_scratch.write_bundle(ROOT, target, 512)
        yield json.loads(file.read_text())


def invocation():
    return SimpleNamespace(resume=None, transfer=None, consumed_updates=None,
        learning_rate_scale=None, task_actor_update=None, stage_actor_update=None,
        actor_updates_consumed=0, global_actor_update_offset=0, block_index=0,
        confirmed_jump_bins_json="[]", seed=290927, num_envs=512, stage="speed",
        updates=local_scratch.read_plan(ROOT)["update_ceiling"])


def test_latest_reward_and_native_mechanics_are_retained(bundle):
    latest = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v27.json").read_text())
    changed = {"design_preflight", "task_campaign", "integrated_production", "local_scratch",
        "target_num_envs", "ppo_minibatch_samples", "checkpoint_first_update",
        "budget_spec_file", "budget_spec_sha256", "nonstandard_start", "locomotion_difficulty",
        "ordinary_posture_curriculum", "policy_dt", "control_math_source", "control_math_sha256",
        "num_steps_per_env", "gamma", "lam", "action_difference_reference_dt", "initial_noise_std", "entropy_coef"}
    assert {k: v for k, v in bundle.items() if k not in changed} == {
        k: v for k, v in latest.items() if k not in changed}
    for field in ("nonstandard_start", "locomotion_difficulty", "ordinary_posture_curriculum"):
        expected = deepcopy(latest[field])
        expected["origin_actor_update"] = 0
        assert bundle[field] == expected
    assert "task_campaign" not in bundle and "integrated_production" not in bundle
    assert bundle["physics_dt"] == .005 and bundle["policy_dt"] == .005
    assert bundle["num_steps_per_env"] == 96
    assert bundle["checkpoint_first_update"] == 1
    local_scratch.validate_invocation(bundle, invocation(), ROOT)


@pytest.mark.parametrize("population", [256, 512, 1024])
def test_independent_design_and_all_scene_groups(population):
    # The table is adapted from the separately authenticated design, not
    # synthesized from a caller's changed effective contract.
    (ROOT / "runs").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="scratch-design-", dir=ROOT / "runs") as directory:
        file = local_scratch.write_bundle(ROOT, Path(directory) / "config", population)
        config = json.loads(file.read_text())
        control = load_chassis_control(ROOT / config["control_math_source"])
        receipt = design_preflight(config, control, ROOT, stage="speed", role="train",
            seed=290927, num_envs=population, block_index=0)
        rows = choose_scene_groups(config["scene_groups"], population)
        assert len(rows) == population
        assert len({name for name, terrain in rows}) == 135
        assert receipt["budget"]["inherited_consumption"] == 576
        assert receipt["budget"]["transition_ceiling"] == 19424 * population * 96
        assert receipt["runtime_expected"]["decimation"] == 1
        assert receipt["runtime_expected"]["policy_dt"] == .005
        assert config["initial_noise_std"] == .2
        assert config["entropy_coef"] == 0.


def test_entropy_comparison_preserves_the_previous_plant_reward_and_sampling_std():
    directory = ROOT / "runs/entropy-comparison-fixture"
    _, old = local_scratch.build_bundle(ROOT, directory, 512,
        plan_file=local_scratch.LOW_NOISE_PLAN_FILE)
    _, new = local_scratch.build_bundle(ROOT, directory, 512)
    changed = {key for key in old["worker"].keys() | new["worker"].keys()
        if old["worker"].get(key) != new["worker"].get(key)}
    assert changed == {"entropy_coef", "budget_spec_sha256", "design_preflight", "local_scratch"}
    assert "entropy_coef" not in old["worker"]
    assert old["worker"]["initial_noise_std"] == new["worker"]["initial_noise_std"] == .2
    assert old["budget"]["inherited_consumption"] == 64
    assert new["budget"]["inherited_consumption"] == 576
    for bundle in (old, new):
        assert bundle["budget"]["inherited_consumption"] + bundle["budget"]["update_ceiling"] == 20000


@pytest.mark.parametrize("field,value", [
    ("resume", Path("foreign.pt")), ("transfer", Path("foreign.pt")),
    ("consumed_updates", 46237), ("stage_actor_update", 4467),
    ("task_actor_update", 4467), ("actor_updates_consumed", 46057),
    ("seed", 617), ("num_envs", 4096), ("updates", 20001)])
def test_scratch_cannot_import_a_parent_or_foreign_budget(bundle, field, value):
    args = invocation()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="fresh learning state"):
        local_scratch.validate_invocation(bundle, args, ROOT)


@pytest.mark.parametrize("field,value", [
    ("physics_dt", .001), ("position_iterations", 12),
    ("policy_dt", .02), ("num_steps_per_env", 24), ("gamma", .99), ("lam", .95),
    ("action_difference_reference_dt", .005),
    ("termination_event_cost", 1.), ("target_num_envs", 64),
    ("entropy_coef", .005),
    ("dense_tracking", {"enabled": False}), ("task_campaign", {"domain": "speed"})])
def test_contract_changes_cannot_bypass_the_original_guards(bundle, field, value):
    bad = deepcopy(bundle)
    bad[field] = value
    with pytest.raises(ValueError, match="frozen native derivation"):
        local_scratch.validate_config(bad, ROOT)


def test_caller_cannot_derive_an_external_config_directory():
    with pytest.raises(ValueError, match="runs directory"):
        local_scratch.build_bundle(ROOT, "/tmp/foreign-scratch", 512)


def test_rollout_discount_and_gae_keep_the_same_physical_time(bundle):
    # One old decision interval is now four decisions. An impulse four steps
    # into the future must receive the same discount/GAE decay as before.
    assert bundle["num_steps_per_env"] * bundle["policy_dt"] == pytest.approx(24 * .02)
    assert bundle["gamma"]**4 == pytest.approx(.99)
    assert (bundle["gamma"] * bundle["lam"])**4 == pytest.approx(.99 * .95)
    audit = SubstepControlAudit(bundle["physics_dt"], bundle["policy_dt"])
    assert audit.decimation == 1


def test_legacy_50hz_plan_and_original_motor_prior_still_validate():
    with TemporaryDirectory(prefix="scratch-legacy-", dir=ROOT / "runs") as directory:
        file = local_scratch.write_bundle(ROOT, Path(directory) / "config", 512,
            plan_file=local_scratch.LEGACY_PLAN_FILE)
        config = json.loads(file.read_text())
        control = load_chassis_control(ROOT / config["control_math_source"])
        receipt = design_preflight(config, control, ROOT, stage="speed", role="train",
            seed=290927, num_envs=512, block_index=0)
        assert receipt["decimation"] == 4 and config["policy_dt"] == .02
        assert config["num_steps_per_env"] == 24 and "gamma" not in config
        local_scratch.validate_invocation(config, invocation(), ROOT)


def test_200hz_motor_control_requires_an_explicit_matching_scope(bundle):
    control = load_chassis_control(ROOT / bundle["control_math_source"])
    original = load_chassis_control(ROOT / "contracts/v6_wheel_response_control_200hz_v1.json")
    changed = {"control_id", "timing", "simulation_timing_approximation"}
    assert {k: v for k, v in control.items() if k not in changed} == {
        k: v for k, v in original.items() if k not in changed}
    original["timing"] = deepcopy(control["timing"])
    with pytest.raises(ValueError, match="declared"):
        validate_chassis_control(original)
    control["simulation_timing_approximation"]["fresh_feedback_per_policy_step"] = 4
    with pytest.raises(ValueError, match="Unknown"):
        validate_chassis_control(control)


@pytest.mark.parametrize("dt", [.02, .005])
def test_action_smoothness_has_the_same_density_for_a_physical_trajectory(dt):
    import torch
    from wheeled_tasks.chassis.rewards import reward_terms
    dtype = torch.float64
    xyz, motors = torch.zeros(1, 3, dtype=dtype), torch.zeros(1, 6, dtype=dtype)
    ones = torch.ones(1, dtype=dtype)
    gravity = xyz.clone(); gravity[:, 2] = -1.
    # Compare centered first differences on a ramp, and centered second
    # differences on a quadratic with known acceleration.
    common = (xyz, xyz, gravity, ones * .305, xyz, motors, motors, motors)
    tail = (torch.zeros(1, 2, 3, dtype=dtype), ones, ones, ones * 0.)
    ramp = reward_terms(*common, motors + dt, motors, motors - dt, *tail,
        action_difference_scale=.02 / dt)
    quadratic = reward_terms(*common, motors + dt**2, motors, motors + dt**2, *tail,
        action_difference_scale=.02 / dt)
    assert float(ramp["action_rate"][0]) == pytest.approx(-.01 * 6 * .02**2)
    assert float(quadratic["leg_action_smoothness"][0]) == pytest.approx(-.05 * 4 * (2 * .02**2)**2)
    assert float(quadratic["wheel_action_smoothness"][0]) == pytest.approx(-.01 * 2 * (2 * .02**2)**2)


def test_frozen_launch_uses_commit_contents_instead_of_later_workspace_edits(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    source = repository / "policy.py"
    source.write_text("version = 1\n")
    subprocess.run(["git", "add", "policy.py"], cwd=repository, check=True)
    subprocess.run(["git", "-c", "user.name=Local test", "-c", "user.email=local-test@example.invalid",
                    "commit", "-qm", "frozen source"], cwd=repository, check=True)
    source.write_text("version = 2\n")
    spec = importlib.util.spec_from_file_location("train_local_chassis_test", ROOT / "scripts/train_local_chassis.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    directory = tmp_path / "frozen"
    frozen = launcher.freeze_source(repository, directory)
    assert (frozen / "policy.py").read_text() == "version = 1\n"
    assert source.read_text() == "version = 2\n"
    receipt = json.loads((directory / "launch_source.json").read_text())
    assert receipt["commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    assert receipt["source_archive_sha256"] == hashlib.sha256((directory / "source.tar.gz").read_bytes()).hexdigest()
