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
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
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
        confirmed_jump_bins_json="[]", seed=290927, num_envs=512, stage="speed", updates=20000)


def test_latest_reward_and_native_mechanics_are_retained(bundle):
    latest = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v27.json").read_text())
    changed = {"design_preflight", "task_campaign", "integrated_production", "local_scratch",
        "target_num_envs", "ppo_minibatch_samples", "checkpoint_first_update",
        "budget_spec_file", "budget_spec_sha256", "nonstandard_start", "locomotion_difficulty",
        "ordinary_posture_curriculum"}
    assert {k: v for k, v in bundle.items() if k not in changed} == {
        k: v for k, v in latest.items() if k not in changed}
    for field in ("nonstandard_start", "locomotion_difficulty", "ordinary_posture_curriculum"):
        expected = deepcopy(latest[field])
        expected["origin_actor_update"] = 0
        assert bundle[field] == expected
    assert "task_campaign" not in bundle and "integrated_production" not in bundle
    assert bundle["physics_dt"] == .005 and bundle["policy_dt"] == .02
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
        assert receipt["budget"]["inherited_consumption"] == 0
        assert receipt["budget"]["transition_ceiling"] == 20000 * population * 24


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
    ("termination_event_cost", 1.), ("target_num_envs", 64),
    ("dense_tracking", {"enabled": False}), ("task_campaign", {"domain": "speed"})])
def test_contract_changes_cannot_bypass_the_original_guards(bundle, field, value):
    bad = deepcopy(bundle)
    bad[field] = value
    with pytest.raises(ValueError, match="frozen native derivation"):
        local_scratch.validate_config(bad, ROOT)


def test_caller_cannot_derive_an_external_config_directory():
    with pytest.raises(ValueError, match="runs directory"):
        local_scratch.build_bundle(ROOT, "/tmp/foreign-scratch", 512)


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
