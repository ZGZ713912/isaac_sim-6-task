"""Compare the frozen009 proposal with007 before committing or starting Isaac."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.scratch_standing import agent_config


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_ast(path):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == "evaluate"), include_attributes=False)


prepared = ROOT / "runs/scratch_standing_20261026/preflight009_leg_velocity_separated003"
baseline = ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero"
manifest = read(prepared / "source_hashes.json")
original = read(baseline / "source_hashes.json")
current_task, original_task = [read(p / "task_contract.json") for p in (prepared, baseline)]
config, original_config = [read(p / "config/contract.json") for p in (prepared, baseline)]
current_core, original_core = [dict(d) for d in (config, original_config)]
current_core.pop("design_preflight")
original_core.pop("design_preflight")
settings, old_settings = [dict(d["settings"]) for d in (current_task, original_task)]
settings.pop("version")
old_settings.pop("version")
agent = agent_config(261001, "cuda:0", 128, normalize_observations=True,
                     wheel_velocity_increment=True, separate_leg_velocity_feedback=True)
old_agent = read(baseline / "agent_config.json")
actor_name = agent["actor"]["class_name"]
agent["actor"]["class_name"] = old_agent["actor"]["class_name"]
common_changes = sorted(name for name in manifest.keys() & original.keys() if manifest[name] != original[name])
additions = sorted(manifest.keys() - original.keys())
base = ROOT / "contracts/scratch_standing_base_mechanism_v1.json"
base_copy = read(base)
base_copy["design_preflight"]["file"] = read(ROOT / "runs/flat_base_20261020/pilot034_directional200_1346/config/contract.json")["design_preflight"]["file"]
checks = dict(
    all264_snapshot_sources_authentic=len(manifest) == 264 and all(
        sha(prepared / "source_snapshot" / name) == expected == sha(ROOT / name) for name, expected in manifest.items()),
    original_fixed_evaluation_AST_unchanged=evaluate_ast(ROOT / "scripts/train_scratch_standing.py") == evaluate_ast(baseline / "source_snapshot/scripts/train_scratch_standing.py"),
    original_mechanism_core_exact=current_core == original_core,
    settings_differ_only_by_architecture_version=settings == old_settings,
    original_static_mechanical_prior_exact=current_task["static_load_initialization"] == original_task["static_load_initialization"],
    original_reset_bank_exact=current_task["bank_sha256"] == original_task["bank_sha256"] == sha(prepared / "pose_bank.json"),
    original_agent_configuration_except_actor_class_exact=agent == old_agent,
    intended_separated_actor_class=actor_name == "wheeled_algo.leg_velocity_separated:LegVelocitySeparatedModel",
    contact_event_cost_zero=current_task["standing_contact_continuity_contract"] == original_task["standing_contact_continuity_contract"],
    rejected_target_cost_not_enabled=current_task["standing_leg_target_rate_contract"] is None,
    existing_source_changes_only_declared_files=common_changes == ["scripts/train_scratch_standing.py", "src/wheeled_algo/scratch_standing.py", "src/wheeled_tasks/chassis/scratch_standing_env.py"],
    environment_change_only_removes_blank_EOF=(ROOT / "src/wheeled_tasks/chassis/scratch_standing_env.py").read_bytes().rstrip(b"\n") == (baseline / "source_snapshot/src/wheeled_tasks/chassis/scratch_standing_env.py").read_bytes().rstrip(b"\n"),
    declared_new_actor_and_unused_target_cost_sources_only=additions == ["src/wheeled_algo/leg_velocity_separated.py", "src/wheeled_tasks/chassis/standing_leg_target_rate.py"],
    mechanism_promotion_only_changes_design_path=base_copy == read(ROOT / "runs/flat_base_20261020/pilot034_directional200_1346/config/contract.json"),
    promoted_design_byte_exact=sha(ROOT / "contracts/scratch_standing_base_design_v1.json") == "e5161d0cedc6f607e20d64819979e1a689c42493f1f9858ef99f2a5c50cb30d3",
    mid_height_exposure_matches007=current_task["curriculum"] == original_task["curriculum"],
    same_policy_and_physics_rate=current_task["policy_hz"] == 200 and current_task["physics_hz"] == 1000,
    no_runtime_filter_or_target_hold=current_task["leg_velocity_feedback_architecture"]["action_filter"] is False,
    reset_only_state_writes=current_task["state_writes"] == "reset only",
    diagnostic_closed_and_unqualified=read(Path(__file__).with_name("scratch_leg_encoder007_terminal_review001.json"))["original_process_exited"] is True,
    no_training_or_model_created=not (prepared / "model_initial.pt").exists() and not (prepared / "progress.json").exists(),
)
result = dict(schema="scratch-separated-leg-velocity-preflight-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    prepared=str(prepared), baseline=str(baseline), checks=checks, all_checks_pass=all(checks.values()),
    completed_CPU_tests=67, architecture_reward_integrity_tests=63, committed_source_guard_tests=4,
    test_files=["tests/test_leg_velocity_separated.py", "tests/test_wheel_velocity_increment.py", "tests/test_scratch_standing.py",
        "tests/test_standing_load_initialization.py", "tests/test_standing_failure_termination.py", "tests/test_precise_standing_reward.py",
        "tests/test_standing_contact_objective.py", "tests/test_standing_contact_continuity.py", "tests/test_standing_geometry107.py",
        "tests/test_scratch_integrity_seal.py", "tests/test_scratch_committed_source.py"],
    common_source_changes=common_changes, source_additions=additions, source_hashes=manifest,
    source_count=len(manifest), source_commit_verified=False, native_validation=False, new_PPO=0,
    requested_fresh_updates=512, qualified=False, hardware_ready=False, reader_sha256=sha(__file__))
output = Path(__file__).with_name("scratch_separated009_preflight002.json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps(dict(all_checks_pass=result["all_checks_pass"], checks=checks), ensure_ascii=False))
assert result["all_checks_pass"]
