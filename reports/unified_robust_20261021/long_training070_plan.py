"""Prepare the conditional long run without starting native simulation or PPO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_arguments(parent, output, updates, seed):
    return [sys.executable, "scripts/train_robust_getup.py", "--output", str(output),
        "--resume", str(parent), "--mode", "train", "--updates", str(updates),
        "--num-envs", "128", "--rollout-steps", "128", "--seed", str(seed),
        "--actor-lr", ".00001", "--domain-scale", ".35", "--friction-min", ".3",
        "--isolated-origins", "--recovery-only", "--eval-seconds", "2"]


def task_settings(parent, output, seed):
    from train_robust_getup import prepare
    args = SimpleNamespace(output=output, resume=parent,
        bank=REPORTS / "pose_bank002.json", no_domains=False, curriculum_updates=128,
        curriculum_origin=0, quiet_leg_alpha=.25, quiet_wheel_alpha=.65,
        domain_scale=.35, recovery_only=True, num_envs=128, rollout_steps=128,
        mode="train", friction="high", friction_min=.3, seed=seed,
        isolated_origins=True)
    return prepare(args)[1]


def make_plan():
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.standing_reference_lifecycle import validate_observation_contract
    parent = RUNS / "quiet_severe_reference065/model_migrated.pt"
    torch.set_num_threads(1)
    mean, info = load_mean(parent)
    with tempfile.TemporaryDirectory(prefix="recovery070-preflight-") as folder:
        task = task_settings(parent, Path(folder) / "preflight", 211431)
    assert validate_observation_contract(task)
    assert task["standing_position_reference"] == dict(capture_delay_seconds=.05, reset_on_severe_tilt=True)
    assert task["standing_position_filter_seconds"] == .15
    preserved = ("quiet_critic_anchor", "quiet_spectral_cost", "standing_support_training",
        "quiet_position_weight", "standing_early_position_weight", "standing_drift_weight",
        "quiet_actor_alpha", "quiet_exploration_fraction", "quiet_wheel_exploration_fraction",
        "quiet_high_leg_prediction_s", "quiet_leg_prediction_all_heights",
        "standing_position_observation", "standing_position_reference")
    assert all(task.get(key) == info["settings"].get(key) for key in preserved)
    assert task["quiet_training"] is False and task["motion_fraction"] == .4
    assert task["motion_push_fraction"] == 1. and not task["side_recovery_birth_focus"]
    changes = {key: dict(previous=info["settings"].get(key), current=value)
               for key, value in task.items() if value != info["settings"].get(key)}
    schedule = [64] + [256] * 19 + [192]
    assert sum(schedule) == 5120 and all(1 <= size <= 256 for size in schedule)
    expected_trainable = sum(p.numel() for p in mean.recovery.parameters())
    assert expected_trainable == 21894
    return dict(schema="recovery070-conditional-long-plan-v1", status="prepared_not_started",
        parent_checkpoint=str(parent.relative_to(ROOT)), parent_sha256=sha(parent),
        parent_lineage_clock=info["completed_updates"], robust_charged_before=1160,
        new_long_updates_target=5120, block_sizes=schedule,
        expected_final_lineage_clock=info["completed_updates"] + sum(schedule),
        expected_charged_if_all_completed=1160 + sum(schedule),
        admission_required="admission067_final.json: full_training_admitted=true and exact parent SHA",
        actor_scope="recovery head only; all quiet MLPs/position heads, motion/parent, std and geometry frozen",
        trainable_actor_values=expected_trainable, actor_lr=1e-5, critic_lr=1e-3,
        num_envs=128, rollout_steps=128, policy_hz=200, physics_hz=1000,
        domain_scale=.35, ground_friction=[.3, .5], seed_first_block=211431,
        native_origin_mode="isolated_overlap_v1", expected_settings=task,
        declared_profile_changes=changes, preserved_standing_settings=list(preserved),
        reset_exposure="40% upright rehearsal, 60% fallen; front/back/left/right/inverted/compound weights .1/.1/.2/.2/.3/.1",
        command_exposure="30% stand,20% straight,10% spin,40% arcs; existing high-speed samples retained",
        checkpoint_continuity="actor,critic,named Adam and all saved RNG; each block starts fresh native episodes",
        block_acceptance="48x12s randomized standing and140x12s full suite; frozen5cm/3mm/HF and motion/recovery gates",
        periodic_admission="Fresh two-seed nominal and30s randomized standing after each1024 new updates and final",
        interrupted_accounting="Charge all completed updates and any unsealed in-flight update conservatively; no refunds",
        long_training_admitted=False, new_macro_updates=0, hardware_ready=False,
        installed_in_rmcs=False, strong_push_boundary="Original80N/32Nm stationary push failed29/32 impact protection; retain receipt065")


if __name__ == "__main__":
    from iteration045_runner import write
    torch_env_plan = make_plan()
    destination = REPORTS / "long_training070_plan.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == torch_env_plan
    else:
        write(destination, torch_env_plan)
    print(json.dumps({key: deepcopy(torch_env_plan[key]) for key in (
        "status", "parent_sha256", "parent_lineage_clock", "new_long_updates_target",
        "trainable_actor_values", "declared_profile_changes", "long_training_admitted")}), flush=True)
