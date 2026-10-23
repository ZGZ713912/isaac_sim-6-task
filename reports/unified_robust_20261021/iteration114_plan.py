"""Prepare one bounded high-standing expert adaptation after110, no native work."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
PARENT = RUNS / "quiet_position_gain110/model_migrated.pt"
OUTPUT = RUNS / "pilot034_high_startup114"
UPDATES, SEED = 32, 211728
PREFIXES = ("mlp.mean.quiet_high.", "mlp.mean.quiet_velocity.high.")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_arguments():
    import torch
    prior = torch.load(PARENT, map_location="cpu", weights_only=False)["infos"]["settings"]
    return [sys.executable, "scripts/train_robust_getup.py", "--output", str(OUTPUT), "--resume", str(PARENT),
        "--mode", "train", "--updates", str(UPDATES), "--num-envs", "128", "--rollout-steps", "128",
        "--seed", str(SEED), "--actor-lr", ".00001", "--domain-scale", ".35", "--friction-min", ".3",
        "--isolated-origins", "--quiet-only", "--high-standing-only", "--standing-support-training",
        "--early-standing-position", "--standing-drift-objective", "--standing-drift-speed-scale",
        str(prior["standing_drift_speed_scale_m_s"]), "--eval-seconds", "2"]


def expected_settings(output):
    import torch
    from train_robust_getup import prepare
    prior = torch.load(PARENT, map_location="cpu", weights_only=False)["infos"]["settings"]
    args = SimpleNamespace(output=output, resume=PARENT, bank=REPORTS / "pose_bank002.json",
        no_domains=False, curriculum_updates=128, curriculum_origin=0,
        quiet_leg_alpha=.25, quiet_wheel_alpha=.65, domain_scale=.35, recovery_only=False,
        num_envs=128, rollout_steps=128, mode="train", friction="high", friction_min=.3,
        seed=SEED, isolated_origins=True, quiet_only=True, high_standing_only=True,
        standing_support_training=True, early_standing_position=True, standing_drift_objective=True,
        standing_drift_speed_scale=prior["standing_drift_speed_scale_m_s"])
    return prepare(args)[1]


def make_plan():
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.standing_reference_lifecycle import validate_observation_contract
    torch.set_num_threads(1)
    assert sha(PARENT) == "c38a376b240c5e04a282e24a8357b9dfb68f092b2aa77b0552c11b69ad12bf2a"
    mean, info = load_mean(PARENT)
    with tempfile.TemporaryDirectory(prefix="standing114-plan-") as folder:
        task = expected_settings(Path(folder) / "prepare")
    assert validate_observation_contract(task)
    preserved = ("quiet_critic_anchor", "quiet_spectral_cost", "standing_support_training", "quiet_position_weight",
        "standing_early_position_weight", "standing_drift_weight", "standing_drift_speed_scale_m_s",
        "quiet_exploration_fraction", "quiet_wheel_exploration_fraction", "quiet_actor_alpha",
        "quiet_output_wheel_alpha", "quiet_high_leg_prediction_s", "quiet_leg_prediction_all_heights",
        "standing_position_observation", "standing_position_filter_seconds", "standing_position_reference")
    assert all(task.get(key) == info["settings"].get(key) for key in preserved)
    assert task["quiet_training"] and task["quiet_height_probabilities"] == [0., 0., 1.]
    assert task["quiet_output_wheel_alpha"] == .32 and task["standing_position_filter_seconds"] == .15
    trainable = {key: value for key, value in mean.named_parameters() if ("mlp.mean." + key).startswith(PREFIXES)}
    assert len(trainable) == 5 and sum(value.numel() for value in trainable.values()) == 2702
    return dict(schema="standing114-bounded-startup-plan-v1", status="prepared_not_started",
        parent_checkpoint=str(PARENT.relative_to(ROOT)), parent_sha256=sha(PARENT),
        hypothesis="After position gain reduced late creep, high startup remains5.535cm at~0.6s; let the existing high MLP learn state-conditioned transition compensation rather than increase constant bias or proportional gain",
        evidence="106/110 paired random batch:48alive and6 actualHF below036; low/mid pass, four high startup excursions remain; fixed gain improvement is diminishing",
        new_macro_updates_target=UPDATES, robust_charged_before=1384, robust_charged_if_complete=1416,
        selected_ppo_lineage=505, candidate_lineage_if_complete=537,
        new_offline_steps=0, actual_experiment_offline_steps=128,
        actor_lr=1e-5, critic_lr=1e-3, seed=SEED, num_envs=128, rollout_steps=128,
        trained=list(trainable), trainable_actor_values=2702,
        frozen="all parent/motion/recovery/low/mid/std/geometry and their named Adam slots",
        reward_and_domain_changed=False, expected_settings=task,
        declared_profile_changes={key: dict(previous=info["settings"].get(key), current=value)
                                  for key, value in task.items() if value != info["settings"].get(key)},
        preserved_standing_settings=list(preserved), command=training_arguments(),
        evaluation_order=[dict(name="high_startup_tuning114", seed=211425, seconds=12., baseline="position_gain_tuning110"),
                          dict(name="high_startup_original114", seed=211322, seconds=12., baseline="high_bias_static097"),
                          dict(name="high_startup_holdout114", seed=211626, seconds=12., baseline=None)],
        stop_after_static_failure=True, thresholds="all48alive; all full-window trueXY<=5cm;6 actualHF<=036; post.2s air<=3mm",
        retention="low/mid32 trajectories bitwise110 on tuning seed; middle16 exact097 on original;201066 same-input motion isolation before native evaluations",
        no_automatic_extra_ppo=True, full_training_admitted=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/train_robust_getup.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/robust_getup.py",
            ROOT / "src/wheeled_algo/quiet_output_actor.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py",
            Path(__file__))})


if __name__ == "__main__":
    from iteration045_runner import write
    plan = make_plan()
    path = REPORTS / "iteration114_plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        write(path, plan)
    print(json.dumps({key: plan[key] for key in ("status", "trainable_actor_values", "declared_profile_changes", "robust_charged_if_complete")}), flush=True)
