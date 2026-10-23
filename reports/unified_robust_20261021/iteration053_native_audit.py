"""Check resumed prediction scope and native ownership for bounded adaptation053."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_native_state_audit import named_slots, finite


def main():
    torch.set_num_threads(1)
    parent_path = RUNS / "quiet_all_prediction052/model_migrated.pt"
    initial_path = RUNS / "pilot028_all_prediction053/model_initial.pt"
    parent, initial = [torch.load(p, map_location="cpu", weights_only=False) for p in (parent_path, initial_path)]
    parent_mean, _ = load_mean(parent_path)
    initial_mean, _ = load_mean(initial_path)
    assert equal(parent["actor_state_dict"], initial["actor_state_dict"])
    assert equal(parent["critic_state_dict"], initial["critic_state_dict"])
    parent_slots, initial_slots = named_slots(parent, parent_mean), named_slots(initial, initial_mean)
    assert equal(parent_slots, initial_slots)
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(parent["infos"][key], initial["infos"][key])
    settings = initial["infos"]["settings"]
    assert settings["quiet_leg_prediction_all_heights"] is True
    assert settings["quiet_high_leg_prediction_s"] == .005
    assert initial_mean.quiet_leg_prediction_all_heights is True
    assert initial_mean.quiet_high_leg_prediction_s == .005
    assert settings["quiet_velocity_feedback_mode"] == "rolling_velocity"
    # No task, randomization, cost, or model changes in this continuation.
    parent_settings = parent["infos"]["settings"]
    version_labels = {"parent": parent_settings["version"], "native_initial": settings["version"]}
    assert version_labels == {"parent": "robust-v2.31-all-height-leg-prediction",
                              "native_initial": "robust-v2.15-standing-drift"}
    assert equal({k: v for k, v in parent_settings.items() if k != "version"},
                 {k: v for k, v in settings.items() if k != "version"})
    assert initial["infos"]["completed_updates"] == 505
    heads = {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    old_names = set(initial_slots["actor"]) - heads
    assert {int(initial_slots["actor"][name]["step"].item()) for name in heads} == {768}
    progress = json.loads((initial_path.parent / "progress.json").read_text())
    assert progress["trainable_actor_parameters"] == 12
    result = dict(schema="prediction053-native-initial-audit-v1", initial_clock=505,
        parent_actor_critic_named_adam_rng_and_semantic_settings_exact=True,
        only_settings_metadata_difference=version_labels,
        all_height_prediction=True, prediction_seconds=.005, trainable_actor_values=12,
        old_actor_parameter_slots=len(old_names), initial_feedback_adam_step=768,
        checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest())
    (REPORTS / "iteration053_initial.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    final_path = initial_path.parent / "model_final.pt"
    if not final_path.exists():
        return
    assert progress["status"] == "completed" and progress["new_optimizer_updates"] == 32
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    final_mean, info = load_mean(final_path)
    assert all(finite(final[key]) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
    changed = {name for name, value in final["actor_state_dict"].items()
               if not equal(value, initial["actor_state_dict"][name])}
    assert changed == heads
    final_slots = named_slots(final, final_mean)
    assert all(equal(initial_slots["actor"][name], final_slots["actor"][name]) for name in old_names)
    steps = {name: int(final_slots["actor"][name]["step"].item()) for name in heads}
    assert set(steps.values()) == {1152}
    critic_changed = [name for name, value in final["critic_state_dict"].items()
                      if not equal(value, initial["critic_state_dict"][name])]
    assert len(critic_changed) == 8 and info["completed_updates"] == 537
    assert equal(settings, info["settings"])
    result = dict(schema="prediction053-native-final-audit-v1",
        checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
        new_macro_updates=32, lineage_clock=537, robust_consumed=1057,
        only_feedback_actor_weights_changed=sorted(changed),
        frozen_geometry_and_old_actor_std_named_adam_exact=True,
        feedback_adam_minibatch_steps=steps, critic_changed=critic_changed,
        prediction_scope_and_task_settings_exact=True, finite=True, hardware_ready=False)
    (REPORTS / "iteration053_final_state.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
