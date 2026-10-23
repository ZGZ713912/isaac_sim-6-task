"""Verify actual native initial/final model and Adam ownership by parameter name."""
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
    parent_path = RUNS / "quiet_rolling050/model_migrated.pt"
    initial_path = RUNS / "pilot027_rolling_feedback050/model_initial.pt"
    parent, initial = [torch.load(p, map_location="cpu", weights_only=False) for p in (parent_path, initial_path)]
    parent_mean, _ = load_mean(parent_path)
    initial_mean, _ = load_mean(initial_path)
    assert equal(parent["actor_state_dict"], initial["actor_state_dict"])
    assert equal(parent["critic_state_dict"], initial["critic_state_dict"])
    parent_slots, initial_slots = named_slots(parent, parent_mean), named_slots(initial, initial_mean)
    assert equal(parent_slots, initial_slots)
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(parent["infos"][key], initial["infos"][key])
    heads = {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    old_names = set(initial_slots["actor"]) - heads
    assert all(not initial_slots["actor"][name] for name in heads)
    progress = json.loads((initial_path.parent / "progress.json").read_text())
    assert progress["trainable_actor_parameters"] == 12
    output = dict(schema="rolling050-native-initial-audit-v1", initial_clock=441,
        parent_actor_critic_named_adam_rng_exact=True, new_empty_adam_parameter_slots=3,
        trainable_actor_values=12, old_actor_parameter_slots=len(old_names),
        initial_checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest())
    (REPORTS / "iteration050_initial.json").write_text(json.dumps(output, indent=2) + "\n")
    final_path = initial_path.parent / "model_final.pt"
    if final_path.exists():
        assert progress["status"] == "completed" and progress["new_optimizer_updates"] == 64
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        final_mean, info = load_mean(final_path)
        assert all(finite(final[key]) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
        changed = {name for name,value in final["actor_state_dict"].items() if not equal(value, initial["actor_state_dict"][name])}
        assert changed == heads
        final_slots = named_slots(final, final_mean)
        assert all(equal(initial_slots["actor"][name], final_slots["actor"][name]) for name in old_names)
        steps = {name:int(final_slots["actor"][name]["step"].item()) for name in heads}
        assert set(steps.values()) == {768}
        critic_changed = [name for name,value in final["critic_state_dict"].items() if not equal(value, initial["critic_state_dict"][name])]
        assert len(critic_changed) == 8 and info["completed_updates"] == 505
        result = dict(schema="rolling050-native-final-audit-v1", checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
            new_macro_updates=64, lineage_clock=505, robust_consumed=1025,
            only_new_actor_weights_changed=sorted(changed), frozen_geometry_and_old_actor_std_named_adam_exact=True,
            new_adam_minibatch_steps=steps, critic_changed=critic_changed, finite=True, hardware_ready=False)
        (REPORTS / "iteration050_final_state.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
