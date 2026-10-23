"""Actual initial/final state and frozen-motion audit of joint quiet adaptation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_native_state_audit import named_slots, finite
from iteration036_compare import read
from wheeled_algo.robust_getup import quiet_gate
from wheeled_algo.standing_reference_lifecycle import validate_observation_contract

PREFIXES = tuple("mlp.mean." + key for key in ("quiet.", "quiet_low.", "quiet_high.", "quiet_velocity."))


def main():
    torch.set_num_threads(1)
    parent_path = RUNS / "quiet_position_reference062/model_migrated.pt"
    initial_path = RUNS / "pilot032_joint_quiet064/model_initial.pt"
    parent, initial = [torch.load(path, map_location="cpu", weights_only=False) for path in (parent_path, initial_path)]
    before, _ = load_mean(parent_path)
    mean, info = load_mean(initial_path)
    for key in ("actor_state_dict", "critic_state_dict"):
        assert equal(parent[key], initial[key])
    old_slots, slots = named_slots(parent, before), named_slots(initial, mean)
    assert equal(old_slots, slots)
    assert all(equal(parent["infos"][key], info[key]) for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"))
    cfg = info["settings"]
    assert validate_observation_contract(cfg)
    assert cfg["standing_position_reference"] == dict(capture_delay_seconds=.05, reset_on_severe_tilt=False)
    assert cfg["standing_position_filter_seconds"] == .15
    assert cfg["quiet_exploration_fraction"] == .06 and cfg["quiet_wheel_exploration_fraction"] == .35
    assert info["quiet_only_tuning"] and not info["quiet_velocity_only_tuning"]
    assert info["completed_updates"] == 473
    progress = json.loads((initial_path.parent / "progress.json").read_text())
    expected_count = sum(p.numel() for name, p in mean.named_parameters()
                         if ("mlp.mean." + name).startswith(PREFIXES))
    if "trainable_actor_parameters" in progress:
        assert progress["trainable_actor_parameters"] == expected_count
    rates = {group["name"]: group["lr"] for group in initial["optimizer_state_dict"]["param_groups"]}
    assert rates == dict(actor=1e-5, critic=1e-3)
    result = dict(schema="joint064-initial-v1", checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                  initial_clock=473, expected_trainable_actor_parameters=expected_count,
                  native_progress_trainable_count=progress.get("trainable_actor_parameters"),
                  parent_actor_critic_all_named_adam_rng_exact=True, actor_lr=1e-5, critic_lr=1e-3,
                  quiet_MLPs_and_position_heads_trainable=True, motion_recovery_std_frozen=True,
                  updates_seen_at_initial_audit=progress["new_optimizer_updates"])
    initial_receipt = REPORTS / "iteration064_initial.json"
    if not initial_receipt.exists():
        initial_receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    final_path = initial_path.parent / "model_final.pt"
    if not final_path.exists():
        return
    assert progress["status"] == "completed" and progress["new_optimizer_updates"] == 32
    assert progress["trainable_actor_parameters"] == expected_count
    assert progress["optimizer_learning_rates"] == rates
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    after, final_info = load_mean(final_path)
    assert all(finite(final[key]) for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
    changed = [key for key, value in final["actor_state_dict"].items() if not equal(value, initial["actor_state_dict"][key])]
    assert changed and all(key.startswith(PREFIXES) for key in changed)
    final_slots = named_slots(final, after)
    frozen = [key for key in slots["actor"] if not key.startswith(PREFIXES)]
    assert all(equal(slots["actor"][key], final_slots["actor"][key]) for key in frozen)
    steps = {}
    for key in slots["actor"]:
        if key.startswith(PREFIXES):
            start = int(slots["actor"][key].get("step", torch.tensor(0.)).item())
            steps[key] = int(final_slots["actor"][key]["step"].item()) - start
    assert set(steps.values()) == {384}
    assert equal(cfg, final_info["settings"]) and final_info["completed_updates"] == 505
    critic_changed = [key for key, value in final["critic_state_dict"].items()
                      if not equal(value, initial["critic_state_dict"][key])]
    assert len(critic_changed) == 8
    x = torch.from_numpy(np.concatenate([read(name)[3]["observations"][::20].reshape(-1, 35).copy()
        for name in ("position_reference_static062", "side_confirmation032")]))
    inactive = quiet_gate(x).ravel() == 0
    with torch.inference_mode():
        assert torch.equal(before(x)[inactive], after(x)[inactive])
    result = dict(schema="joint064-final-v1", checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
                  new_macro_updates=32, lineage_clock=505, robust_consumed=1160, only_actor_changes=changed,
                  frozen_actor_std_and_named_adam_exact=True, allowed_adam_minibatch_step_deltas=steps,
                  critic_changed=critic_changed, declared_observation_abi_exact=True,
                  actual_nonquiet_inputs_bitwise_exact=int(inactive.sum()), finite=True,
                  native_trainable_actor_parameters=expected_count,
                  long_training_admitted=False, hardware_ready=False)
    (REPORTS / "iteration064_final_state.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
