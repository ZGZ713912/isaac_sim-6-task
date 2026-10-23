"""Audit actual native initial/final state of the 24-value position pilot."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT/"runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_native_state_audit import named_slots, finite
from wheeled_algo.standing_displacement import observation_contract


def main():
    torch.set_num_threads(1)
    parent_path = RUNS/"quiet_position056_sealed/model_migrated.pt"
    initial_path = RUNS/"pilot031_standing_position058/model_initial.pt"
    parent, initial = [torch.load(p,map_location="cpu",weights_only=False) for p in (parent_path,initial_path)]
    parent_mean,_ = load_mean(parent_path)
    initial_mean,_ = load_mean(initial_path)
    assert equal(parent["actor_state_dict"],initial["actor_state_dict"])
    assert equal(parent["critic_state_dict"],initial["critic_state_dict"])
    parent_slots,initial_slots = named_slots(parent,parent_mean),named_slots(initial,initial_mean)
    assert equal(parent_slots,initial_slots)
    for key in ("environment_rng","domain_rng","torch_rng","cuda_rng"):
        assert equal(parent["infos"][key],initial["infos"][key])
    cfg = initial["infos"]["settings"]
    assert cfg["quiet_velocity_feedback_mode"] == "standing_displacement"
    assert cfg["standing_position_observation"] == observation_contract()
    assert cfg["quiet_exploration_fraction"] == .06 and cfg["quiet_wheel_exploration_fraction"] == .35
    assert cfg["quiet_high_leg_prediction_s"] == .005 and not cfg.get("quiet_leg_prediction_all_heights",False)
    assert initial["infos"]["completed_updates"] == 441
    heads = {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid","low","high")}
    assert all(not initial_slots["actor"][name] for name in heads)
    progress = json.loads((initial_path.parent/"progress.json").read_text())
    assert progress["trainable_actor_parameters"] == 24
    result = dict(schema="position058-native-initial-v1",checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        initial_clock=441,new_trainable_values=24,parent_actor_critic_named_adam_rng_exact=True,
        declared_position_abi_exact=True, new_feedback_adam_empty=True, legacy_actor_and_std_frozen=True)
    (REPORTS/"iteration058_initial.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    final_path = initial_path.parent/"model_final.pt"
    if not final_path.exists():
        return
    assert progress["status"] == "completed" and progress["new_optimizer_updates"] == 32
    final = torch.load(final_path,map_location="cpu",weights_only=False)
    final_mean,info = load_mean(final_path)
    assert all(finite(final[k]) for k in ("actor_state_dict","critic_state_dict","optimizer_state_dict"))
    changed = {n for n,v in final["actor_state_dict"].items() if not equal(v,initial["actor_state_dict"][n])}
    assert changed == heads
    slots = named_slots(final,final_mean)
    old = set(initial_slots["actor"])-heads
    assert all(equal(initial_slots["actor"][n],slots["actor"][n]) for n in old)
    steps = {n:int(slots["actor"][n]["step"].item()) for n in heads}
    assert set(steps.values()) == {384}
    critic = [n for n,v in final["critic_state_dict"].items() if not equal(v,initial["critic_state_dict"][n])]
    assert len(critic) == 8 and info["completed_updates"] == 473
    assert equal(cfg,info["settings"])
    result = dict(schema="position058-native-final-v1",checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
        new_macro_updates=32,lineage_clock=473,robust_consumed=1128,only_actor_changes=sorted(changed),
        old_actor_std_named_adam_exact=True,feedback_adam_minibatch_steps=steps,critic_changed=critic,
        task_and_observation_abi_exact=True,finite=True,long_training_admitted=False,hardware_ready=False)
    (REPORTS/"iteration058_final_state.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
