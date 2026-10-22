"""Audit frozen Adam as well as frozen actor tensors after the native pilot."""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).parent)]
from export_robust_getup import load_mean
from iteration038_migration_audit import equal
from wheeled_algo.robust_getup import SensorConditionedMean


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(x) for x in value)
    return True


def main():
    torch.set_num_threads(1)
    run = ROOT / "runs/unified_robust_20261022/pilot023_wheel_velocity038"
    initial = torch.load(run / "model_initial.pt", map_location="cpu", weights_only=False)
    final = torch.load(run / "model_final.pt", map_location="cpu", weights_only=False)
    mean, info = load_mean(run / "model_final.pt")
    assert all(finite(final[k]) for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
    changed = [k for k, v in final["actor_state_dict"].items() if not equal(v, initial["actor_state_dict"][k])]
    assert set(changed) == {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    names = ["mlp." + k for k, p in SensorConditionedMean(mean).named_parameters() if p.requires_grad]
    old_group = next(g for g in initial["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
    new_group = next(g for g in final["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
    assert len(names) == len(old_group["params"]) == len(new_group["params"])
    frozen_slots, steps = 0, {}
    for name, old_id, new_id in zip(names, old_group["params"], new_group["params"], strict=True):
        before = initial["optimizer_state_dict"]["state"].get(old_id, {})
        after = final["optimizer_state_dict"]["state"].get(new_id, {})
        if name in changed:
            assert not before and after
            steps[name] = int(after["step"].item())
        else:
            assert equal(before, after), name
            frozen_slots += 1
    critic_changed = [k for k, v in final["critic_state_dict"].items() if not equal(v, initial["critic_state_dict"][k])]
    assert critic_changed and info["completed_updates"] == 473
    result = dict(schema="quiet038-final-state-audit-v1", all_model_and_adam_tensors_finite=True,
        frozen_actor_parameter_slots_with_exact_unchanged_adam=frozen_slots,
        only_added_velocity_weights_changed=changed, critic_changed=critic_changed,
        new_velocity_adam_minibatch_steps=steps, macro_ppo_updates=32, base_clock=473,
        robust_consumed=865, hardware_ready=False)
    (ROOT / "reports/unified_robust_20261021/iteration038_final_state.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
