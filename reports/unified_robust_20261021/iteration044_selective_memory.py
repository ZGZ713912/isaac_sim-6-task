"""Retain only040's high-height learned feedback; no new optimization."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import SensorConditionedMean, quiet_gate


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    torch.set_num_threads(1)
    base_path = RUNS / "quiet_wheel_memory040/model_migrated.pt"
    high_path = RUNS / "pilot025_wheel_memory040/model_final.pt"
    output = RUNS / "quiet_selective_memory044"
    if output.exists():
        raise ValueError("Refuse to overwrite selective candidate")
    base, high = [torch.load(p, map_location="cpu", weights_only=False) for p in (base_path, high_path)]
    base_mean, _ = load_mean(base_path, migration_source_root=base_path.parent / "source")
    high_mean, _ = load_mean(high_path, migration_source_root=high_path.parent / "source")
    changed = [k for k, v in base["actor_state_dict"].items() if not equal(v, high["actor_state_dict"][k])]
    expected = ["mlp.mean.quiet_velocity." + name + ".weight" for name in ("mid", "low", "high")]
    assert set(changed) == set(expected)
    selected = deepcopy(base)
    key = "mlp.mean.quiet_velocity.high.weight"
    selected["actor_state_dict"][key] = high["actor_state_dict"][key].clone()
    maps = []
    for model, state in ((base_mean, base), (high_mean, high)):
        names = [n for n, p in SensorConditionedMean(model).named_parameters() if p.requires_grad]
        group = next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"] == "actor")
        assert len(names) == len(group["params"])
        maps.append(dict(zip(names, group["params"], strict=True)))
    target_id, source_id = [m[key.removeprefix("mlp.")] for m in maps]
    selected["optimizer_state_dict"]["state"][target_id] = deepcopy(high["optimizer_state_dict"]["state"][source_id])
    lineage = dict(source_path=str(high_path), source_sha256=hashlib.sha256(high_path.read_bytes()).hexdigest(),
        source_clock=high["infos"]["completed_updates"], retained_base_clock=base["infos"]["completed_updates"],
        selected_actor_parameter=key, selected_adam_slot=target_id, optimizer_updates=0,
        scope="Only high-height feedback weight and its Adam moments selected; clocks are not added; candidate not qualified")
    selected["infos"]["partial_head_selection"] = lineage
    assert equal(base["critic_state_dict"], selected["critic_state_dict"])
    assert equal(base["optimizer_state_dict"]["param_groups"], selected["optimizer_state_dict"]["param_groups"])
    for identifier, state in base["optimizer_state_dict"]["state"].items():
        if identifier != target_id:
            assert equal(state, selected["optimizer_state_dict"]["state"][identifier])
    assert equal(base["infos"], {k: v for k, v in selected["infos"].items() if k != "partial_head_selection"})
    output.mkdir()
    checkpoint = output / "model_selected.pt"
    torch.save(selected, checkpoint)
    for relative, expected in selected["infos"]["source_sha256"].items():
        archived = base_path.parent / "source" / relative
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archived.read_bytes())
    task = base_path.parent / "task_contract.json"
    assert hashlib.sha256(task.read_bytes()).hexdigest() == selected["infos"]["task_contract_sha256"]
    (output / "task_contract.json").write_bytes(task.read_bytes())
    mean, _ = load_mean(checkpoint, migration_source_root=output / "source")
    observations = []
    for name in ("quiet_velocity_baseline036", "side_confirmation032", "wheel_memory_static040"):
        data = np.load(RUNS / name / "robust/trajectory.npz")
        observations.append(torch.from_numpy(data["observations"][::24].reshape(-1, 35).copy()))
    x = torch.cat(observations)
    with torch.inference_mode():
        actual, old, trained = mean(x), base_mean(x), high_mean(x)
        anchors = x.new_tensor([.23, .305, .42]) * 5.
        low_mid = (x[:, 3] == anchors[0]) | (x[:, 3] == anchors[1])
        high_mask = x[:, 3] == anchors[2]
        outside = quiet_gate(x).squeeze(-1) == 0.
        assert torch.equal(actual[low_mid], old[low_mid])
        assert torch.equal(actual[high_mask], trained[high_mask])
        assert torch.equal(actual[outside], old[outside])
    result = dict(schema="quiet044-selective-memory-v1", **lineage,
        base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        observations=len(x), low_mid_exact_samples=int(low_mid.sum()), high_matches040_samples=int(high_mask.sum()),
        outside_quiet_exact_samples=int(outside.sum()), all_other_actor_critic_adam_rng_exact=True,
        robust_consumed=929, long_training_admitted=False, native_behavior_verified=False)
    (REPORTS / "iteration044_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
