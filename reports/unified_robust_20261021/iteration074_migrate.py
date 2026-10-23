"""Explicit opt-in alpha.25 migration after the nominal073 grid, zero PPO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_runner import write
from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    result_path = REPORTS / "iteration073_final.json"
    result = json.loads(result_path.read_text())
    eligible = [row for row in result["variants"].values() if not row["basic_failures"]
                and row["target20_95_mean_improvement"] >= .25]
    alpha = max(row["alpha"] for row in eligible)
    assert alpha == .25 and result["status"] == "completed"
    parent = RUNS / "quiet_severe_reference065/model_migrated.pt"
    assert sha(parent) == "12669ef1f470ce88acb6a9a17b907b31f2b34dd96396417f94079dbdcf813dba"
    old = torch.load(parent, map_location="cpu", weights_only=False)
    before, _ = load_mean(parent)
    state = deepcopy(old)
    task = state["infos"]["settings"]
    assert "quiet_output_wheel_alpha" not in task
    task["quiet_output_wheel_alpha"] = alpha
    task["version"] = "robust-v2.40-complete-quiet-wheel-output-filter"
    changed = [key for key in task if task[key] != old["infos"]["settings"].get(key)]
    assert set(changed) == {"quiet_output_wheel_alpha", "version"}
    output = RUNS / "quiet_whole_wheel074"
    output.mkdir(exist_ok=False)
    revision = dict(schema="quiet074-complete-output-migration-v1", parent=str(parent), parent_sha256=sha(parent),
        changed_settings=changed, alpha=alpha, grid_result_sha256=sha(result_path),
        new_macro_updates=0, selected_lineage=505, robust_consumed=1160,
        learned_tensor_changes=False, full_training_admitted=False, installed_in_rmcs=False)
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    shutil.copytree(parent.parent / "source", output / "source")
    paths = set(state["infos"]["source_sha256"]) | {
        "src/wheeled_algo/quiet_output_actor.py", "scripts/export_robust_getup.py",
        "scripts/evaluate_robust_getup_batch.py", "scripts/train_robust_getup.py",
        str(Path(__file__).resolve().relative_to(ROOT))}
    for relative in sorted(paths):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        state["infos"]["source_sha256"][relative] = sha(target)
    contract = json.loads((parent.parent / "task_contract.json").read_text())
    contract.update(settings=task, control_revision=revision)
    write(output / "task_contract.json", contract)
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    after, _ = load_mean(checkpoint)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert equal(state[key], old[key]), key
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(state["infos"][key], old["infos"][key]), key
    with np.load(RUNS / "high_recenter_full066/robust/trajectory.npz") as data:
        x = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    with torch.inference_mode():
        original, actual = before(x), after(x)
        expected = QuietWheelOutputFilter(before, alpha)(x)
        nonquiet = quiet_gate(x).ravel() == 0.
        assert torch.equal(actual, expected)
        assert torch.equal(actual[:, :4], original[:, :4])
        assert torch.equal(actual[nonquiet], original[nonquiet])
    receipt = dict(revision, checkpoint_sha256=sha(checkpoint), actor_critic_named_adam_rng_exact=True,
        strict_load_valid=True, actual_input_frames=len(x), nonquiet_actions_exact=int(nonquiet.sum()),
        legs_at_same_input_exact=True, diagnostic_wrapper_actions_exact=True,
        randomized_behavior_proven=False)
    write(REPORTS / "iteration074_migration.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
