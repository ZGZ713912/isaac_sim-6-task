"""Enable the already-versioned severe-tilt reference lifecycle, without PPO."""
import argparse
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
from iteration036_compare import read
from wheeled_algo.standing_reference_lifecycle import observation_contract, reference_settings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--static-result", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    parent = args.parent.resolve()
    gate = json.loads(args.static_result.read_text())
    assert gate["checkpoint_sha256"] == sha(parent) and not gate["static_failures"]
    old = torch.load(parent, map_location="cpu", weights_only=False)
    before, _ = load_mean(parent)
    state = deepcopy(old)
    task = state["infos"]["settings"]
    assert task["standing_position_reference"] == reference_settings(.05, False)
    assert task["standing_position_filter_seconds"] == .15
    task["standing_position_reference"] = reference_settings(.05, True)
    task["standing_position_observation"] = observation_contract(.15, .05, True)
    task["version"] = "robust-v2.39-severe-standing-reference-reset"
    changed = [key for key in task if task[key] != old["infos"]["settings"].get(key)]
    assert set(changed) == {"version", "standing_position_reference", "standing_position_observation"}
    output = RUNS / "quiet_severe_reference065"
    revision = dict(schema="reference065-lifecycle-revision-v1", source_checkpoint=str(parent),
        source_sha256=sha(parent), changed_settings=changed, new_macro_updates=0,
        lineage_clock=state["infos"]["completed_updates"], robust_consumed=1160,
        long_training_admitted=False, installed_in_rmcs=False)
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent / "source", output / "source")
    paths = set(state["infos"]["source_sha256"]) | {
        "scripts/evaluate_robust_getup_batch.py", "scripts/train_robust_getup.py",
        str(Path(__file__).resolve().relative_to(ROOT))}
    for relative in sorted(paths):
        source = ROOT / relative
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        state["infos"]["source_sha256"][relative] = sha(source)
    contract = json.loads((parent.parent / "task_contract.json").read_text())
    contract.update(settings=task, control_revision=revision)
    (output / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    after, _ = load_mean(checkpoint)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert equal(state[key], old[key]), key
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(state["infos"][key], old["infos"][key]), key
    x = torch.from_numpy(np.concatenate([read(name)[3]["observations"][::20].reshape(-1, 35).copy()
                                        for name in ("joint_quiet_static064", "side_confirmation032")]))
    with torch.inference_mode():
        assert torch.equal(before(x), after(x))
    result = dict(revision, checkpoint_sha256=sha(checkpoint),
                  actor_critic_all_named_adam_rng_exact=True, actual_input_actions_bitwise_exact=len(x),
                  source_static_result=str(args.static_result), source_static_result_sha256=sha(args.static_result))
    (REPORTS / "iteration065_migration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
