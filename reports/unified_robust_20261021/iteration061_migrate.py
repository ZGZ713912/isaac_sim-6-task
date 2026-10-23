"""Explicit zero-PPO 50-to-150 ms position-observation revision."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration036_compare import read
from wheeled_algo.standing_displacement_contract import observation_contract, observer_from_settings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = RUNS / "quiet_high_transfer060/model_migrated.pt"
    output = RUNS / "quiet_position_filter061"
    assert sha(parent) == "5e36081c49aba88de1ad53d39a6f97208ae1870761cbbe730edf3eae806e5a48"
    old = torch.load(parent, map_location="cpu", weights_only=False)
    before, _ = load_mean(parent)
    state = deepcopy(old)
    task = state["infos"]["settings"]
    task["version"] = "robust-v2.36-position-filter150ms"
    task["standing_position_filter_seconds"] = .15
    task["standing_position_observation"] = observation_contract(.15)
    changed_settings = [key for key in task if task[key] != old["infos"]["settings"].get(key)]
    assert set(changed_settings) == {"version", "standing_position_filter_seconds", "standing_position_observation"}
    observer = observer_from_settings(task, 1, "cpu")
    assert observer.alpha == .005 / (.15 + .005)
    revision = dict(schema="position061-filter-revision-v1", source_checkpoint=str(parent),
                    source_sha256=sha(parent), changed_settings=changed_settings,
                    filter_before_seconds=.05, filter_after_seconds=.15,
                    new_macro_updates=0, lineage_clock=473, robust_consumed=1128,
                    long_training_admitted=False)
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent / "source", output / "source")
    paths = set(state["infos"]["source_sha256"]) | {
        "src/wheeled_algo/standing_displacement_contract.py", "scripts/export_robust_getup.py",
        "scripts/evaluate_robust_getup_batch.py", str(Path(__file__).resolve().relative_to(ROOT))}
    for relative in sorted(paths):
        source = ROOT / relative
        assert source.is_file(), relative
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
        for name in ("high_transfer_static060", "low_transfer_static059", "side_confirmation032")]))
    with torch.inference_mode():
        assert torch.equal(before(x), after(x))
    rejected = []
    with tempfile.TemporaryDirectory(prefix="position061-contract-") as folder:
        for change in ("missing_setting", "wrong_contract", "wrong_type"):
            bad = deepcopy(state)
            settings = bad["infos"]["settings"]
            if change == "missing_setting":
                del settings["standing_position_filter_seconds"]
            elif change == "wrong_contract":
                settings["standing_position_observation"]["filter_seconds"] = .05
            else:
                settings["standing_position_filter_seconds"] = "0.15"
            path = Path(folder) / (change + ".pt")
            torch.save(bad, path)
            try:
                load_mean(path)
            except ValueError:
                rejected.append(change)
            else:
                raise AssertionError("Malformed PT accepted: " + change)
    result = dict(revision, checkpoint_sha256=sha(checkpoint), actor_critic_all_adam_rng_exact=True,
                  actual_input_actions_bitwise_exact=len(x), actual_bad_checkpoints_rejected=rejected,
                  old_frozen_geometry_and_estimator_source_unchanged=True, installed_in_rmcs=False)
    (REPORTS / "iteration061_migration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
