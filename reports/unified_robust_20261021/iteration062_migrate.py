"""Explicit zero-PPO .2-to-.05 s position-reference capture revision."""
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
from wheeled_algo.standing_reference_lifecycle import observation_contract, observer_from_settings, reference_settings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = RUNS / "quiet_position_filter061/model_migrated.pt"
    output = RUNS / "quiet_position_reference062"
    assert sha(parent) == "2c4903502a9cb5d974ad708f45a5fb79c7be8a95507d76c7528a67c7b6372251"
    old = torch.load(parent, map_location="cpu", weights_only=False)
    before, _ = load_mean(parent)
    state = deepcopy(old)
    task = state["infos"]["settings"]
    task["version"] = "robust-v2.37-position-reference50ms"
    task["standing_position_filter_seconds"] = .15
    task["standing_position_reference"] = reference_settings(.05, False)
    task["standing_position_observation"] = observation_contract(.15, .05, False)
    changed_settings = [key for key in task if task[key] != old["infos"]["settings"].get(key)]
    assert set(changed_settings) == {"version", "standing_position_reference", "standing_position_observation"}
    observer = observer_from_settings(task, 1, "cpu")
    assert observer.alpha == .005 / (.15 + .005) and observer.capture_delay_seconds == .05
    assert observer.reset_on_severe_tilt is False
    revision = dict(schema="position062-reference-revision-v1", source_checkpoint=str(parent),
                    source_sha256=sha(parent), changed_settings=changed_settings,
                    filter_before_seconds=.15, filter_after_seconds=.15,
                    capture_before_seconds=.2, capture_after_seconds=.05, severe_tilt_reset=False,
                    new_macro_updates=0, lineage_clock=473, robust_consumed=1128,
                    long_training_admitted=False)
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent / "source", output / "source")
    paths = set(state["infos"]["source_sha256"]) | {
        "src/wheeled_algo/standing_reference_lifecycle.py", "scripts/export_robust_getup.py",
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
        for name in ("position_filter_static061", "low_transfer_static059", "side_confirmation032")]))
    with torch.inference_mode():
        assert torch.equal(before(x), after(x))
    rejected = []
    with tempfile.TemporaryDirectory(prefix="position062-contract-") as folder:
        for change in ("missing_reference", "wrong_contract", "wrong_type"):
            bad = deepcopy(state)
            settings = bad["infos"]["settings"]
            if change == "missing_reference":
                del settings["standing_position_reference"]
            elif change == "wrong_contract":
                settings["standing_position_observation"]["capture_delay_seconds"] = .2
            else:
                settings["standing_position_reference"]["capture_delay_seconds"] = "0.05"
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
    (REPORTS / "iteration062_migration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
