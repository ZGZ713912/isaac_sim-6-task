"""Prepare an unqualified composition; adoption requires independent behavior."""
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
from wheeled_algo.robust_getup import quiet_gate, tilt_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    recovery = RUNS / "offline_recovery096/model_migrated.pt"
    quiet = RUNS / "quiet_high_bias097/model_migrated.pt"
    assert sha(recovery) == "eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f"
    assert sha(quiet) == "10cca03e85ffa1c08b36d88830cf0d3b8fd6d08a1e95ad942fa17c5801e04cbf"
    assert json.loads((REPORTS / "iteration097_final.json").read_text())["static_pass"]
    prior = json.loads((REPORTS / "iteration096_final.json").read_text())
    assert prior["results"]["robust"]["cases"]["moving_push_left"]["practical"] == 4
    assert prior["results"]["robust"]["cases"]["inverted_asymmetric"]["practical"] == 4
    old = torch.load(recovery, map_location="cpu", weights_only=False)
    q = torch.load(quiet, map_location="cpu", weights_only=False)
    state = deepcopy(old)
    key = "mlp.mean.quiet_high.2.bias"
    state["actor_state_dict"][key] = q["actor_state_dict"][key].clone()
    changed = [k for k in old["actor_state_dict"] if not equal(old["actor_state_dict"][k], state["actor_state_dict"][k])]
    assert changed == [key]
    for name in ("critic_state_dict", "optimizer_state_dict"):
        assert equal(old[name], state[name])
    for name in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng", "offline_recovery_optimizer"):
        assert equal(old["infos"][name], state["infos"][name])
    task = state["infos"]["settings"]
    assert task["standing_position_filter_seconds"] == .15
    task.update(version="robust-v2.49-composed105", quiet_output_wheel_alpha=.32)
    revision = dict(schema="composed105-state-v1", recovery_parent_sha256=sha(recovery),
        quiet_parent_sha256=sha(quiet), change="097 high quiet bias and complete wheel alpha.32 on096 recovery",
        new_macro_updates=0, new_offline_steps=0, cumulative_offline_steps=64,
        robust_consumed=1384, candidate_ppo_lineage=569, candidate_adopted=False,
        full_training_admitted=False, hardware_ready=False,
        gates="100 quiet30s/fresh pass, full random140, nominal protocol003; disclose096 strong right failed; do not revive099")
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(recovery), hardware_ready=False)
    output = RUNS / "composed_getup105"
    output.mkdir(exist_ok=False)
    shutil.copytree(recovery.parent / "source", output / "source")
    sources = set(old["infos"]["source_sha256"]) | set(q["infos"]["source_sha256"]) | {str(Path(__file__).relative_to(ROOT))}
    for relative in sorted(sources):
        path = ROOT / relative
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        state["infos"]["source_sha256"][relative] = sha(target)
    contract = json.loads((recovery.parent / "task_contract.json").read_text())
    contract.update(settings=task, control_revision=revision)
    write(output / "task_contract.json", contract)
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    merged, _ = load_mean(checkpoint)
    quiet_mean, _ = load_mean(quiet)
    recovery_mean, _ = load_mean(recovery)
    source = RUNS / "high_bias_static097/robust/trajectory.npz"
    data = np.load(source)
    observations = torch.from_numpy(data["observations"].reshape(-1, 35).copy())
    with torch.inference_mode():
        gate = tilt_gate(observations, merged.stance_aware_recovery, merged.recovery_shape_aware)
        assert torch.count_nonzero(gate) == 0
        for x in observations.split(2048):
            assert torch.equal(merged(x), quiet_mean(x))
        source2 = RUNS / "dynamic_recovery_full079/robust/trajectory.npz"
        other = np.load(source2)
        x = torch.from_numpy(other["observations"].reshape(-1, 35)[::23].copy())
        nonquiet = quiet_gate(x).ravel() == 0.
        assert torch.equal(merged(x[nonquiet]), recovery_mean(x[nonquiet]))
    write(REPORTS / "iteration105_migration.json", dict(revision, checkpoint_sha256=sha(checkpoint),
        strict_load_valid=True, changed_actor_relative096=changed,
        frozen_actor_std_critic_PPOAdam_offlineAdam_fourRNG_exact=True,
        quiet097_recorded_inputs=len(observations), quiet097_outputs_exact=True,
        nonquiet096_recorded_inputs=int(nonquiet.sum()), nonquiet096_outputs_exact=True,
        source_static_trajectory_sha256=sha(source), source_dynamic_trajectory_sha256=sha(source2),
        native_composed_behavior_verified=False))
    print(json.dumps(dict(checkpoint_sha256=sha(checkpoint), quiet_frames=len(observations),
        nonquiet_frames=int(nonquiet.sum()), qualified=False)), flush=True)


if __name__ == "__main__":
    main()
