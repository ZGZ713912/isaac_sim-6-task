"""One bounded high-position-feedback step after the measured106 improvement."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
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
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    assert json.loads((REPORTS / "iteration106_pipeline.json").read_text())["status"] == "completed"
    parent = RUNS / "quiet_position_gain106/model_migrated.pt"
    anchor = RUNS / "quiet_high_bias097/model_migrated.pt"
    assert sha(parent) == "a513167a51a3a485ab56fbe99f2731359ee972cceef30daaa789a33be2aad837"
    assert sha(anchor) == "10cca03e85ffa1c08b36d88830cf0d3b8fd6d08a1e95ad942fa17c5801e04cbf"
    original = torch.load(parent, map_location="cpu", weights_only=False)
    state = deepcopy(original)
    key = "mlp.mean.quiet_velocity.high.weight"
    base = torch.load(anchor, map_location="cpu", weights_only=False)
    state["actor_state_dict"][key] = base["actor_state_dict"][key] * 2.
    changed = [name for name, value in state["actor_state_dict"].items()
               if not equal(value, original["actor_state_dict"][name])]
    assert changed == [key] and int((state["actor_state_dict"][key] != original["actor_state_dict"][key]).sum()) == 8
    settings = state["infos"]["settings"]
    settings["version"] = "robust-v2.51-high-position-gain110"
    revision = dict(schema="quiet110-position-gain-v1", parent_sha256=sha(parent), anchor097_sha256=sha(anchor),
        changed_actor=changed, changed_values=8, position_gain_relative097=dict(low=1.5, mid=1., high=2.),
        reason="106 low drift5.442->3.852cm, high6.910->5.623cm with all6HF below036; only high still fails",
        early_late_evidence="High early max5.725->5.623cm; late max6.910->4.179cm; this step may still fail startup and must not be adopted on tail-only metrics",
        hypothesis="A further bounded high position-feedback step improves its transient while preserving low/mid and ordinary motion",
        unchanged="DC.003; quiet wheel alpha.32; observer150ms; original action/physics/protection contract",
        new_macro_updates=0, new_offline_steps=0, candidate_lineage=505, robust_consumed=1384,
        full_training_admitted=False, installed_in_rmcs=False)
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    output = RUNS / "quiet_position_gain110"
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent / "source", output / "source")
    for relative in sorted(set(state["infos"]["source_sha256"]) | {str(Path(__file__).relative_to(ROOT))}):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        state["infos"]["source_sha256"][relative] = sha(target)
    contract = json.loads((parent.parent / "task_contract.json").read_text())
    contract.update(settings=settings, control_revision=revision)
    write(output / "task_contract.json", contract)
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    before, _ = load_mean(parent)
    after, _ = load_mean(checkpoint)
    for name in ("critic_state_dict", "optimizer_state_dict"):
        assert equal(original[name], state[name])
    for name in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(original["infos"][name], state["infos"][name])
    with np.load(RUNS / "high_recenter_full066/robust/trajectory.npz") as data:
        x = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    with torch.inference_mode():
        a, b = before(x), after(x)
        low_mid = x[:, 3] <= x.new_tensor(.305) * 5.
        nonquiet = quiet_gate(x).ravel() == 0.
        assert torch.equal(a[low_mid], b[low_mid]) and torch.equal(a[nonquiet], b[nonquiet])
        zero = x.clone(); zero[:, 14:16] = 0.
        assert torch.equal(before(zero), after(zero))
        peak = float(((b[:, :4] - a[:, :4]) * .25).abs().max())
        assert peak <= .02
    write(REPORTS / "iteration110_migration.json", dict(revision, checkpoint_sha256=sha(checkpoint),
        strict_load_valid=True, other_actor_std_critic_Adam_fourRNG_exact=True,
        recorded_inputs=len(x), low_mid_actions_exact=int(low_mid.sum()), nonquiet_actions_exact=int(nonquiet.sum()),
        zero_displacement_actions_exact=True, maximum_same_input_leg_target_change_rad=peak,
        native_behavior_proven=False))
    print(sha(checkpoint), flush=True)


if __name__ == "__main__":
    main()
