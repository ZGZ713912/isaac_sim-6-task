"""Zero-PPO high quiet bias recentering at an analytic sensor-only anchor."""
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
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = RUNS / "pilot032_joint_quiet064/model_final.pt"
    reference = RUNS / "quiet_position_reference062/model_migrated.pt"
    output = RUNS / "quiet_high_recenter066"
    assert sha(parent) == "cb892fe99a96794e5eccee3ca33a71a69e9940eef179c33564ad158cec20d7ff"
    assert sha(reference) == "fa4bdd207ee47b3672e597d010c2c9be86cc8cfd1a2291471a0c3dcb927d0f48"
    before, _ = load_mean(parent)
    old_mean, _ = load_mean(reference)
    anchor = torch.zeros(1, 35)
    anchor[:, 3], anchor[:, 9] = .42 * 5., -1.
    features = torch.tanh(anchor * before.feature_scale)
    with torch.inference_mode():
        shift = before.quiet_high(features) - old_mean.quiet_high(features)
    old = torch.load(parent, map_location="cpu", weights_only=False)
    state = deepcopy(old)
    target = "mlp.mean.quiet_high.2.bias"
    state["actor_state_dict"][target].sub_(shift[0])
    changed = [key for key, value in state["actor_state_dict"].items()
               if not equal(value, old["actor_state_dict"][key])]
    assert changed == [target]
    for key in ("critic_state_dict", "optimizer_state_dict"):
        assert equal(state[key], old[key]), key
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(state["infos"][key], old["infos"][key]), key
    revision = dict(schema="quiet066-analytic-high-bias-recenter-v1", source_checkpoint=str(parent),
        source_sha256=sha(parent), reference_sha256=sha(reference), changed_actor=changed,
        changed_values=6, canonical_sensor_observation=anchor[0].tolist(),
        subtracted_high_logits=shift[0].tolist(),
        rationale="Retain the learned high quiet weights/Jacobian, remove its DC logit shift at the nominal sensor anchor",
        optimizer="All named Adam states unchanged; explicit zero-PPO parameter revision",
        new_macro_updates=0, lineage_clock=505, robust_consumed=1160,
        long_training_updates=0, long_training_admitted=False, installed_in_rmcs=False)
    task = state["infos"]["settings"]
    task["version"] = "robust-v2.38-high-quiet-analytic-recenter"
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent / "source", output / "source")
    relative = Path(__file__).resolve().relative_to(ROOT)
    archived = output / "source" / relative
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(Path(__file__).read_bytes())
    state["infos"]["source_sha256"][str(relative)] = sha(Path(__file__))
    contract = json.loads((parent.parent / "task_contract.json").read_text())
    contract.update(settings=task, control_revision=revision)
    (output / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    after, _ = load_mean(checkpoint)
    with torch.inference_mode():
        anchor_error = float((after.quiet_high(features) - old_mean.quiet_high(features)).abs().max())
        assert anchor_error < 1e-7
    names = ("position_reference_static062", "joint_quiet_static064", "side_confirmation032")
    x = torch.from_numpy(np.concatenate([read(name)[3]["observations"][::20].reshape(-1, 35).copy()
                                        for name in names]))
    nonhigh = x[:, 3] <= torch.tensor(.305) * 5.
    nonquiet = quiet_gate(x).ravel() == 0
    with torch.inference_mode():
        a, b = before(x), after(x)
        assert torch.equal(a[nonhigh], b[nonhigh])
        assert torch.equal(a[nonquiet], b[nonquiet])
    result = dict(revision, checkpoint_sha256=sha(checkpoint), strict_load_passed=True,
        actor_change_restricted_to_six_high_output_bias_values=True,
        high_dynamic_weights_position_heads_other_actor_std_critic_all_adam_rng_exact=True,
        canonical_high_logit_error=anchor_error, actual_inputs=len(x),
        low_mid_actions_bitwise_exact=int(nonhigh.sum()), nonquiet_actions_bitwise_exact=int(nonquiet.sum()),
        max_raw_action_change=float((a - b).abs().max()),
        native_plan=dict(replicas_per_height=16, seconds=12, seed=211322, domain_scale=.35,
                         ground_friction=[.3, .5], policy_hz=200, physics_hz=1000,
                         origin="isolated_overlap_v1", default_static_gates_unchanged=True))
    (REPORTS / "iteration066_migration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
