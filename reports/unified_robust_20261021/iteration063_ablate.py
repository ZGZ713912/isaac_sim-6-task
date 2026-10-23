"""Zero-PPO isolation of direct leg-position feedback; old RL balancing retained."""
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
from iteration036_compare import read
from iteration044_selective_memory import equal
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent = RUNS / "quiet_position_reference062/model_migrated.pt"
    output = RUNS / "quiet_position_wheels063"
    assert sha(parent) == "fa4bdd207ee47b3672e597d010c2c9be86cc8cfd1a2291471a0c3dcb927d0f48"
    old = torch.load(parent, map_location="cpu", weights_only=False)
    before, _ = load_mean(parent)
    state = deepcopy(old)
    names = [f"mlp.mean.quiet_velocity.{height}.weight" for height in ("low", "mid", "high")]
    for name in names:
        weight = state["actor_state_dict"][name]
        assert weight.shape == (4, 2) and weight[:2].count_nonzero() == 4
        weight[:2].zero_()
        assert torch.equal(weight[2:], old["actor_state_dict"][name][2:])
    changed = [key for key, value in state["actor_state_dict"].items() if not equal(value, old["actor_state_dict"][key])]
    assert set(changed) == set(names)
    assert all(equal(v, state["actor_state_dict"][k]) for k, v in old["actor_state_dict"].items() if k not in names)
    for key in ("critic_state_dict", "optimizer_state_dict"):
        assert equal(state[key], old[key])
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(state["infos"][key], old["infos"][key])
    revision = dict(schema="position063-zero-direct-leg-feedback-v1", source_checkpoint=str(parent), source_sha256=sha(parent),
                    changed_actor=names, scalar_values_zeroed=12, optimizer="all Adam retained; no optimizer step",
                    new_macro_updates=0, lineage_clock=473, robust_consumed=1128, long_training_admitted=False)
    state["infos"]["settings"]["version"] = "robust-v2.38-position-wheel-ablation"
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(parent), hardware_ready=False)
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent / "source", output / "source")
    relative = Path(__file__).resolve().relative_to(ROOT)
    archived = output / "source" / relative
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(Path(__file__).read_bytes())
    state["infos"]["source_sha256"][str(relative)] = sha(Path(__file__))
    contract = json.loads((parent.parent / "task_contract.json").read_text())
    contract.update(settings=state["infos"]["settings"], control_revision=revision)
    (output / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    after, _ = load_mean(checkpoint)
    x = torch.from_numpy(np.concatenate([read(name)[3]["observations"][::20].reshape(-1, 35).copy()
        for name in ("position_reference_static062", "position_filter_static061", "side_confirmation032")]))
    with torch.inference_mode():
        a, b = before(x), after(x)
        assert torch.equal(a[:, 4:6], b[:, 4:6])
        inactive = quiet_gate(x).ravel() == 0
        assert torch.equal(a[inactive], b[inactive])
        zero = x.clone(); zero[:, 14:16] = 0
        assert torch.equal(before(zero), after(zero))
        difference = float((a - b).abs().max())
        assert difference > 0
    result = dict(revision, checkpoint_sha256=sha(checkpoint), old_actor_critic_all_adam_rng_exact=True,
                  all_wheel_rows_and_fixed_input_wheel_actions_exact=True, actual_inputs=len(x),
                  nonquiet_actions_bitwise_exact=int(inactive.sum()), zero_position_actions_exact=len(x),
                  maximum_leg_raw_action_difference=difference, installed_in_rmcs=False,
                  scope="Direct position-to-leg rows only. Existing RL leg/quiet/recovery outputs remain active.")
    (REPORTS / "iteration063_migration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
