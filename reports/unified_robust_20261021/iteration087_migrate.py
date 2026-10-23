"""Zero-PPO alpha.32 migration selected from fresh same-batch086 evidence."""
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
    result = json.loads((REPORTS / "iteration086_final.json").read_text())
    assert result["status"] == "completed"
    choices = [v for v in result["variants"].values() if not v["basic_failures"]
        and v["target20_95_vs_same_batch_alpha1"]["mean"] >= .25]
    alpha = max(choices, key=lambda v:v["target20_95_vs_same_batch_alpha1"]["mean"])["alpha"]
    assert alpha == .32
    parent = RUNS / "quiet_severe_reference065/model_migrated.pt"
    assert sha(parent) == "12669ef1f470ce88acb6a9a17b907b31f2b34dd96396417f94079dbdcf813dba"
    old = torch.load(parent,map_location="cpu",weights_only=False)
    before,_ = load_mean(parent)
    state = deepcopy(old)
    task = state["infos"]["settings"]
    assert "quiet_output_wheel_alpha" not in task
    task["quiet_output_wheel_alpha"] = alpha
    task["version"] = "robust-v2.41-complete-quiet-wheel032"
    changed = [k for k in task if task[k] != old["infos"]["settings"].get(k)]
    assert set(changed) == {"quiet_output_wheel_alpha","version"}
    output = RUNS / "quiet_whole_wheel087"
    output.mkdir(exist_ok=False)
    revision = dict(schema="quiet087-zero-PPO-migration-v1",parent_sha256=sha(parent),alpha=alpha,
        changed_settings=changed,grid_result_sha256=sha(REPORTS/"iteration086_final.json"),
        new_macro_updates=0,candidate_lineage=505,robust_consumed=1384,
        learned_tensor_changes=False,full_training_admitted=False)
    state["infos"].update(control_revision=revision,migrated_from_sha256=sha(parent),hardware_ready=False)
    shutil.copytree(parent.parent/"source",output/"source")
    paths=set(state["infos"]["source_sha256"]) | {"src/wheeled_algo/quiet_output_actor.py",
        "scripts/export_robust_getup.py","scripts/evaluate_robust_getup_batch.py",
        "scripts/train_robust_getup.py",str(Path(__file__).resolve().relative_to(ROOT))}
    for relative in sorted(paths):
        target=output/"source"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        state["infos"]["source_sha256"][relative]=sha(target)
    contract=json.loads((parent.parent/"task_contract.json").read_text())
    contract.update(settings=task,control_revision=revision)
    write(output/"task_contract.json",contract)
    state["infos"]["task_contract_sha256"]=sha(output/"task_contract.json")
    checkpoint=output/"model_migrated.pt"
    torch.save(state,checkpoint)
    after,_=load_mean(checkpoint)
    for key in ("actor_state_dict","critic_state_dict","optimizer_state_dict"):
        assert equal(old[key],state[key]),key
    for key in ("environment_rng","domain_rng","torch_rng","cuda_rng"):
        assert equal(old["infos"][key],state["infos"][key]),key
    source=RUNS/"high_recenter_full066/robust/trajectory.npz"
    with np.load(source) as saved:
        x=torch.from_numpy(saved["observations"].reshape(-1,35)[::23].copy())
    with torch.inference_mode():
        original,actual=before(x),after(x)
        assert torch.equal(actual,QuietWheelOutputFilter(before,alpha)(x))
        outside=quiet_gate(x).ravel()==0.
        assert torch.equal(actual[:, :4],original[:, :4])
        assert torch.equal(actual[outside],original[outside])
    write(REPORTS/"iteration087_migration.json",dict(revision,checkpoint_sha256=sha(checkpoint),
        strict_load_valid=True,actor_critic_Adam_fourRNG_exact=True,recorded_inputs=len(x),
        diagnostic_wrapper_exact=True,nonquiet_actions_exact=int(outside.sum()),
        randomized_behavior_proven=False))
    print(sha(checkpoint),flush=True)


if __name__=="__main__":
    main()
