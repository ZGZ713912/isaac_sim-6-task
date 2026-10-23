"""Zero-PPO position-observation lag test, retaining087 wheel filtering.

The150ms coordinate filter helped prior jitter;087 adds full wheel filtering
but fails the startup5cm bound. Test100ms instead of weakening that bound.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
REPORTS=Path(__file__).parent
RUNS=ROOT/"runs/unified_robust_20261022"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_runner import write
from wheeled_algo.standing_reference_lifecycle import observation_contract,observer_from_settings


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent=RUNS/"quiet_whole_wheel087/model_migrated.pt"
    assert sha(parent)=="bce79708a7bf1140a0051b9094056c1dcf32f7b7fc819015adabe09a0c3f9e91"
    assert json.loads((REPORTS/"iteration087_decision.json").read_text())["status"]=="rejected"
    old=torch.load(parent,map_location="cpu",weights_only=False);state=deepcopy(old)
    task=state["infos"]["settings"]
    assert task["standing_position_filter_seconds"]==.15 and task["quiet_output_wheel_alpha"]==.32
    task["standing_position_filter_seconds"]=.1
    task["standing_position_observation"]=observation_contract(.1,**task["standing_position_reference"])
    task["version"]="robust-v2.42-position100ms-wheel032"
    changed=[k for k in task if task[k]!=old["infos"]["settings"].get(k)]
    assert set(changed)=={"standing_position_filter_seconds","standing_position_observation","version"}
    observer=observer_from_settings(task,1,"cpu")
    assert observer.alpha==.005/(.1+.005) and observer.capture_delay_seconds==.05 and observer.reset_on_severe_tilt
    output=RUNS/"quiet_position_fast090";output.mkdir(exist_ok=False)
    revision=dict(schema="quiet090-zero-PPO-observation-lag-v1",parent_sha256=sha(parent),
        filter_before_seconds=.15,filter_after_seconds=.1,changed_settings=changed,
        reason="087 worst high lane reaches5.419cm at.63s while its position input lags startup roll",
        hypothesis="Reduce position-input lag while keeping constant wheel filtering and learned feedback intact",
        new_macro_updates=0,candidate_lineage=505,robust_consumed=1384,
        learned_tensor_changes=False,full_training_admitted=False,installed_in_rmcs=False)
    state["infos"].update(control_revision=revision,migrated_from_sha256=sha(parent),hardware_ready=False)
    shutil.copytree(parent.parent/"source",output/"source")
    paths=set(state["infos"]["source_sha256"])|{str(Path(__file__).relative_to(ROOT))}
    for relative in sorted(paths):
        target=output/"source"/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes());state["infos"]["source_sha256"][relative]=sha(target)
    contract=json.loads((parent.parent/"task_contract.json").read_text())
    contract.update(settings=task,control_revision=revision);write(output/"task_contract.json",contract)
    state["infos"]["task_contract_sha256"]=sha(output/"task_contract.json")
    checkpoint=output/"model_migrated.pt";torch.save(state,checkpoint)
    before,_=load_mean(parent);after,_=load_mean(checkpoint)
    for key in ["actor_state_dict","critic_state_dict","optimizer_state_dict"]:assert equal(old[key],state[key]),key
    for key in ["environment_rng","domain_rng","torch_rng","cuda_rng"]:assert equal(old["infos"][key],state["infos"][key]),key
    directory=RUNS/"constant_wheel_random087/robust"
    with np.load(directory/"trajectory.npz") as data:
        x=torch.from_numpy(data["observations"][::8].reshape(-1,35).copy())
    with torch.inference_mode():assert torch.equal(before(x),after(x))
    result=dict(revision,checkpoint_sha256=sha(checkpoint),strict_load_valid=True,
        actor_critic_Adam_fourRNG_exact=True,same_input_actions_exact=len(x),
        changed_observations_in_closed_loop_not_yet_tested=True,observation_contract=task["standing_position_observation"])
    write(REPORTS/"iteration090_migration.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
