"""Bounded high quiet-network bias proposal; only native behavior can accept it."""
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
from wheeled_algo.robust_getup import quiet_gate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    assert json.loads((REPORTS/"iteration093_decision.json").read_text())["status"]=="rejected"
    parent=RUNS/"quiet_whole_wheel087/model_migrated.pt"
    assert sha(parent)=="bce79708a7bf1140a0051b9094056c1dcf32f7b7fc819015adabe09a0c3f9e91"
    old=torch.load(parent,map_location="cpu",weights_only=False);state=deepcopy(old)
    before,_=load_mean(parent);key="mlp.mean.quiet_high.2.bias"
    shift=torch.tensor([.004,.004,-.004,-.004,0.,0.])
    state["actor_state_dict"][key]+=shift
    changed=[k for k,v in state["actor_state_dict"].items() if not equal(v,old["actor_state_dict"][k])]
    assert changed==[key] and int((state["actor_state_dict"][key]!=old["actor_state_dict"][key]).sum())==4
    task=state["infos"]["settings"]
    assert task["standing_position_filter_seconds"]==.15 and task["quiet_output_wheel_alpha"]==.32
    task["version"]="robust-v2.44-high-quiet-bias004"
    output=RUNS/"quiet_high_bias095";output.mkdir(exist_ok=False)
    revision=dict(schema="quiet095-bounded-network-bias-v1",parent_sha256=sha(parent),
        changed_actor=changed,changed_values=4,added_high_output_logits=shift.tolist(),
        reason="093 .002 shift improved all16 high lanes and worst5.419->5.069cm; same startup peak remains",
        hypothesis="Small forward common-angle DC change in existing high quiet MLP reduces startup excursion",
        parameter_step_basis="Double093 single-coordinate step to .004 total logits relative087; predicted target change below.401mrad; one bounded extension only",
        motor_commands="Existing neural actor only; no new controller, observer, gate or scripted angle",
        optimizer="All named Adam moments kept; explicit zero-PPO parameter proposal",
        new_macro_updates=0,candidate_lineage=505,robust_consumed=1384,
        full_training_admitted=False,installed_in_rmcs=False)
    state["infos"].update(control_revision=revision,migrated_from_sha256=sha(parent),hardware_ready=False)
    shutil.copytree(parent.parent/"source",output/"source")
    for relative in sorted(set(state["infos"]["source_sha256"])|{str(Path(__file__).relative_to(ROOT))}):
        target=output/"source"/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes());state["infos"]["source_sha256"][relative]=sha(target)
    contract=json.loads((parent.parent/"task_contract.json").read_text())
    contract.update(settings=task,control_revision=revision);write(output/"task_contract.json",contract)
    state["infos"]["task_contract_sha256"]=sha(output/"task_contract.json")
    checkpoint=output/"model_migrated.pt";torch.save(state,checkpoint);after,_=load_mean(checkpoint)
    for name in ["critic_state_dict","optimizer_state_dict"]:assert equal(old[name],state[name]),name
    for name in ["environment_rng","domain_rng","torch_rng","cuda_rng"]:assert equal(old["infos"][name],state["infos"][name]),name
    directory=RUNS/"high_recenter_full066/robust"
    with np.load(directory/"trajectory.npz") as data:
        x=torch.from_numpy(data["observations"].reshape(-1,35)[::23].copy())
    with torch.inference_mode():
        a,b=before(x),after(x)
        low_mid=x[:,3]<=x.new_tensor(.305)*5.;outside=quiet_gate(x).ravel()==0
        assert torch.equal(a[low_mid],b[low_mid]) and torch.equal(a[outside],b[outside])
        assert torch.equal(a[:,4:],b[:,4:])
        physical_max=float((((b[:,:4]-a[:,:4])*.25).abs()).max())
        assert physical_max<=.000401
    write(REPORTS/"iteration095_migration.json",dict(revision,checkpoint_sha256=sha(checkpoint),
        strict_load_valid=True,other_actor_std_critic_Adam_fourRNG_exact=True,
        recorded_inputs=len(x),low_mid_actions_exact=int(low_mid.sum()),nonquiet_actions_exact=int(outside.sum()),
        same_input_wheel_outputs_exact=True,maximum_same_input_leg_target_change_rad=physical_max,
        native_behavior_proven=False))
    print(sha(checkpoint),flush=True)


if __name__=="__main__":main()
