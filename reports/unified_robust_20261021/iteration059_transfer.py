"""Zero-PPO test of transferring only the learned metric position feedback."""
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
RUNS = ROOT/"runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_native_state_audit import named_slots
from iteration036_compare import read
from wheeled_algo.robust_getup import SensorConditionedMean


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent=RUNS/"pilot031_standing_position058/model_final.pt"
    output=RUNS/"quiet_low_transfer059"
    assert sha(parent)=="3745d2668286bd33363a70e660e72ec54ffa15bcf5e75ebac3d036dd2beccb3d"
    old=torch.load(parent,map_location="cpu",weights_only=False)
    state=deepcopy(old)
    before,info=load_mean(parent)
    names=["mlp."+n for n,p in SensorConditionedMean(before).named_parameters() if p.requires_grad]
    group=next(g for g in state["optimizer_state_dict"]["param_groups"] if g["name"]=="actor")
    assert len(names)==len(group["params"])
    identifiers=dict(zip(names,group["params"],strict=True))
    source="mlp.mean.quiet_velocity.mid.weight"
    target="mlp.mean.quiet_velocity.low.weight"
    state["actor_state_dict"][target]=state["actor_state_dict"][source].clone()
    state["optimizer_state_dict"]["state"][identifiers[target]]=deepcopy(state["optimizer_state_dict"]["state"][identifiers[source]])
    changed=[n for n,v in state["actor_state_dict"].items() if not equal(v,old["actor_state_dict"][n])]
    assert changed==[target]
    assert equal(state["critic_state_dict"],old["critic_state_dict"])
    for key in ("environment_rng","domain_rng","torch_rng","cuda_rng"):
        assert equal(state["infos"][key],old["infos"][key])
    revision=dict(schema="position059-mid-to-low-v1",source_checkpoint=str(parent),source_sha256=sha(parent),
        copied_actor=source,target_actor=target,parameters_copied=8,optimizer="only low named Adam copied from mid; all others retained",
        new_macro_updates=0,lineage_clock=473,robust_consumed=1128,long_training_admitted=False)
    state["infos"]["settings"]["version"]="robust-v2.34-low-position-transfer"
    state["infos"]["control_revision"]=revision
    state["infos"]["migrated_from_sha256"]=sha(parent)
    state["infos"]["hardware_ready"]=False
    output.mkdir(exist_ok=False)
    shutil.copytree(parent.parent/"source",output/"source")
    relative=Path(__file__).resolve().relative_to(ROOT)
    archived=output/"source"/relative
    archived.parent.mkdir(parents=True,exist_ok=True)
    archived.write_bytes(Path(__file__).read_bytes())
    state["infos"]["source_sha256"][str(relative)]=sha(Path(__file__))
    contract=json.loads((parent.parent/"task_contract.json").read_text())
    contract["settings"]=state["infos"]["settings"]
    contract["control_revision"]=revision
    (output/"task_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    state["infos"]["task_contract_sha256"]=sha(output/"task_contract.json")
    checkpoint=output/"model_migrated.pt"
    torch.save(state,checkpoint)
    after,_=load_mean(checkpoint)
    a,b=named_slots(old,before),named_slots(state,after)
    assert equal(b["actor"][target],a["actor"][source])
    assert all(equal(v,b["actor"][n]) for n,v in a["actor"].items() if n!=target)
    assert equal(a["critic"],b["critic"])
    evidence=read("learned_position_static058")
    x=torch.from_numpy(evidence[3]["observations"][::20].reshape(-1,35).copy())
    nonlow=x[:,3] >= torch.tensor(.305)*5.
    with torch.inference_mode():
        assert torch.equal(before(x)[nonlow],after(x)[nonlow])
        zero=x.clone();zero[:,14:16]=0.
        assert torch.equal(before(zero),after(zero))
        difference=float((before(x)-after(x)).abs().max())
    assert difference>0.
    result=dict(revision,checkpoint_sha256=sha(checkpoint),only_low_position_weight_changed=True,
        low_quiet_MLP_and_mid_high_weights_exact=True,old_actor_std_critic_other_named_adam_rng_exact=True,
        real_static_inputs=len(x),nonlow_actions_bitwise_exact=int(nonlow.sum()),zero_position_actions_exact=len(x),
        maximum_low_raw_action_difference=difference,installed_in_rmcs=False)
    (REPORTS/"iteration059_migration.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
