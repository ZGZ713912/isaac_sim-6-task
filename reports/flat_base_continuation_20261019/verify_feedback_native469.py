"""Read native milestone files; no optimizer, environment, or process signals."""
from pathlib import Path
import json
import torch
import numpy as np
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
torch.set_num_threads(2)
paths = {"parent":RUN / "pilot010/model_000448.pt", "initial":RUN / "pilot011/initial_state.pt",
         "first_critic":RUN / "pilot011/model_000449.pt", "warmup_end":RUN / "pilot011/model_000468.pt",
         "first_actor":RUN / "pilot011/model_000469.pt"}
states = {name:torch.load(path, map_location="cpu", weights_only=True) for name,path in paths.items()}
parent, initial, first, warm, actor = [states[name] for name in paths]


def exact(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


checks = dict(full_native_restore=all(initial["infos"]["full_restore_verification"].values()),
    parent_critic_exact=exact(parent["critic_state_dict"],initial["critic_state_dict"]),
    actor_unchanged_first_critic=exact(initial["actor_state_dict"],first["actor_state_dict"]),
    actor_unchanged_all_twenty_warmup=exact(initial["actor_state_dict"],warm["actor_state_dict"]))
before, after = initial["optimizer_state_dict"], warm["optimizer_state_dict"]
checks["actor_Adam_unchanged_all_twenty_warmup"] = all(
    exact(before["state"].get(k),after["state"].get(k)) for k in before["param_groups"][0]["params"])
for group in (0,1):
    old = parent["optimizer_state_dict"]["param_groups"][group]["params"]
    new = initial["optimizer_state_dict"]["param_groups"][group]["params"]
    checks["parent_Adam_group_"+str(group)+"_exact"] = all(
        exact(parent["optimizer_state_dict"]["state"].get(a),initial["optimizer_state_dict"]["state"].get(b))
        for a,b in zip(old,new))
checks["new_feedback_Adam_initially_empty"] = all(k not in before["state"] for k in (13,14,15,16))
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
a,b=mean_from_checkpoint(parent,source),mean_from_checkpoint(initial,source)
obs=torch.from_numpy(np.load(RUN / "rmcs448_envelope001/traces.npz")["observations"][::50].reshape(-1,35))
with torch.no_grad():
    checks["native_initial_mean_exact_parent_real_inputs"] = torch.equal(a(obs),b(obs))
initial_names = initial["actor_state_dict"]
changed = [k for k,v in initial_names.items() if not torch.equal(v,actor["actor_state_dict"][k])]
checks["first_actor_changes_only_corrections"] = bool(changed) and all(k.startswith(
    ("mlp.command.residual.","mlp.feedback.")) for k in changed)
checks["first_actor_updates_feedback"] = any(k.startswith("mlp.feedback.") for k in changed)
checks["first_actor_updates_command_branch"] = any(k.startswith("mlp.command.residual.") for k in changed)
checks["first_critic_value_updates"] = not exact(initial["critic_state_dict"],first["critic_state_dict"])
checks["twenty_critic_updates_charged"] = (warm["infos"]["successful_updates_total"]==468
    and warm["infos"]["actor_updates_total"]==378 and warm["infos"]["consumed_optimizer_updates_total"]==469
    and warm["infos"]["feedback_branch_actor_updates"]==0)
checks["first_actor_counter"] = (actor["infos"]["feedback_branch_actor_updates"]==1
    and actor["infos"]["active_branch_actor_updates"]==45
    and actor["infos"]["actor_updates_total"]==379)
timing=actor["infos"]["startup"]["control_timing"]
checks["native_50Hz_policy_200Hz_physics_four_PD"] = (
    timing["policy_dt"]==.02 and timing["physics_dt"]==.005 and timing["decimation"]==4
    and timing["runtime_substep_verification"]["pd_updates"]==4)
settings=actor["infos"]["training_command_reference"]
checks["native_population_mixed_commands"] = settings["direct_lanes"]==364 and settings["ramped_lanes"]==148
result=dict(schema="flat-feedback-native469-milestone-proof-v1",passed=all(checks.values()),checks=checks,
    milestones_sha256={name:digest(path) for name,path in paths.items()},real_inputs=len(obs),
    changed_first_actor_state_names=changed,new_optimizer_updates_by_this_inspector=0,
    successful_updates_at_last_milestone=469,consumed_updates_at_last_milestone=470,
    actor_updates_all_local_trials=379,command_branch_actor_updates=45,feedback_branch_actor_updates=1,
    hardware_qualified=False,inspector_sha256=digest(__file__))
(ROOT / "reports/flat_base_continuation_20261019/feedback_native469.json").write_bytes(encode(result))
print(json.dumps(result,indent=2))
if not result["passed"]:
    raise SystemExit(1)
