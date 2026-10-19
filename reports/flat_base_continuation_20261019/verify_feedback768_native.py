"""Validate native parent restoration, the declared rate change and scoped learning."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import TRACKING_KIND

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--first-only", action="store_true")
args = parser.parse_args()
torch.set_num_threads(4)
RUN = ROOT / "runs/flat_base_20261019"
TRIAL = RUN / "pilot015"
HERE = Path(__file__).resolve().parent
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
expected = load(RUN / "pilot014/model_000704.pt")
initial = load(TRIAL / "initial_state.pt")
parent_mean = mean_from_checkpoint(expected, source)
names = ["distribution.std_param"] + ["mlp." + n for n, _ in parent_mean.named_parameters()]
expected_opt = deepcopy(expected["optimizer_state_dict"])
expected_opt["param_groups"][0]["lr"] = 3e-4


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(v, b[k]) for k,v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a == b


info = initial["infos"]
revision = json.loads((HERE / "training_plan_revision013.json").read_text())
checks = {"initial_" + k:exact(initial[k], expected[k]) for k in ("actor_state_dict", "critic_state_dict")}
checks.update(initial_optimizer_only_declared_actor_lr=exact(initial["optimizer_state_dict"], expected_opt),
    native_parent_restore_before_rate_and_reset=all(info["full_restore_verification"].values()),
    native_post_rate_exact=info["optimizer_after_declared_rate_exact"],
    actual_rate_contract=info["optimizer_rate_contract"]["before"] == dict(actor=1e-4, critic=5e-4)
        and info["optimizer_rate_contract"]["after"] == dict(actor=3e-4, critic=5e-4)
        and info["optimizer_rate_contract"]["declared_actor_rate_change"],
    parent_counters_retained=(info["successful_updates_total"],info["actor_updates_total"],
        info["active_branch_actor_updates"],info["feedback_branch_actor_updates"],info["consumed_optimizer_updates_total"])
        == (704,554,44,176,705),
    no_additional_critic_warmup=info["critic_warmup_remaining"] == 0,
    no_new_gate_or_command_composition=info["actor_adaptation"] == TRACKING_KIND
        and not any(k in info for k in ("feedback_gate_revision", "command_branch_restore")),
    exact_declared_sources=info["source_sha256"] == revision["new_source_sha256"],
    exact_declared_plan=info["plan_sha256"] == revision["new_plan_sha256"],
    exact_declared_parent=info["parent_checkpoint_sha256"] == revision["parent_checkpoint_sha256"])
base_names = [n for n in initial["actor_state_dict"] if n.startswith("mlp.command.base.")]
checks["original_eight_tensors_present"] = len(base_names) == 8 and sum(
    initial["actor_state_dict"][n].numel() for n in base_names) == 50758
data = np.load(RUN / "rmcs704_envelope001/traces.npz")
x = torch.from_numpy(data["observations"][::5].reshape(-1,35))
with torch.no_grad():
    checks["initial_mean_exact_parent_recorded_inputs"] = torch.equal(
        parent_mean(x), mean_from_checkpoint(initial,source)(x))
groups = initial["optimizer_state_dict"]["param_groups"]
actor_ids = groups[0]["params"]
frozen_ids = [i for n,i in zip(names,actor_ids,strict=True) if not n.startswith("mlp.feedback.")]
feedback_ids = [i for n,i in zip(names,actor_ids,strict=True) if n.startswith("mlp.feedback.")]
initial_slots = initial["optimizer_state_dict"]["state"]
milestones = {}
required = (705,) if args.first_only else (705,736,768)
for step in required:
    path = TRIAL / f"model_{step:06d}.pt"
    if not path.exists():
        continue
    state=load(path); info=state["infos"]; slots=state["optimizer_state_dict"]["state"]
    delta=step-704
    row = dict(checkpoint_sha256=sha(path),successful_updates_total=info["successful_updates_total"],
        actor_updates_total=info["actor_updates_total"],command_branch_actor_updates=info["active_branch_actor_updates"],
        feedback_branch_actor_updates=info["feedback_branch_actor_updates"])
    row["checks"] = dict(frozen_mean_std_buffers_exact=all(torch.equal(v,initial["actor_state_dict"][n])
        for n,v in state["actor_state_dict"].items() if not n.startswith("mlp.feedback.")),
        frozen_Adam_slots_exact=all(exact(slots.get(i),initial_slots.get(i)) for i in frozen_ids),
        all_four_feedback_tensors_learned=all(not torch.equal(v,initial["actor_state_dict"][n])
            for n,v in state["actor_state_dict"].items() if n.startswith("mlp.feedback.")),
        only1126_feedback_parameters_trainable=info["actor_trainability"]["trainable_parameters"] == 1126
            and all(n.startswith("mlp.feedback.") for n in info["actor_trainability"]["trainable_names"]),
        all_updates_are_actor_updates=info["actor_updates_total"] == 554+delta
            and info["feedback_branch_actor_updates"] == 176+delta and info["critic_warmup_remaining"] == 0,
        command_clock_fixed44=info["active_branch_actor_updates"] == 44,
        consumed_without_refund=info["consumed_optimizer_updates_total"] == step+1,
        feedback_and_critic_Adam_steps_retained=all(float(slots[i]["step"]) == float(initial_slots[i]["step"])+12*delta
            for i in feedback_ids+groups[1]["params"]),
        actual_learning_rates=[g["lr"] for g in state["optimizer_state_dict"]["param_groups"]] == [3e-4,5e-4],
        actual_rate_log=math.isclose(info["last_losses"]["actor_learning_rate"],3e-4,rel_tol=1e-12)
            and math.isclose(info["last_losses"]["critic_learning_rate"],5e-4,rel_tol=1e-12),
        all_logged_losses_finite=all(np.isfinite(v) for v in info["last_losses"].values()),
        original_eight_tensors_exact=len(base_names) == 8 and all(
            torch.equal(initial["actor_state_dict"][n],state["actor_state_dict"][n]) for n in base_names))
    row["last_losses"] = info["last_losses"]
    milestones[str(step)]=row
    checks.update({str(step)+"_"+k:v for k,v in row["checks"].items()})
receipt=dict(schema="flat-feedback768-native-milestone-verification-v1",passed=all(checks.values()),checks=checks,
    milestones=milestones,all_required_milestones_present=len(milestones)==len(required),
    optimizer_updates_performed_by_verifier=0,qualification_eligible=False,first_only=args.first_only)
path = HERE / ("feedback768_native705.json" if args.first_only else "feedback768_native_milestones.json")
path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
print(json.dumps(dict(passed=receipt["passed"],checks=len(checks),milestones=list(milestones),
    all_required_milestones_present=receipt["all_required_milestones_present"],failures=[k for k,v in checks.items() if not v])))
if not receipt["passed"] or not receipt["all_required_milestones_present"]:
    raise SystemExit(1)
