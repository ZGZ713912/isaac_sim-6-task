"""Inspect actual saved native milestones; no sampling or optimization."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.guarded_feedback_residual import TRACKING_KIND

RUN = ROOT / "runs/flat_base_20261019"
TRIAL = RUN / "pilot014"
source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
expected = load(RUN / "pilot013/model_000640.pt")
mean = mean_from_checkpoint(expected, source)
names = ["distribution.std_param"] + ["mlp." + n for n, _ in mean.named_parameters()]
initial = load(TRIAL / "initial_state.pt")


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


checks = {"initial_" + k:exact(initial[k], expected[k])
          for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
checks["native_restore_before_reset_all_true"] = all(initial["infos"]["full_restore_verification"].values())
checks["initial_count_and_consumption_retained"] = (initial["infos"]["successful_updates_total"],
    initial["infos"]["consumed_optimizer_updates_total"], initial["infos"]["active_branch_actor_updates"],
    initial["infos"]["feedback_branch_actor_updates"]) == (640, 641, 44, 132)
checks["same_tracking_gate_kind"] = initial["infos"]["actor_adaptation"] == TRACKING_KIND
checks["no_new_gate_or_command_composition"] = not any(k in initial["infos"] for k in
    ("feedback_gate_revision", "command_branch_restore"))
xdata = np.load(RUN / "rmcs640_envelope001/traces.npz")
x = torch.from_numpy(xdata["observations"][::5].reshape(-1, 35))
actual_mean = mean_from_checkpoint(initial, source)
with torch.no_grad():
    checks["initial_mean_exact_parent_all_recorded_inputs"] = torch.equal(mean(x), actual_mean(x))
groups = initial["optimizer_state_dict"]["param_groups"]
actor_ids = groups[0]["params"]
frozen_ids = [i for n, i in zip(names, actor_ids, strict=True) if not n.startswith("mlp.feedback.")]
feedback_ids = [i for n, i in zip(names, actor_ids, strict=True) if n.startswith("mlp.feedback.")]
milestones = {}
for step in (641, 660, 661, 704):
    path = TRIAL / f"model_{step:06d}.pt"
    if not path.exists():
        continue
    state = load(path)
    info = state["infos"]
    row = dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        successful_updates_total=info["successful_updates_total"], actor_updates_total=info["actor_updates_total"],
        command_branch_actor_updates=info["active_branch_actor_updates"], feedback_branch_actor_updates=info["feedback_branch_actor_updates"])
    opt = state["optimizer_state_dict"]["state"]
    row["checks"] = dict(
        frozen_mean_std_buffers_exact=all(torch.equal(v, initial["actor_state_dict"][n])
            for n,v in state["actor_state_dict"].items() if not n.startswith("mlp.feedback.")),
        frozen_Adam_exact=all(exact(opt.get(i), initial["optimizer_state_dict"]["state"].get(i)) for i in frozen_ids),
        command_counter_fixed44=info["active_branch_actor_updates"] == 44,
        consumed_without_refund=info["consumed_optimizer_updates_total"] == step + 1,
        feedback_clock_correct=info["feedback_branch_actor_updates"] == 132 + max(step - 660, 0),
        actor_clock_correct=info["actor_updates_total"] == 510 + max(step - 660, 0),
        native_restore_all_true=all(info["full_restore_verification"].values()))
    if step <= 660:
        row["checks"].update(warmup_actor_all_exact=exact(state["actor_state_dict"], initial["actor_state_dict"]),
            warmup_actor_Adam_all_exact=all(exact(opt.get(i), initial["optimizer_state_dict"]["state"].get(i)) for i in actor_ids))
    else:
        row["checks"]["all_four_feedback_tensors_learned"] = all(not torch.equal(v, initial["actor_state_dict"][n])
            for n,v in state["actor_state_dict"].items() if n.startswith("mlp.feedback."))
        row["checks"]["only_1126_feedback_parameters_trainable"] = (info["actor_trainability"]["trainable_parameters"] == 1126
            and all(n.startswith("mlp.feedback.") for n in info["actor_trainability"]["trainable_names"]))
    row["command_Adam_steps"] = [float(opt[i]["step"]) for n,i in zip(names, actor_ids, strict=True) if n.startswith("mlp.command.residual.")]
    row["feedback_Adam_steps"] = [float(opt[i]["step"]) for i in feedback_ids]
    row["critic_Adam_steps"] = [float(opt[i]["step"]) for i in groups[1]["params"]]
    milestones[str(step)] = row
    checks.update({str(step) + "_" + k:v for k,v in row["checks"].items()})
receipt = dict(schema="flat-feedback704-native-milestone-verification-v1", passed=all(checks.values()),
    checks=checks, milestones=milestones, optimizer_updates_performed_by_verifier=0,
    qualification_eligible=False, all_required_milestones_present=len(milestones) == 4)
path = ROOT / "reports/flat_base_continuation_20261019/feedback704_native_milestones.json"
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(passed=receipt["passed"], checks=len(checks), milestones=list(milestones),
    all_required_milestones_present=receipt["all_required_milestones_present"], failures=[k for k,v in checks.items() if not v])))
if not receipt["passed"] or not receipt["all_required_milestones_present"]:
    raise SystemExit(1)
