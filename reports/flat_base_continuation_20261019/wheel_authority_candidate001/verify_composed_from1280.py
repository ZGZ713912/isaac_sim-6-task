"""Verify the already-written v4-1280 -> v5 -> v6 composition artifact."""
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "runs/flat_base_20261019"
parent_path = RUN / "pilot025/model_001280.pt"
composed_path = RUN / "pilot025/model_001280_wheel_authority_composed.pt"
static_path = RUN / "pilot023/model_001248.pt"
plan_path = HERE / "plan_wheel_from1280.json"
out = HERE / "compose_from1280_cpu001.json"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: torch.load(p, map_location="cpu", weights_only=True)

if not composed_path.is_file() or out.exists():
    raise SystemExit("Preserve the existing composition or receipt")

def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b

parent, composed, static = load(parent_path), load(composed_path), load(static_path)
plan = json.loads(plan_path.read_text())
spec = importlib.util.spec_from_file_location("wheel_authority_composed_verify", HERE / "wheel_authority.py")
wheel = importlib.util.module_from_spec(spec)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
spec.loader.exec_module(wheel)
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.static_feedback import StaticFeedbackActor

source = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
moving = mean_from_checkpoint(parent, source)
recalled = mean_from_checkpoint(static, source)
static_actor = StaticFeedbackActor(moving, recalled, plan["static_feedback_recall"]["sha256"])
wheel_actor = wheel.WheelAuthorityActor(static_actor)

parent_actor = parent["actor_state_dict"]
state = composed["actor_state_dict"]
old_names = set(parent_actor)
wheel_names = {"mlp.wheel_authority.0.weight", "mlp.wheel_authority.0.bias",
               "mlp.wheel_authority.2.weight", "mlp.wheel_authority.2.bias"}
static_names = {"mlp.static_feedback.0.weight", "mlp.static_feedback.0.bias",
                "mlp.static_feedback.2.weight", "mlp.static_feedback.2.bias"}
expected_new = static_names | wheel_names | {"mlp.authority_bound"}
checks = {
    "parent_checkpoint_exact": sha(parent_path) == plan["parent_composition"]["parent_sha256"],
    "parent_lifetime_counters_exact": all(composed["infos"][k] == parent["infos"][k] for k in (
        "successful_updates_total", "actor_updates_total", "consumed_optimizer_updates_total",
        "training_transitions_total")),
    "parent_critic_and_sampling_rng_exact": all(exact(composed[k], parent[k])
        for k in ("critic_state_dict", "torch_rng_state", "cuda_rng_state", "env_generator_state")),
    "old_actor_weights_exact": all(torch.equal(state[k], parent_actor[k]) for k in old_names),
    "only_declared_new_actor_state": set(state) - old_names == expected_new,
    "static_recall_exact": all(torch.equal(state["mlp." + k], v) for k, v in
        static_actor.state_dict().items() if k.startswith("static_feedback."))
        and composed["infos"]["static_feedback_recall"] == plan["static_feedback_recall"],
    "wheel_final_layer_zero": all(torch.count_nonzero(state[k]).item() == 0 for k in (
        "mlp.wheel_authority.2.weight", "mlp.wheel_authority.2.bias")),
    "wheel_first_layer_finite": all(torch.isfinite(state[k]).all().item() for k in (
        "mlp.wheel_authority.0.weight", "mlp.wheel_authority.0.bias")),
    "wheel_bound_exact": torch.equal(state["mlp.authority_bound"], torch.tensor([6., 6.])),
    "declared_v6_plan_and_kind": composed["infos"]["actor_adaptation"] == wheel.KIND
        and composed["infos"]["plan_file"] == str(plan_path.relative_to(ROOT))
        and composed["infos"]["plan_sha256"] == sha(plan_path),
    "no_optimizer_update_or_native_step": composed["infos"]["new_optimizer_updates"] == 0
        and composed["infos"].get("new_native_steps", 0) == 0,
    "composed_checkpoint_exists": composed_path.exists(),
}
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2, ensure_ascii=False))
out.write_text(json.dumps({
    "schema": "flat-wheel-authority-from1280-composition-CPU-v1",
    "recorded_at": datetime.now(timezone.utc).isoformat(), "passed": True,
    "checks": checks, "parent_checkpoint_sha256": sha(parent_path),
    "composed_checkpoint_sha256": sha(composed_path),
    "static_reference_sha256": sha(static_path), "plan_sha256": sha(plan_path),
    "new_optimizer_updates": 0, "new_native_steps": 0, "adopted": False,
    "behavior_qualified": False,
    "scope": "Composition artifact only; no environment step or optimizer update. Behavior must be re-evaluated from this exact composed checkpoint.",
}, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({"passed": True, "checks": len(checks),
                  "composed_checkpoint_sha256": sha(composed_path)}, indent=2))
