"""Explicit migration of the flat policy clock from 50 to 200 Hz."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


DEFAULT_PLAN = "contracts/flat_base_finetune_200hz_v1.json"
CONTROL = "contracts/flat_base_200hz_control_v1.json"
SCHEMA = "flat-base-finetune-plan-v18"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_reference_at_policy_rate(source, policy_hz):
    """Retain inspected RMCS provenance while declaring a simulation clock."""
    if policy_hz == source["policy_hz"]:
        return deepcopy(source)
    if source["policy_hz"] != 50 or policy_hz != 200:
        raise ValueError("Only the declared 50 to 200 Hz command migration is supported")
    result = deepcopy(source)
    result.update(schema="flat-rmcs-command-reference-policy-rate-v2", policy_hz=200,
        source_policy_hz=50, source_scope=source["scope"],
        scope="200 Hz native simulation derived from the inspected 50 Hz RMCS command math")
    return result


def make_policy_rate_plan(root, old, old_plan_file):
    root = Path(root)
    if old["policy_dt"] != .02 or old["physics_dt"] != .001:
        raise ValueError("200 Hz flat migration requires the saved 50/1000 Hz parent")
    result = deepcopy(old)
    ratio = .005 / old["policy_dt"]
    rollout = old["num_steps_per_env"] / ratio
    if not rollout.is_integer():
        raise ValueError("Policy migration must retain an integral rollout horizon")
    result.update(schema=SCHEMA, policy_dt=.005, gamma=old["gamma"]**ratio,
        lam=old["lam"]**ratio, num_steps_per_env=int(rollout),
        control_file=CONTROL, control_sha256=_digest(root / CONTROL))
    result["policy_rate_transition"] = dict(schema="flat-policy-rate-transition-v1",
        parent_plan_file=str(old_plan_file), parent_plan_sha256=_digest(root / old_plan_file),
        source_policy_hz=50, policy_hz=200, physics_hz=1000,
        rollout_seconds=old["num_steps_per_env"] * old["policy_dt"],
        discount_and_gae="exponential physical time", action_difference_reference_dt=.02,
        reward_integration="densities multiplied by policy_dt once; event costs once per event",
        command_and_episode_timers="seconds unchanged",
        previous_action_observation_dt=.005,
        exploration="saved std retained; 200 Hz sampling has a different temporal correlation",
        behavior_equivalence_verified=False)
    if result.get("training_command_reference"):
        result["training_command_reference"]["contract"] = command_reference_at_policy_rate(
            old["training_command_reference"]["contract"], 200)
    return result


def validate_policy_rate_transition(root, old, new, parent_control_sha256):
    """Authenticate the full time contract, without changing learned state."""
    root = Path(root)
    transition = new.get("policy_rate_transition", {})
    parent = transition.get("parent_plan_file", "")
    if (new.get("schema") != SCHEMA or not parent or
            _digest(root / parent) != transition.get("parent_plan_sha256") or
            json.loads((root / parent).read_text()) != old or
            parent_control_sha256 != old["control_sha256"]):
        raise ValueError("Policy clock migration does not authenticate its exact parent")
    if new != make_policy_rate_plan(root, old, parent):
        raise ValueError("200 Hz migration changed rollout, discount, costs, commands or another undeclared field")
    controls = []
    for plan in (old, new):
        path = root / plan["control_file"]
        if _digest(path) != plan["control_sha256"]:
            raise ValueError("Policy clock control source changed")
        controls.append(json.loads(path.read_text()))
    ignored = {"control_id", "timing"}
    if {k:v for k,v in controls[0].items() if k not in ignored} != {
            k:v for k,v in controls[1].items() if k not in ignored}:
        raise ValueError("Policy clock migration changed motors, geometry or gains")
    expected = dict(physics_dt=.001, policy_dt=.005, pc_control_dt=.001, decimation=5)
    if controls[1]["timing"] != expected:
        raise ValueError("200 Hz policy requires five 1 ms physical/PD steps")
    for name in ("gamma", "lam"):
        if not math.isclose(new[name]**4, old[name], abs_tol=1e-14, rel_tol=0):
            raise ValueError("Discount/GAE physical horizon changed")
    return dict(schema="flat-policy-clock-migration-verification-v1", policy_hz=200,
        physics_hz=1000, feedback_pd_hz=1000, decimation=5,
        gamma=new["gamma"], lam=new["lam"], num_steps_per_env=new["num_steps_per_env"],
        rollout_seconds=new["num_steps_per_env"] * new["policy_dt"],
        policy_samples_per_update_multiplier=4, simulated_seconds_per_update_unchanged=True,
        action_difference_scale=4., second_difference_scale=16.,
        weights_std_adam_rng_counters="must restore exactly before any optimizer update",
        previous_action_and_exploration_time_semantics_changed=True,
        behavior_equivalence_verified=False, new_optimizer_updates=0)
