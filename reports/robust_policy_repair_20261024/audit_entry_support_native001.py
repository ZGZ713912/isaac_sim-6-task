"""Admit a bounded reward A/B only after completed native causality checks."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ["src", "scripts"]
import numpy as np
import torch
from wheeled_tasks.chassis.recovery_support_reward import recovery_support_cost

report = Path("reports/robust_policy_repair_20261024")
base = Path("runs/robust_policy_repair_20261024")
queue = json.loads((report / "diagnosis_queue018.json").read_text())
assert queue["status"] == "completed" and all(j["status"] == "completed" for j in queue["jobs"])
before, after = [base / f"diagnosis018_entry_support{w}_native" for w in (0, 1)]
traces = [p / "evaluation/trajectory.npz" for p in (before, after)]
a, b = [np.load(p) for p in traces]
settings = [json.loads((p / "task_contract.json").read_text())["settings"] for p in (before, after)]
assert settings[0]["recovery_support_weight"] == 0. and settings[1]["recovery_support_weight"] == 1.
assert {k: v for k, v in settings[0].items() if k != "recovery_support_weight"} == {
    k: v for k, v in settings[1].items() if k != "recovery_support_weight"}
excluded = {"native_reward", "reward_density", "recovery_support_cost_density"}
differences = {k: float(np.abs(a[k].astype("float64") - b[k].astype("float64")).max())
               for k in a.files if k not in excluded and not np.array_equal(a[k], b[k])}
# Previous native repeats differ by one raw force ULP; all integrated state,
# impulse, applied torque, observations, actions and decisions must be exact.
checks = dict(all_control_state_and_decisions_exact=all(k == "impact" and v <= 2e-5
                                                       for k, v in differences.items()),
              zero_weight_cost_exactly_zero=bool(np.all(a["recovery_support_cost_density"] == 0.)))
reconstructed = recovery_support_cost(torch.tensor(b["wheel_clearance"]),
    torch.tensor(b["gravity"][..., 2]), torch.tensor(b["reward_recovery_active"]), 1.).numpy()
cost = b["recovery_support_cost_density"]
cost_error = float(np.abs(cost - reconstructed).max())
reward_error = float(np.abs(b["native_reward"] - a["native_reward"] + cost * .005 * ~b["passive"]).max())
density_error = float(np.abs(b["reward_density"] - a["reward_density"] + cost).max())
checks.update(native_cost_reconstructs=cost_error <= 5e-6,
    reward_delta_is_only_added_cost=reward_error <= 1e-6,
    density_delta_is_only_added_cost=density_error <= 1e-5,
    cost_exercises_nonzero_frames=bool(np.any(cost > 1.)),
    outside_recovery_exact_zero=bool(np.all(cost[~b["reward_recovery_active"]] == 0.)),
    side_inverted_exact_zero=bool(np.all(cost[b["gravity"][..., 2] >= -.7] == 0.)))
for key in ("root_pose", "motor_position", "motor_velocity"):
    x, y = [np.load(p / "evaluation_precontrol_state.npz")[key] for p in (before, after)]
    checks["precontrol_" + key + "_exact"] = bool(np.array_equal(x, y))
preflight = json.loads((report / "entry_learning_preflight001.json").read_text())
checks["CPU_all_parameter_and_Adam_checks"] = all(all(r["checks"].values()) for r in preflight["rows"])
checks["return_order_gate"] = json.loads((report / "recovery_support_return_order_gate001.json").read_text())["all_pass"]
receipt = dict(schema="paired-entry-support-native-admission-v1",
    decision="admit_one_predeclared_pair8plus8" if all(checks.values()) else "reject_PPO",
    checks=checks, differences=differences, reconstructed_cost_max_abs_error=cost_error,
    native_reward_delta_max_abs_residual=reward_error, density_delta_max_abs_residual=density_error,
    nonzero_cost_frames=int((cost > 0).sum()),
    trajectory_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in traces],
    protocol=str(report / "entry_reach_AB_protocol001.json"), new_PPO=0,
    actual_experiment_consumed=2587, qualification=None, hardware_started=False,
    created_unix=time.time())
path = report / "entry_reach_AB_admission001.json"
assert not path.exists()
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
assert all(checks.values()), checks
