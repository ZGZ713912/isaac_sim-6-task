"""Single-variable zero-PPO policy-observation-delay counterfactual for050."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import write, native
from iteration036_compare import read, standing


def pairing(before, after):
    a, b = before[1], after[1]
    checks = {key: [r.get(key) for r in a["lanes"]] == [r.get(key) for r in b["lanes"]]
              for key in ("name", "pose_index", "command", "recovery", "world_origin_m",
                          "material_coefficients", "sequence", "push", "push_sign")}
    domains, shifted, errors = [], [], []
    first_a, first_b = before[3]["observations"][0], after[3]["observations"][0]
    checks["commands_history_context_exact"] = bool(np.array_equal(
        first_a[:, np.r_[0:4, 22:35]], first_b[:, np.r_[0:4, 22:35]]))
    for old, new in zip(a["lanes"], b["lanes"]):
        x, y = copy.deepcopy(old["first_reset_domain"]), copy.deepcopy(new["first_reset_domain"])
        delay = x.pop("sensor_delay_policy_ticks")
        domains.append(y.pop("sensor_delay_policy_ticks") == 0 and x == y)
        i = old["lane"]
        # Delayed lane's second observation contains the original first noisy
        # measurement. Zero-delay lanes retain the original first measurement.
        expected = before[3]["observations"][1 if delay else 0, i, 4:22]
        errors.append(float(np.max(np.abs(first_b[i, 4:22] - expected))))
        if delay:
            shifted.append(i)
    checks["only_declared_domain_delay_changed"] = all(domains)
    checks["first_measurement_matches_declared_time_shift"] = max(errors) < 1e-6
    checks["native_origin_mode"] = a["native_origin_mode"] == b["native_origin_mode"]
    checks["checkpoint_exact"] = a["checkpoint_sha256"] == b["checkpoint_sha256"]
    checks["only_declared_override"] = b["evaluation_overrides"] == {"sensor_delay_policy_ticks": 0}
    if not all(checks.values()):
        raise ValueError(f"Counterfactual mismatch: {checks}; measurement error {max(errors)}")
    return dict(checks=checks, affected_lanes=shifted, first_measurement_max_abs_error=max(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args()
    name = "instant_policy_delay051"
    output = RUNS / name
    checkpoint = RUNS / "pilot027_rolling_feedback050/model_final.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "e4a9d18332256990c1cdf689b31dcd18d64cf2b98e116c7ca04a0fef4e21dd30"
    before = read("learned_rolling_static050")
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
        "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
        "--case-names", "stand_low", "stand", "stand_high", "--sensor-delay-policy-ticks", "0"]
    pipeline = dict(schema="policy-delay051-counterfactual-v1", status="running", started=time.time(),
        optimizer_updates=0, robust_consumed=1025, long_training_admitted=False,
        command=command, checkpoint_sha256=before[1]["checkpoint_sha256"],
        intervention="Only policy observation delay becomes zero after matched reset; noise/bias, actuator/PD delays and all policy weights retained",
        qualification=False, hardware_ready=False)
    write(REPORTS / "iteration051_pipeline.json", pipeline)
    plan_path = REPORTS / "iteration051_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["run_started"] = True
    write(plan_path, plan)
    try:
        if args.reuse_completed:
            receipt = json.loads((REPORTS / (name + "_process.json")).read_text())
            progress = json.loads((output / "progress.json").read_text())
            assert receipt["command"] == command and receipt["returncode"] == 0 and receipt["native_status"] == "completed"
            assert progress["status"] == "completed" and progress["checkpoint_sha256"] == pipeline["checkpoint_sha256"]
        else:
            native(name, command, output)
        with (REPORTS / (name + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        after = read(name)
        result = dict(schema="policy-delay051-results-v1", pairing=pairing(before, after),
            before=standing(before), after=standing(after), optimizer_updates=0, robust_consumed=1025,
            long_training_admitted=False, qualification=False, hardware_ready=False,
            trajectory_sha256=after[1]["trajectory_sha256"],
            report_sha256=hashlib.sha256((after[0] / "report.json").read_bytes()).hexdigest())
        write(REPORTS / "iteration051_final.json", result)
        pipeline.update(status="completed", finished=time.time())
        write(REPORTS / "iteration051_pipeline.json", pipeline)
        print(json.dumps(result), flush=True)
    except BaseException:
        pipeline.update(status="failed", finished=time.time(), error=traceback.format_exc())
        write(REPORTS / "iteration051_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
