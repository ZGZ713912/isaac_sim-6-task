"""Three serial, matched, zero-PPO bias counterfactuals on the retained mean."""
from pathlib import Path
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
sys.path[:0] = [str(REPORTS)]
from iteration040_runner import native
from iteration036_compare import read, standing


def write(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def match_counterfactual(before, after, component):
    field = "pd_encoder_bias_rad" if component == "pd_encoder" else "sensor_bias"
    excluded = slice(None) if component == "pd_encoder" else slice(6, 10) if component == "policy_encoder" else slice(0, 3)
    checks = {}
    for key in ("name", "pose_index", "command", "recovery", "world_origin_m", "material_coefficients", "sequence", "push", "push_sign"):
        checks[key] = [x.get(key) for x in before[1]["lanes"]] == [x.get(key) for x in after[1]["lanes"]]
    original = [x["first_reset_domain"] for x in before[1]["lanes"]]
    changed = [x["first_reset_domain"] for x in after[1]["lanes"]]
    for a, b in zip(original, changed, strict=True):
        expected = np.asarray(a[field]).copy()
        expected[excluded] = 0.
        assert np.array_equal(expected, np.asarray(b[field]))
    checks["all_other_domain_fields"] = (
        [{k: v for k, v in x.items() if k != field} for x in original]
        == [{k: v for k, v in x.items() if k != field} for x in changed])
    checks["only_declared_bias_neutralized"] = True
    first_a, first_b = before[3]["observations"][0], after[3]["observations"][0]
    # Reset histories contain clean measurements. Record rather than conceal
    # whether the first frame already contains the declared bias intervention.
    checks["first_observation"] = np.array_equal(first_a, first_b)
    checks["native_origin_mode"] = before[1]["native_origin_mode"] == after[1]["native_origin_mode"]
    if not all(checks.values()):
        raise ValueError("Unexpected counterfactual mismatch: " + str(checks))
    return checks


def main():
    pipeline_path = REPORTS / "iteration042_pipeline.json"
    result = dict(schema="quiet-bias042-counterfactual-v1", status="running", started=time.time(),
        optimizer_updates=0, robust_consumed=929, long_training_admitted=False,
        qualification=False, intervention_results={})
    write(pipeline_path, result)
    baseline = read("quiet_velocity_baseline036")
    parent = RUNS / "quiet_wheel_memory040/model_migrated.pt"
    result.update(checkpoint_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                  baseline=standing(baseline),
                  baseline_mean_equivalence="040 migration15600real actions exact; all three new2x4weights remain zero")
    try:
        for component in ("policy_encoder", "policy_gyro", "pd_encoder"):
            name = "bias_" + component + "042"
            output = RUNS / name
            result["phase"] = component
            write(pipeline_path, result)
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(parent),
                "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
                "--case-names", "stand_low", "stand", "stand_high", "--neutralize-bias", component]
            native(name, command, output)
            with (REPORTS / (name + "_analysis.log")).open("w") as log:
                subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            evidence = read(name)
            result["intervention_results"][component] = dict(pairing=match_counterfactual(baseline, evidence, component),
                standing=standing(evidence), intervention=evidence[1]["bias_intervention"],
                trajectory_sha256=evidence[1]["trajectory_sha256"])
            write(pipeline_path, result)
            print(component, json.dumps(result["intervention_results"][component]), flush=True)
        result.update(status="completed", finished=time.time(), phase="all_complete")
        write(REPORTS / "iteration042_final.json", result)
    except BaseException:
        result.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(pipeline_path, result)


if __name__ == "__main__":
    main()
