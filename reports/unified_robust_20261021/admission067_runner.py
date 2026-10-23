"""Serial fresh nominal confirmation and long standing; no PPO is started here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import assess
from iteration045_runner import native, write
from iteration036_compare import read, standing
from iteration048_runner import static_failures


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory):
    with (REPORTS / (directory.parent.name + "_" + directory.name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-result", type=Path, default=REPORTS / "iteration065_final.json")
    args = parser.parse_args()
    assert args.lifecycle_result.resolve() in {REPORTS.resolve() / name
        for name in ("iteration065_final.json", "iteration069_final.json", "iteration071_final.json")}
    checkpoint = RUNS / "quiet_severe_reference065/model_migrated.pt"
    protocol_path = REPORTS / "acceptance_protocol003.json"
    protocol = json.loads(protocol_path.read_text())
    lifecycle = json.loads(args.lifecycle_result.read_text())
    strong_push = json.loads((REPORTS / "iteration065_final.json").read_text())
    migration = json.loads((REPORTS / "iteration065_migration.json").read_text())
    static = json.loads((REPORTS / "iteration066_final.json").read_text())
    assert lifecycle["native_lifecycle_exercised_and_valid"]
    assert lifecycle["mechanical_gate_passed"]
    assert migration["checkpoint_sha256"] == sha(checkpoint)
    assert lifecycle["after"]["checkpoint_sha256"] == sha(checkpoint)
    assert lifecycle["before"]["checkpoint_sha256"] == migration["source_sha256"]
    assert migration["source_sha256"] == static["checkpoint_sha256"]
    assert not static["static_failures"] and static["full_suite_complete"]
    pipeline = dict(schema="reference067-admission-pipeline-v1", status="running", phase="fresh_nominal",
        started=time.time(), checkpoint_sha256=sha(checkpoint), protocol_sha256=sha(protocol_path),
        new_macro_updates=0, full_training_admitted=False)
    write(REPORTS / "admission067_pipeline.json", pipeline)
    try:
        reports, baselines = [], []
        for seed in protocol["required_confirmation_seeds"]:
            name = f"fresh_nominal067_{seed}"
            output = RUNS / name
            # 35 cases x 3 replicas x 2 actors = 210 native environments.
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
                "--output", str(output), "--variants", "parent", "robust", "--replicas", "3",
                "--seed", str(seed), "--domain-scale", "0", "--seconds", "12", "--isolated-origins"]
            native(name, command, output)
            for variant in ("parent", "robust"):
                analyze(output / variant)
            reports.append(output / "robust")
            baselines.append(output / "parent")
        nominal = assess(reports, baselines, checkpoint, protocol)
        # Preserve the original assessor and additionally check the full window.
        nominal_full_window = {}
        for seed in protocol["required_confirmation_seeds"]:
            evidence = read(f"fresh_nominal067_{seed}")
            rows = standing(evidence)
            nominal_full_window[str(seed)] = rows
            for case, row in rows.items():
                if row["full_window_max_excursion_m"] > .05:
                    nominal["failures"].append(f"seed {seed} {case} full-window5cm")
                if row["wheel_clearance_after_02s_peak_m"] > .003:
                    nominal["failures"].append(f"seed {seed} {case} post0.2s clearance3mm")
        nominal["adequate"] = not nominal["failures"]
        result = dict(schema="reference067-admission-v1", checkpoint_sha256=sha(checkpoint),
            protocol_sha256=sha(protocol_path), nominal=nominal, nominal_full_window=nominal_full_window,
            lifecycle_result=str(args.lifecycle_result), lifecycle_result_sha256=sha(args.lifecycle_result),
            known_strong_stationary_push_result_sha256=sha(REPORTS / "iteration065_final.json"),
            known_strong_stationary_push_summary=strong_push["after"]["summary"],
            new_macro_updates=0, robust_consumed=1160, long_training_updates=0,
            full_training_admitted=False, hardware_ready=False, installed_in_rmcs=False)
        write(REPORTS / "admission067_partial.json", result)
        if nominal["adequate"]:
            pipeline["phase"] = "long_standing"
            write(REPORTS / "admission067_pipeline.json", pipeline)
            name = "standing_30s_reference067"
            output = RUNS / name
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
                "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "30", "--isolated-origins",
                "--case-names", "stand_low", "stand", "stand_high"]
            native(name, command, output)
            analyze(output / "robust")
            evidence = read(name)
            rows = standing(evidence)
            failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
            result.update(long_standing=rows, long_standing_failures=failures,
                          long_standing_trajectory_sha256=evidence[1]["trajectory_sha256"])
            result["full_training_admitted"] = not failures
        result["status"] = "simulation_admitted_for_long_training" if result["full_training_admitted"] else "needs_iteration"
        write(REPORTS / "admission067_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time(),
                        full_training_admitted=result["full_training_admitted"])
        write(REPORTS / "admission067_pipeline.json", pipeline)
        print(json.dumps({k: result[k] for k in ("status", "full_training_admitted")}), flush=True)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(REPORTS / "admission067_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
