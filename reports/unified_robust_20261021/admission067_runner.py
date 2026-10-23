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


def reuse_completed(name, command, output, checkpoint, variants):
    receipt = json.loads((REPORTS / (name + "_process.json")).read_text())
    progress = json.loads((output / "progress.json").read_text())
    if (receipt.get("status") != "completed" or receipt.get("returncode") != 0
            or receipt.get("native_status") != "completed" or receipt.get("command") != command
            or progress.get("status") != "completed"):
        raise ValueError("Completed native command identity required for reuse: " + name)
    for variant in variants:
        directory = output / variant
        report = json.loads((directory / "report.json").read_text())
        if (report["checkpoint_sha256"] != sha(checkpoint)
                or report["trajectory_sha256"] != sha(directory / "trajectory.npz")
                or report["actor_variant"] != variant):
            raise ValueError("Native artifact identity required for reuse: " + name)
        analysis = json.loads((directory / "analysis.json").read_text())
        if analysis["report_sha256"] != sha(directory / "report.json"):
            raise ValueError("Analysis no longer matches completed native report: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-result", type=Path, default=REPORTS / "iteration065_final.json")
    parser.add_argument("--collect-basic-only", action="store_true",
                        help="Collect nominal/long-standing evidence while lifecycle remains pending; never admit PPO")
    parser.add_argument("--reuse-completed", action="store_true",
                        help="Explicitly reuse hash-checked completed identical native jobs")
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
    if not args.collect_basic_only:
        assert lifecycle["native_lifecycle_exercised_and_valid"]
    assert not lifecycle["lifecycle_violations"]
    assert lifecycle["mechanical_gate_passed"]
    assert migration["checkpoint_sha256"] == sha(checkpoint)
    assert lifecycle["after"]["checkpoint_sha256"] == sha(checkpoint)
    assert lifecycle["before"]["checkpoint_sha256"] == migration["source_sha256"]
    assert migration["source_sha256"] == static["checkpoint_sha256"]
    assert not static["static_failures"] and static["full_suite_complete"]
    pipeline = dict(schema="reference067-admission-pipeline-v1", status="running", phase="fresh_nominal",
        started=time.time(), checkpoint_sha256=sha(checkpoint), protocol_sha256=sha(protocol_path),
        new_macro_updates=0, full_training_admitted=False)
    suffix = "_basic" if args.collect_basic_only else ""
    pipeline_path = REPORTS / f"admission067{suffix}_pipeline.json"
    write(pipeline_path, pipeline)
    try:
        reports, baselines = [], []
        for seed in protocol["required_confirmation_seeds"]:
            name = f"fresh_nominal067_{seed}"
            output = RUNS / name
            # 35 cases x 3 replicas x 2 actors = 210 native environments.
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
                "--output", str(output), "--variants", "parent", "robust", "--replicas", "3",
                "--seed", str(seed), "--domain-scale", "0", "--seconds", "12", "--isolated-origins"]
            if args.reuse_completed and output.exists():
                reuse_completed(name, command, output, checkpoint, ("parent", "robust"))
            else:
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
        result["lifecycle_complete"] = lifecycle["native_lifecycle_exercised_and_valid"]
        result["collect_basic_only"] = args.collect_basic_only
        write(REPORTS / f"admission067{suffix}_partial.json", result)
        if nominal["adequate"]:
            pipeline["phase"] = "long_standing"
            write(pipeline_path, pipeline)
            name = "standing_30s_reference067"
            output = RUNS / name
            command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
                "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
                "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "30", "--isolated-origins",
                "--case-names", "stand_low", "stand", "stand_high"]
            if args.reuse_completed and output.exists():
                reuse_completed(name, command, output, checkpoint, ("robust",))
            else:
                native(name, command, output)
                analyze(output / "robust")
            evidence = read(name)
            rows = standing(evidence)
            failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
            result.update(long_standing=rows, long_standing_failures=failures,
                          long_standing_trajectory_sha256=evidence[1]["trajectory_sha256"])
            result["long_standing_passed"] = not failures
            result["full_training_admitted"] = not failures and not args.collect_basic_only
        result["status"] = "simulation_admitted_for_long_training" if result["full_training_admitted"] else "needs_iteration"
        if args.collect_basic_only and result.get("long_standing_passed"):
            result["status"] = "basic_behavior_pass_lifecycle_pending"
        write(REPORTS / f"admission067{suffix}_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time(),
                        full_training_admitted=result["full_training_admitted"])
        write(pipeline_path, pipeline)
        print(json.dumps({k: result[k] for k in ("status", "full_training_admitted")}), flush=True)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(pipeline_path, pipeline)
        raise


if __name__ == "__main__":
    main()
