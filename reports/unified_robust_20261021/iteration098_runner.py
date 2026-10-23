"""Scope-gated native mean test after offline recovery098; no automatic PPO."""
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
from iteration045_runner import native, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state = json.loads((REPORTS / "iteration098_state.json").read_text())
    plan = json.loads((REPORTS / "iteration098_plan.json").read_text())
    assert state["scope_pass"] and state["actual_offline_steps"] == plan["new_offline_steps"] == 64
    assert json.loads((REPORTS / "iteration097_pipeline.json").read_text())["status"] == "completed"
    checkpoint = RUNS / "offline_recovery098/model_migrated.pt"
    parent = RUNS / "offline_recovery096/model_migrated.pt"
    assert sha(checkpoint) == state["checkpoint_sha256"] and sha(parent) == plan["parent_sha256"]
    assert all(sha(ROOT / relative) == expected for relative, expected in plan["implementation_sha256"].items())
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *plan["implementation_sha256"],
                    str(Path(__file__).relative_to(ROOT))], cwd=ROOT, check=True)
    pipeline = dict(status="running", phase="paired_mean_recovery", started=time.time(),
                    new_macro_updates=0, actual_offline_steps=64)
    write(REPORTS / "iteration098_pipeline.json", pipeline)
    try:
        output = RUNS / "offline_recovery_target098"
        native("offline_recovery_target098", [sys.executable, "scripts/evaluate_robust_getup_batch.py",
            "--checkpoint", str(checkpoint), "--reference-checkpoint", str(parent), "--output", str(output),
            "--variants", "robust", "reference", "--replicas", "4", "--seed", "211322",
            "--domain-scale", "0", "--friction-min", ".5", "--seconds", "12", "--isolated-origins",
            "--case-names", "moving_push_left", "moving_push_right", "inverted_asymmetric"], output)
        import numpy as np
        from iteration036_compare import paired
        evidence, rows = {}, {}
        for name in ("reference", "robust"):
            directory = output / name
            with (REPORTS / ("offline_recovery_target098_" + name + "_analysis.log")).open("w") as log:
                subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((directory / "report.json").read_text())
            metrics = json.loads((directory / "analysis.json").read_text())
            assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
            data = np.load(directory / "trajectory.npz")
            evidence[name] = (directory, report, metrics, data)
            cases = {}
            for case in ("moving_push_left", "moving_push_right", "inverted_asymmetric"):
                ids = [row["lane"] for row in report["lanes"] if row["name"] == case]
                cases[case] = dict(episodes=len(ids),
                    recovered=sum(report["lanes"][i]["recovered"] for i in ids),
                    practical=sum(metrics["practical_followup_by_lane"][i] for i in ids),
                    contact_impulse_peak_ns=max(report["lanes"][i]["max_contact_impulse_ns"] for i in ids))
            rows[name] = dict(cases=cases, closure_peak_m=metrics["max_gap_m"],
                              passive_motor_peak_nm=metrics["passive_motor_peak_nm"],
                              trajectory_sha256=report["trajectory_sha256"])
        matching = paired(evidence["reference"], evidence["robust"])
        moving = [row["lane"] for row in evidence["robust"][1]["lanes"] if row["name"].startswith("moving_")]
        prepush = {key: np.array_equal(evidence["reference"][3][key][:1000, moving],
                                      evidence["robust"][3][key][:1000, moving])
                   for key in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")}
        assert all(prepush.values()), prepush
        candidate = rows["robust"]
        passed = (all(row["practical"] == row["episodes"] == 4 for row in candidate["cases"].values())
                  and candidate["closure_peak_m"] < .003 and candidate["passive_motor_peak_nm"] == 0.)
        result = dict(schema="offline-recovery098-native-result-v1", status="completed",
            initial_pairing=matching, moving_prepush1000frames_exact=prepush,
            results=rows, nominal_diagnostic_pass=passed,
            checkpoint_sha256=sha(checkpoint), state_audit_sha256=sha(REPORTS / "iteration098_state.json"),
            new_macro_updates=0, actual_offline_steps=64, cumulative_offline_steps=128, robust_consumed=1384,
            full_training_admitted=False, candidate_adopted=False,
            scope="Nominal diagnostic only;24 lanes include repeated deterministic cases, not heldout diversity")
        write(REPORTS / "iteration098_final.json", result)
        print("098 RESULT", json.dumps(result), flush=True)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration098_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
