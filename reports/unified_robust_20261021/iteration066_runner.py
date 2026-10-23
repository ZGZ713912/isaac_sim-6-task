"""Fixed static test, then the unchanged full suite only after a static pass."""
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
from iteration036_compare import read, paired, standing
from iteration048_runner import static_failures


def evaluate(name, static):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint",
        str(RUNS / "quiet_high_recenter066/model_migrated.pt"), "--output", str(output),
        "--variants", "robust", "--replicas", "16" if static else "4", "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"]
    if static:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(name)


def main():
    pipeline = dict(schema="quiet066-pipeline-v1", status="running", phase="fixed_static",
                    started=time.time(), new_macro_updates=0, long_training_admitted=False)
    write(REPORTS / "iteration066_pipeline.json", pipeline)
    try:
        after = evaluate("high_recenter_static066", True)
        before, parent = read("quiet_velocity_baseline036"), read("joint_quiet_static064")
        rows = standing(after)
        failures = static_failures(rows, standing(before))
        assert not after[1]["evaluation_overrides"]
        # The low/mid actors and the complete native setup were retained.
        import numpy as np
        mask = [row["name"] != "stand_high" for row in after[1]["lanes"]]
        for key in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive"):
            assert np.array_equal(parent[3][key][:, mask], after[3][key][:, mask]), key
        result = dict(schema="quiet066-static-result-v1", pairing=paired(before, after),
            parent_pairing=paired(parent, after), low_mid_complete_traces_bitwise_exact=True,
            before=standing(before), parent=standing(parent), after=rows, static_failures=failures,
            checkpoint_sha256=after[1]["checkpoint_sha256"], trajectory_sha256=after[1]["trajectory_sha256"],
            new_macro_updates=0, lineage_clock=505, robust_consumed=1160, full_suite_complete=False,
            long_training_admitted=False, installed_in_rmcs=False)
        if not failures:
            pipeline["phase"] = "full_randomized_suite"
            write(REPORTS / "iteration066_pipeline.json", pipeline)
            full = evaluate("high_recenter_full066", False)
            result.update(full_suite_complete=True, full_suite_pairing=paired(read("side_confirmation032"), full),
                          full_analysis=full[2])
        result["status"] = "static_target_not_met" if failures else "requires_fresh_formal_admission"
        write(REPORTS / "iteration066_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time())
        write(REPORTS / "iteration066_pipeline.json", pipeline)
        print(json.dumps({k: result[k] for k in ("status", "after", "static_failures", "full_suite_complete")}), flush=True)
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        write(REPORTS / "iteration066_pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
