"""Full random pose/motion characterization of retained recovery096, no PPO."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write

CHECKPOINT = RUNS / "offline_recovery096/model_migrated.pt"
NAME = "recovery_matrix112"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    baseline = RUNS / "dynamic_recovery_full079/robust/report.json"
    old = json.loads(baseline.read_text())
    assert len(old["lanes"]) == 140 and old["seed"] == 211322 and old["seconds"] == 12.
    assert sha(CHECKPOINT) == "eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f"
    return dict(schema="recovery112-characterization-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        purpose="Separate actual fallen-birth recovery diversity from the known failed strong-right moving push; characterize motion retention before further recovery learning",
        baseline=str(baseline.relative_to(ROOT)), baseline_report_sha256=sha(baseline),
        seed=211322, replicas=4, environments=140, seconds=12., domain_scale=.35,
        friction_min=.3, friction_max=.5, actor_variant="robust", native_origin_mode="isolated_overlap_v1",
        cases="All frozen35 cases including asymmetric side and inverted poses, standing and simultaneous speed/yaw",
        scope="Existing096 as-is; old standing policy is not the106/110 position correction, so no unified qualification",
        known_failed_stress="096 nominal strong right moving push0/4 and0/8; left and inverted nominal successes do not prove arbitrary starts",
        pairing="All recorded initial domains/cases/materials exact079; verify all12 ordinary standing trajectories bitwise079",
        protocol_sha256=sha(REPORTS / "acceptance_protocol003.json"),
        new_macro_updates=0, new_offline_steps=0, candidate_ppo_lineage=569, candidate_offline_steps=64,
        robust_consumed=1384, actual_experiment_offline_steps=128, full_training_admitted=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "scripts/analyze_robust_getup.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_algo/quiet_output_actor.py",
            ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py", Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration112_plan.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(path.read_text()) == value
    assert json.loads((REPORTS / "iteration110_pipeline.json").read_text())["status"] == "completed"
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *value["sources"], str(path.relative_to(ROOT))], cwd=ROOT, check=True)
    pipeline = dict(status="running", phase="all35_native", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration112_pipeline.json", pipeline)
    try:
        output = RUNS / NAME
        native(NAME, [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "4", "--seed", "211322",
            "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"], output)
        pipeline["phase"] = "analysis"
        write(REPORTS / "iteration112_pipeline.json", pipeline)
        with (REPORTS / (NAME + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        import numpy as np
        from iteration036_compare import read, paired, standing
        old, new = read("dynamic_recovery_full079"), read(NAME)
        initial = paired(old, new)
        static = [row["lane"] for row in new[1]["lanes"] if row["name"] in ("stand_low", "stand", "stand_high")]
        exact = {key: np.array_equal(old[3][key][:, static], new[3][key][:, static])
                 for key in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")}
        assert len(static) == 12 and all(exact.values()), exact
        def summary(evidence):
            metrics = evidence[2]
            return dict(settled=metrics["actual_settled_attitudes"], cases=metrics["cases"],
                motion_survived=metrics["motion_survived"], motion_episodes=metrics["motion_episodes"],
                practical_moving_fall_recoveries=metrics["practical_moving_fall_recoveries"],
                closure_peak_m=metrics["max_gap_m"], passive_motor_peak_nm=metrics["passive_motor_peak_nm"],
                standing=standing(evidence))
        result = dict(schema="recovery112-characterization-result-v1", status="completed", initial_pairing=initial,
            frozen12_standing_trajectories_exact079=exact, before=summary(old), after=summary(new),
            checkpoint_sha256=sha(CHECKPOINT), trajectory_sha256=new[1]["trajectory_sha256"],
            new_macro_updates=0, new_offline_steps=0, robust_consumed=1384,
            full_training_admitted=False, candidate_adopted=False,
            limitation="One known-seed randomized batch, not independent heldout or unified quiet/recovery qualification")
        write(REPORTS / "iteration112_final.json", result)
        print("112 RESULT", json.dumps({key: result[key] for key in (
            "status", "checkpoint_sha256", "trajectory_sha256", "full_training_admitted")}
            | {"actual_settled": result["after"]["settled"], "motion": [result["after"]["motion_survived"], result["after"]["motion_episodes"]]}), flush=True)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration112_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
