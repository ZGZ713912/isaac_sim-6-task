"""Random48 gyro-filter confirmation with the same environment indices as074."""
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
from iteration045_runner import native, write

CHECKPOINT = RUNS / "quiet_severe_reference065/model_migrated.pt"
NAME = "gyro_release_random084"
BAND = [.05, .2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    return dict(schema="quiet084-serial-random-confirmation-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        gyro_band_rad_s=BAND, alpha_min=.25, baseline="quiet_whole_static074, fixedalpha.25 on the same065 weights",
        baseline_process_sha256=sha(REPORTS / "quiet_whole_static074_process.json"),
        num_envs=48, replicas=16, seconds=12, seed=211322, domain_scale=.35,
        ground_friction=[.3, .5], policy_hz=200, physics_hz=1000, origin="isolated_overlap_v1",
        randomized_variant_batching=False, required="Strict initial pairing; full5cm/air3mm/all static survive and six actual3-95HF<=036",
        new_macro_updates=0, robust_consumed=1256, full_training_admitted=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/evaluation_gyro_filter.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")})


def analyze(output):
    from iteration036_compare import read, paired, standing
    from iteration048_runner import static_failures
    with (REPORTS / (NAME + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    before, after = read("quiet_whole_static074"), read(NAME)
    assert after[1]["checkpoint_sha256"] == sha(CHECKPOINT)
    assert after[1]["quiet_wheel_gyro_release_band_rad_s"] == BAND
    initial = paired(before, after)
    rows = standing(after)
    failures = static_failures(rows, standing(read("quiet_velocity_baseline036")))
    for case, row in rows.items():
        for key in ("leg_target_hf_rms_rad", "wheel_target_hf_rms_rad_s"):
            row[key + "_20_95"] = after[2]["cases"][case]["quiet"][key]
    result = dict(schema="quiet084-serial-random-result-v1", status="completed", initial_pairing=initial,
        before=standing(before), after=rows, static_failures=failures, static_pass=not failures,
        trajectory_sha256=after[1]["trajectory_sha256"], analysis_sha256=sha(output / "robust/analysis.json"),
        new_macro_updates=0, robust_consumed=1256, full_training_admitted=False,
        scope="One paired randomized static seed; formal policy migration and fresh two-seed admission still required")
    write(REPORTS / "iteration084_final.json", result)
    print("084 RESULT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration084_plan.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(path.read_text()) == value
    assert json.loads((REPORTS / "iteration082_pipeline.json").read_text())["status"] == "completed"
    output = RUNS / NAME
    pipeline = dict(status="running", phase="same_index_random_static", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration084_pipeline.json", pipeline)
    try:
        native(NAME, [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
            "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
            "--case-names", "stand_low", "stand", "stand_high", "--quiet-wheel-gyro-release-band", *map(str, BAND)], output)
        analyze(output)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration084_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
