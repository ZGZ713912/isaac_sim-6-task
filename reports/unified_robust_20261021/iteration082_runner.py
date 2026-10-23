"""Zero-PPO nominal test of gyro-released complete quiet wheel filtering.

Run only after079 and after the explicit diagnostic implementation is installed.
Randomized confirmation, policy migration and long PPO are separate decisions.
"""
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
NAME = "gyro_release_static082"
BANDS = [.05, .2, .03, .1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    from wheeled_algo.evaluation_gyro_filter import gyro_filter_variants
    return dict(schema="quiet082-gyro-release-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        variants=gyro_filter_variants(BANDS), alpha_min=.25, alpha_max=1.,
        release_sensor="norm(obs4:6)*2, roll/pitch gyro rad/s", hidden_state_added=False,
        reason="081 shows direct quiet neural correction has tiny20-95Hz contribution;074 complete wheel filtering reduces HF but incurs startup displacement",
        environments=36, replicas=4, seed=211322, domain_scale=0., seconds=12,
        cases=["stand_low", "stand", "stand_high"], ground_friction=[.5, .5],
        policy_hz=200, physics_hz=1000, origin="isolated_overlap_v1",
        weights_and_protections_changed=False, new_macro_updates=0, robust_consumed=1256,
        selected_lineage=505, full_training_admitted=False,
        decision="Nominal diagnostic only; every chosen band needs equal-index serial random48 confirmation, formal migration and fresh admission",
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/evaluation_gyro_filter.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")})


def analyze(output):
    import numpy as np
    from iteration036_compare import paired, standing
    from wheeled_algo.evaluation_gyro_filter import gyro_filter_variants
    parent_path = RUNS / "fresh_nominal067_211131/parent/analysis.json"
    parent = json.loads(parent_path.read_text())
    evidence, results = {}, {}
    for name, band in gyro_filter_variants(BANDS).items():
        directory = output / name
        with (REPORTS / (NAME + "_" + name + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        metrics = json.loads((directory / "analysis.json").read_text())
        assert report["checkpoint_sha256"] == sha(CHECKPOINT)
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        assert report["quiet_wheel_gyro_release_band_rad_s"] == band
        item = (directory, report, metrics, np.load(directory / "trajectory.npz"))
        evidence[name] = item
        rows = standing(item)
        improvements = []
        for case, row in rows.items():
            quiet, old = metrics["cases"][case]["quiet"], parent["cases"][case]["quiet"]
            for key in ("leg_target_hf_rms_rad", "wheel_target_hf_rms_rad_s"):
                row[key + "_20_95"] = quiet[key]
                row["parent_" + key + "_20_95"] = old[key]
                improvements.append(1. - quiet[key] / old[key])
        failures = [case for case, row in rows.items() if row["survived"] != row["episodes"]
            or row["full_window_max_excursion_m"] > .05 or row["wheel_clearance_after_02s_peak_m"] > .003]
        results[name] = dict(band_rad_s=band, standing=rows, basic_failures=failures,
            target20_95_mean_improvement=float(np.mean(improvements)),
            trajectory_sha256=report["trajectory_sha256"], analysis_sha256=sha(directory / "analysis.json"))
    baseline = evidence["quiet_filter025"]
    value = dict(schema="quiet082-nominal-result-v1", status="completed", variants=results,
        initial_pairing={name: paired(baseline, item) for name, item in evidence.items()},
        parent_reference=str(parent_path.relative_to(ROOT)),
        parent_limitation="067 separate nominal recording, not a fresh same-batch admission comparison",
        nominal_candidates=[name for name, row in results.items() if name != "quiet_filter025"
            and not row["basic_failures"] and row["target20_95_mean_improvement"] >= .25],
        new_macro_updates=0, robust_consumed=1256, full_training_admitted=False,
        scope="Nominal gyro diagnostic; repeated replicas do not add random coverage")
    write(REPORTS / "iteration082_final.json", value)
    print("082 RESULT", json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration082_plan.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(path.read_text()) == value
    assert json.loads((REPORTS / "iteration079_pipeline.json").read_text())["status"] == "completed"
    output = RUNS / NAME
    pipeline = dict(status="running", phase="nominal_grid", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration082_pipeline.json", pipeline)
    try:
        native(NAME, [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "4", "--seed", "211322",
            "--domain-scale", "0", "--seconds", "12", "--isolated-origins",
            "--case-names", "stand_low", "stand", "stand_high", "--quiet-wheel-gyro-release-bands",
            *map(str, BANDS)], output)
        analyze(output)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration082_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
