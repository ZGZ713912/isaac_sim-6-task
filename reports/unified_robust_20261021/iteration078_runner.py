"""No-PPO same-batch randomized static test of position-released wheel filtering."""
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
from wheeled_algo.evaluation_position_filter import position_filter_variants

CHECKPOINT = RUNS / "quiet_severe_reference065/model_migrated.pt"
NAME = "position_release_static078"
BANDS = [.01, .025, .025, .05]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    return dict(schema="quiet078-position-release-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        baseline="Frozen065 weights plus whole wheel alpha.25; exactly the074 policy function",
        variants=position_filter_variants(BANDS), alpha_min=.25, alpha_max=1.,
        position_source="obs14:16/20 from versioned sensor-only standing observer",
        environments=144, replicas_per_case_variant=16, seed=211322, seconds=12.,
        domain_scale=.35, ground_friction=[.3, .5], origin="isolated_overlap_v1",
        policy_hz=200, physics_hz=1000, protection_thresholds_changed=False,
        learned_parameters_changed=False, hidden_state_added=False,
        required="Original full5cm/air3mm and six actual3-95HF<=036;20-95 target separately reported",
        new_macro_updates=0, robust_consumed=1192, retained_lineage=505, full_training_admitted=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/evaluation_position_filter.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")})


def cpu_probe():
    import numpy as np
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.evaluation_position_filter import PositionReleasedWheelFilter
    from wheeled_algo.robust_getup import quiet_gate
    torch.set_num_threads(1)
    source = RUNS / "high_recenter_full066/robust/trajectory.npz"
    with np.load(source) as data:
        obs = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    mean, _ = load_mean(CHECKPOINT)
    previous, _ = load_mean(RUNS / "quiet_whole_wheel074/model_migrated.pt")
    checks = {}
    with torch.no_grad():
        original = mean(obs)
        nonquiet = quiet_gate(obs).squeeze(-1) == 0.
        rng = torch.get_rng_state()
        for name, band in position_filter_variants(BANDS).items():
            candidate = PositionReleasedWheelFilter(mean, band)
            action = candidate(obs)
            assert torch.equal(action[:, :4], original[:, :4])
            assert torch.equal(action[nonquiet], original[nonquiet])
            if band is None:
                assert torch.equal(action, previous(obs))
            checks[name] = dict(legs_exact=True, nonquiet_exact=True,
                extra_state_keys=[key for key in candidate.state_dict() if not key.startswith("mean.")],
                baseline074_exact=band is None)
            assert not checks[name]["extra_state_keys"]
        assert torch.equal(rng, torch.get_rng_state())
    return dict(schema="quiet078-cpu-scope-v1", frames=len(obs), nonquiet=int(nonquiet.sum()),
        recorded_trajectory_sha256=sha(source), checks=checks, rng_exact=True,
        tests_passed=36, native_behavior_proven=False)


def analyze(output):
    import numpy as np
    from iteration036_compare import paired, standing, read
    from iteration048_runner import static_failures
    baseline_rows = standing(read("quiet_velocity_baseline036"))
    evidence, results = {}, {}
    for name, band in position_filter_variants(BANDS).items():
        directory = output / name
        with (REPORTS / (NAME + "_" + name + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        metrics = json.loads((directory / "analysis.json").read_text())
        assert report["checkpoint_sha256"] == sha(CHECKPOINT)
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        assert report["quiet_wheel_position_release_band_m"] == band
        item = (directory, report, metrics, np.load(directory / "trajectory.npz"))
        evidence[name] = item
        rows = standing(item)
        failures = static_failures(rows, baseline_rows)
        for case, row in rows.items():
            q = metrics["cases"][case]["quiet"]
            row.update(leg_target_hf_rms_rad_20_95=q["leg_target_hf_rms_rad"],
                wheel_target_hf_rms_rad_s_20_95=q["wheel_target_hf_rms_rad_s"])
        results[name] = dict(band_m=band, standing=rows, static_failures=failures,
            trajectory_sha256=report["trajectory_sha256"], analysis_sha256=sha(directory / "analysis.json"))
    baseline = evidence["quiet_filter025"]
    result = dict(schema="quiet078-result-v1", status="completed", variants=results,
        initial_pairing={name: paired(baseline, item) for name, item in evidence.items()},
        baseline074_pairing=paired(read("quiet_whole_static074"), baseline),
        any_static_pass=any(not row["static_failures"] for row in results.values()),
        new_macro_updates=0, robust_consumed=1192, selected_lineage=505, full_training_admitted=False,
        scope="One randomized static seed; no motion/recovery or fresh nominal admission")
    write(REPORTS / "iteration078_final.json", result)
    print("078 RESULT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan_path = REPORTS / "iteration078_plan.json"
    value = plan()
    if args.prepare:
        write(REPORTS / "iteration078_cpu.json", cpu_probe())
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == value
        else:
            write(plan_path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(plan_path.read_text()) == value
    pipeline = dict(status="running", started=time.time(), phase="same_batch_static", new_macro_updates=0)
    write(REPORTS / "iteration078_pipeline.json", pipeline)
    try:
        output = RUNS / NAME
        native(NAME, [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
            "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
            "--case-names", "stand_low", "stand", "stand_high", "--quiet-wheel-position-release-bands",
            *[str(value) for value in BANDS]], output)
        analyze(output)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration078_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
