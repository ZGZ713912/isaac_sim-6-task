"""Zero-PPO matched nominal grid of complete quiet wheel output filtering."""
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
from evaluate_robust_getup_batch import quiet_filter_variants
from iteration045_runner import native, write

ALPHAS = [1., .5, .25, .1]
NAME = "quiet_whole_wheel_grid073"
CHECKPOINT = RUNS / "quiet_severe_reference065/model_migrated.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan():
    return dict(schema="quiet073-nominal-grid-plan-v1", status="frozen_before_native",
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha(CHECKPOINT),
        seed=211322, replicas=4, environments=48, seconds=12, domain_scale=0.,
        policy_hz=200, physics_hz=1000, native_origin_mode="isolated_overlap_v1",
        variants=quiet_filter_variants(ALPHAS), cases=["stand_low", "stand", "stand_high"],
        mechanism="Full quiet wheel output IIR from previous APPLIED action; learned correction is inside this filter",
        guards=["original alpha1 control", "all static alive", "full excursion<=.05m", "post0.2s air<=.003m",
                "report20-95Hz and3-95Hz target and actual motion", "compare exact1346 parent from067"],
        sources={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/evaluation_quiet_output.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")},
        new_macro_updates=0, selected_lineage=505, robust_consumed=1160,
        long_training_admitted=False, deployment_package_replaced=False,
        interpretation="Diagnostic coefficients only; any selected coefficient needs randomized closed-loop confirmation and a formal actor migration")


def cpu_probe():
    import numpy as np
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter
    from wheeled_algo.robust_getup import quiet_gate
    torch.set_num_threads(1)
    source = RUNS / "high_recenter_full066/robust/trajectory.npz"
    with np.load(source) as saved:
        obs = torch.from_numpy(saved["observations"].reshape(-1, 35)[::23].copy())
    mean, _ = load_mean(CHECKPOINT)
    with torch.no_grad():
        action = mean(obs)
        outside = quiet_gate(obs).squeeze(-1) == 0.
        rng = torch.get_rng_state()
        checks = {}
        for alpha in ALPHAS:
            actual = QuietWheelOutputFilter(mean, alpha)(obs)
            checks[str(alpha)] = dict(legs_exact=torch.equal(actual[:, :4], action[:, :4]),
                nonquiet_exact=torch.equal(actual[outside], action[outside]),
                alpha1_all_exact=torch.equal(actual, action) if alpha == 1. else None,
                wheel_raw_bound=float(actual[:, 4:].abs().max()))
            assert checks[str(alpha)]["legs_exact"] and checks[str(alpha)]["nonquiet_exact"]
        assert checks["1.0"]["alpha1_all_exact"] and torch.equal(rng, torch.get_rng_state())
    receipt = dict(schema="quiet073-recorded-input-scope-v1", checkpoint_sha256=sha(CHECKPOINT),
        recorded_input_trajectory_sha256=sha(source), frames=len(obs), nonquiet_frames=int(outside.sum()),
        checks=checks, rng_unchanged=True, native_closed_loop_proven=False,
        unit_tests=dict(passed=18, log_sha256=sha(REPORTS / "iteration073_tests.log")))
    write(REPORTS / "iteration073_cpu.json", receipt)


def analyze(output):
    import numpy as np
    from iteration036_compare import standing, paired
    evidence = {}
    rows = {}
    parent = json.loads((RUNS / "fresh_nominal067_211131/parent/analysis.json").read_text())
    for variant, alpha in quiet_filter_variants(ALPHAS).items():
        directory = output / variant
        with (REPORTS / (NAME + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        analysis = json.loads((directory / "analysis.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        assert report["checkpoint_sha256"] == sha(CHECKPOINT)
        assert report["quiet_wheel_output_alpha"] == alpha
        item = (directory, report, analysis, np.load(directory / "trajectory.npz"))
        evidence[variant] = item
        static = standing(item)
        for case, row in static.items():
            quiet = analysis["cases"][case]["quiet"]
            base = parent["cases"][case]["quiet"]
            row.update(leg_target_hf_rms_rad_20_95=quiet["leg_target_hf_rms_rad"],
                       wheel_target_hf_rms_rad_s_20_95=quiet["wheel_target_hf_rms_rad_s"],
                       parent_leg_target_hf_rms_rad_20_95=base["leg_target_hf_rms_rad"],
                       parent_wheel_target_hf_rms_rad_s_20_95=base["wheel_target_hf_rms_rad_s"])
        failures = [case for case, row in static.items() if row["episodes"] != row["survived"]
                    or row["full_window_max_excursion_m"] > .05
                    or row["wheel_clearance_after_02s_peak_m"] > .003]
        target_improvements = [1. - row[k] / row["parent_" + k] for row in static.values()
                               for k in ("leg_target_hf_rms_rad_20_95", "wheel_target_hf_rms_rad_s_20_95")]
        rows[variant] = dict(alpha=alpha, standing=static, basic_failures=failures,
            target20_95_mean_improvement=float(np.mean(target_improvements)),
            target20_95_all_six_improvements=target_improvements,
            trajectory_sha256=report["trajectory_sha256"], analysis_sha256=sha(directory / "analysis.json"))
    baseline = evidence[next(iter(evidence))]
    pairing = {name: paired(baseline, item) for name, item in evidence.items()}
    result = dict(schema="quiet073-native-grid-result-v1", status="completed", variants=rows,
        matching_initial_conditions=pairing, nominal_parent_reference="fresh_nominal067_211131/parent",
        nominal_parent_note="067 nominal standing was identical across its two seeds; this grid is one nominal setting, not randomized coverage",
        new_macro_updates=0, robust_consumed=1160, full_training_admitted=False)
    write(REPORTS / "iteration073_final.json", result)
    print(json.dumps({key: dict(alpha=row["alpha"], failures=row["basic_failures"],
                               target_improvement=row["target20_95_mean_improvement"],
                               drift=[x["full_window_max_excursion_m"] for x in row["standing"].values()])
                      for key, row in rows.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    path = REPORTS / "iteration073_plan.json"
    if args.prepare:
        cpu_probe()
        if path.exists():
            assert json.loads(path.read_text()) == plan
        else:
            write(path, plan)
        print(json.dumps(plan), flush=True)
        return
    assert json.loads(path.read_text()) == plan
    pipeline = dict(status="running", phase="nominal_grid", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration073_pipeline.json", pipeline)
    try:
        output = RUNS / NAME
        command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "4", "--seed", "211322",
            "--domain-scale", "0", "--seconds", "12", "--isolated-origins",
            "--case-names", "stand_low", "stand", "stand_high", "--quiet-wheel-output-alphas", *map(str, ALPHAS)]
        native(NAME, command, output)
        analyze(output)
        pipeline.update(status="completed", phase="grid_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed", finished=time.time(), error=traceback.format_exc())
        raise
    finally:
        write(REPORTS / "iteration073_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
