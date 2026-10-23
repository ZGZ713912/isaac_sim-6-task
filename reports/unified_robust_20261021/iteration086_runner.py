"""Zero-PPO nominal bracket of constant whole-wheel quiet filtering.

Includes a fresh alpha1 control in the same native batch. No gyro/position
release, no policy weight changes, and no automatic PPO follow-up.
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
ALPHAS = [1., .25, .32, .4]
NAME = "constant_wheel_static086"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    from evaluate_robust_getup_batch import quiet_filter_variants
    return dict(schema="quiet086-constant-bracket-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        hypothesis="A fixed alpha between073 .25 and .5 may reduce .25 startup overshoot without the time-varying gain that failed084",
        reason="074 fixed .25 lowered HF but high randomized excursion5.408cm;084 gyro release caused middle11.840cm/41.449mm clearance",
        variants=quiet_filter_variants(ALPHAS), seed=211322, replicas=4, environments=48,
        cases=["stand_low", "stand", "stand_high"], seconds=12, domain_scale=0., friction=[.5, .5],
        policy_hz=200, physics_hz=1000, native_origin_mode="isolated_overlap_v1",
        parent_control="alpha1 exact selected065 function in the same native batch",
        no_weight_changes=True, no_added_state=True, no_action_or_mechanical_guard_changes=True,
        new_macro_updates=0, robust_consumed=1384, adopted_lineage=505,
        prerequisite="083 training plus both evaluations fully completed; one native worker",
        full_training_admitted=False,
        selection="Report all alphas; any candidate needs equal-index serial random48 and fresh formal admission",
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/evaluation_quiet_output.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")})


def cpu_probe():
    import numpy as np
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter
    from wheeled_algo.robust_getup import quiet_gate
    torch.set_num_threads(1)
    source = RUNS / "high_recenter_full066/robust/trajectory.npz"
    with np.load(source) as saved:
        x = torch.from_numpy(saved["observations"].reshape(-1, 35)[::23].copy())
    mean, _ = load_mean(CHECKPOINT)
    with torch.inference_mode():
        base = mean(x)
        outside = quiet_gate(x).ravel() == 0.
        rng = torch.get_rng_state()
        checks = {}
        for alpha in ALPHAS:
            action = QuietWheelOutputFilter(mean, alpha)(x)
            item = dict(legs_exact=torch.equal(action[:, :4], base[:, :4]),
                nonquiet_exact=torch.equal(action[outside], base[outside]),
                alpha1_exact=torch.equal(action, base) if alpha == 1. else None,
                finite=bool(torch.isfinite(action).all()), wheel_bound=float(action[:, 4:].abs().max()))
            assert item["legs_exact"] and item["nonquiet_exact"] and item["finite"] and item["wheel_bound"] <= 9.
            if alpha == 1.:
                assert item["alpha1_exact"]
            checks[str(alpha)] = item
        assert torch.equal(rng, torch.get_rng_state())
    write(REPORTS / "iteration086_cpu.json", dict(schema="quiet086-scope-v1", frames=len(x),
        nonquiet_frames=int(outside.sum()), checks=checks, rng_unchanged=True,
        checkpoint_sha256=sha(CHECKPOINT), input_trajectory_sha256=sha(source),
        native_behavior_proven=False, new_macro_updates=0))


def analyze(output):
    import numpy as np
    from evaluate_robust_getup_batch import quiet_filter_variants
    from iteration036_compare import paired, standing
    evidence, rows = {}, {}
    for name, alpha in quiet_filter_variants(ALPHAS).items():
        directory = output / name
        with (REPORTS / (NAME + "_" + name + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        metrics = json.loads((directory / "analysis.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        assert report["checkpoint_sha256"] == sha(CHECKPOINT)
        assert report["quiet_wheel_output_alpha"] == alpha
        item = (directory, report, metrics, np.load(directory / "trajectory.npz"))
        evidence[name] = item
        static = standing(item)
        for case, row in static.items():
            quiet = metrics["cases"][case]["quiet"]
            row["leg_target_hf_rms_rad_20_95"] = quiet["leg_target_hf_rms_rad"]
            row["wheel_target_hf_rms_rad_s_20_95"] = quiet["wheel_target_hf_rms_rad_s"]
        failures = [case for case, row in static.items() if row["survived"] != row["episodes"]
            or row["full_window_max_excursion_m"] > .05 or row["wheel_clearance_after_02s_peak_m"] > .003]
        rows[name] = dict(alpha=alpha, standing=static, basic_failures=failures,
            trajectory_sha256=report["trajectory_sha256"], analysis_sha256=sha(directory / "analysis.json"))
    base_name = next(iter(rows))
    matching = {name: paired(evidence[base_name], item) for name, item in evidence.items()}
    for name, row in rows.items():
        improvements = [1. - values[key] / rows[base_name]["standing"][case][key]
            for case, values in row["standing"].items()
            for key in ("leg_target_hf_rms_rad_20_95", "wheel_target_hf_rms_rad_s_20_95")]
        row["target20_95_vs_same_batch_alpha1"] = dict(mean=float(np.mean(improvements)), all_six=improvements)
    value = dict(schema="quiet086-nominal-result-v1", status="completed", initial_pairing=matching,
        variants=rows, reference_alpha1=base_name, new_macro_updates=0, robust_consumed=1384,
        scope="One nominal condition; deterministic copies are not random coverage; comparison is to065, not formal1346 admission",
        full_training_admitted=False, candidate_adopted=False)
    write(REPORTS / "iteration086_final.json", value)
    print("086 RESULT", json.dumps({k: dict(alpha=v["alpha"], failures=v["basic_failures"],
        drift=[x["full_window_max_excursion_m"] for x in v["standing"].values()],
        target20_95=v["target20_95_vs_same_batch_alpha1"]["mean"]) for k, v in rows.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration086_plan.json"
    if args.prepare:
        cpu_probe()
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(path.read_text()) == value
    assert json.loads((REPORTS / "iteration083_pipeline.json").read_text())["status"] == "completed"
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *value["sources"],
        str(path.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT))], cwd=ROOT, check=True)
    output = RUNS / NAME
    pipeline = dict(status="running", phase="nominal_grid", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration086_pipeline.json", pipeline)
    try:
        native(NAME, [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "4", "--seed", "211322",
            "--domain-scale", "0", "--seconds", "12", "--isolated-origins",
            "--case-names", "stand_low", "stand", "stand_high", "--quiet-wheel-output-alphas", *map(str, ALPHAS)], output)
        analyze(output)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration086_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
