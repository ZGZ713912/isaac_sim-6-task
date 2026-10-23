"""Serial zero-learning admission of079 recovery plus110 standing, protocol004."""
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

CHECKPOINT = RUNS / "composed_getup115/model_migrated.pt"
PROTOCOL = REPORTS / "acceptance_protocol004.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    migration = json.loads((REPORTS / "iteration115_migration.json").read_text())
    assert migration["checkpoint_sha256"] == sha(CHECKPOINT)
    return dict(schema="composed115-admission-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        migration_sha256=sha(REPORTS / "iteration115_migration.json"), protocol_sha256=sha(PROTOCOL),
        reason="User high10cm+recenter; withdraw114 without training.112 rejects096 generalization; retain079 recovery with110 static function.",
        static_jobs=[dict(name="composed_static_original115", replicas=16, seconds=30., seed=211322,
                          domain_scale=.35, friction_min=.3),
                     dict(name="composed_static_holdout115", replicas=16, seconds=30., seed=211626,
                          domain_scale=.35, friction_min=.3)],
        random_job=dict(name="composed_full115", replicas=4, seconds=12., seed=211322, domain_scale=.35, friction_min=.3),
        nominal_jobs=[dict(name=f"composed_nominal115_{seed}", replicas=3, seconds=12., seed=seed,
                           domain_scale=0., friction_min=.5) for seed in (211131, 211132)],
        lifecycle_job=dict(name="composed_stationary115", replicas=16, seconds=12., seed=211322, domain_scale=.35, friction_min=.3),
        order="two30s static, full random140, two matched nominal210, stationary reset/recapture32; stop at failed stage",
        static_gates="all48alive; low/mid5cm,high10cm and tail recenter/no sustained outward drift;6 actualHF<=036 and post.2s air<=3mm",
        common_gates="protocol004 actual-settled family minimum2 and25%,85% motion, arc MAEs, one actual moving-fall practical, closure3mm/passive0",
        lifecycle_gates="one actual severe fall+valid reset/recapture+terminal stand per side, unmodified105 lifecycle helper",
        original_static_retention="original115 middle16 full trajectories bitwise100 original30s; complete first115200 quiet inputs exact110 verifiedCPU",
        heldout_seed_status="211626 unused by standing tuning before this plan; becomes observed after this run",
        known_failed_stress="079 nominal diagnostic right strong moving push0/16; do not hide even if aggregate admission passes",
        new_macro_updates=0, new_offline_steps=0, candidate_ppo_lineage=569, candidate_offline_steps=0,
        robust_consumed=1384, actual_experiment_offline_steps=128, full_training_admitted=False,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "scripts/analyze_robust_getup.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "src/wheeled_algo/standing_acceptance.py", ROOT / "src/wheeled_algo/quiet_output_actor.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py",
            REPORTS / "iteration105_runner.py", REPORTS / "iteration036_compare.py", REPORTS / "iteration115_merge.py", Path(__file__))})


def evaluate(job, variants, *, static=False, stationary=False):
    import numpy as np
    output = RUNS / job["name"]
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
        "--output", str(output), "--variants", *variants, "--replicas", str(job["replicas"]), "--seed", str(job["seed"]),
        "--domain-scale", str(job["domain_scale"]), "--friction-min", str(job["friction_min"]),
        "--seconds", str(job["seconds"]), "--isolated-origins"]
    if static:
        command += ["--case-names", "stand_low", "stand", "stand_high"]
    if stationary:
        command += ["--stationary-push-cases"]
    native(job["name"], command, output)
    evidence = {}
    for variant in variants:
        directory = output / variant
        with (REPORTS / (job["name"] + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        evidence[variant] = (directory, report, json.loads((directory / "analysis.json").read_text()),
                             np.load(directory / "trajectory.npz"))
    return evidence


def static_failures(evidence, protocol, *, actual_hf):
    from iteration036_compare import standing
    from wheeled_algo.standing_acceptance import excursion_limit, ACTUAL_HF_FIELDS
    rows, failures = standing(evidence), []
    baseline = protocol["long_standing_confirmation"]["actual_hf_baseline"]["by_case"]
    for case, row in rows.items():
        if row["survived"] != row["episodes"]:
            failures.append(case + " static termination")
        if row["full_window_max_excursion_m"] > excursion_limit(protocol, case):
            failures.append(case + " full-window excursion")
        if row["wheel_clearance_after_02s_peak_m"] > .003:
            failures.append(case + " wheel lift")
        if actual_hf:
            failures.extend(case + " actual vibration " + key for key in ACTUAL_HF_FIELDS if row[key] > baseline[case][key])
    return rows, failures


def common_gates(evidence, protocol):
    metrics = evidence[2]
    rows, failures = static_failures(evidence, protocol, actual_hf=True)
    if metrics["max_gap_m"] > .003 or metrics["passive_motor_peak_nm"] != 0.:
        failures.append("closed-chain or passive torque")
    for family, rate in protocol["family_min_complete_rate"].items():
        row = metrics["actual_settled_attitudes"].get(family, {})
        count = row.get(protocol["complete_metric"], 0)
        if count < protocol["minimum_successful_episodes_per_family"] or count < rate * row.get("episodes", 0):
            failures.append("actual " + family + " practical recovery")
    if metrics["motion_survived"] < protocol["ordinary_motion_survival_rate_min"] * metrics["motion_episodes"]:
        failures.append("ordinary motion survival")
    for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right"):
        row = metrics["cases"][name]
        if row["vx_mae_m_s"] > protocol["ordinary_arc_vx_mae_max_m_s"] or row["yaw_mae_rad_s"] > protocol["ordinary_arc_yaw_mae_max_rad_s"]:
            failures.append(name + " tracking")
    if metrics["practical_moving_fall_recoveries"] < 1:
        failures.append("actual moving fall practical recovery")
    return dict(failures=failures, standing=rows, settled=metrics["actual_settled_attitudes"],
        motion_survived=metrics["motion_survived"], motion_episodes=metrics["motion_episodes"],
        actual_moving_fall_practical=metrics["practical_moving_fall_recoveries"],
        closure_peak_m=metrics["max_gap_m"], passive_motor_peak_nm=metrics["passive_motor_peak_nm"],
        trajectory_sha256=evidence[1]["trajectory_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration115_plan.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(path.read_text()) == value
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *value["source_sha256"], str(path.relative_to(ROOT)),
                    str(PROTOCOL.relative_to(ROOT))], cwd=ROOT, check=True)
    import numpy as np
    from export_robust_getup import assess, assess_extended_standing
    from iteration036_compare import paired, read, standing
    from iteration105_runner import lifecycle
    from wheeled_algo.standing_acceptance import review_extended_standing
    from torch.utils.tensorboard import SummaryWriter
    protocol = json.loads(PROTOCOL.read_text())
    pipeline = dict(status="running", phase="long_standing", started=time.time(), new_macro_updates=0)
    result = dict(schema="composed115-admission-result-v1", checkpoint_sha256=sha(CHECKPOINT), protocol_sha256=sha(PROTOCOL),
        static=[], new_macro_updates=0, new_offline_steps=0, robust_consumed=1384, actual_experiment_offline_steps=128,
        candidate_ppo_lineage=569, candidate_adopted=False, full_training_admitted=False)
    write(REPORTS / "iteration115_pipeline.json", pipeline)
    def seal_progress():
        write(REPORTS / "iteration115_partial.json", result)
    try:
        static_dirs = []
        for job in value["static_jobs"]:
            pipeline["phase"] = job["name"]
            write(REPORTS / "iteration115_pipeline.json", pipeline)
            e = evaluate(job, ["robust"], static=True)["robust"]
            row = review_extended_standing(e[1], e[2], e[3], protocol)
            row.update(**job, trajectory_sha256=e[1]["trajectory_sha256"], standing_summary=standing(e))
            if job["seed"] == 211322:
                old = read("quiet_standing30s100")
                row["initial_pairing"] = paired(old, e)
                lanes = [r["lane"] for r in e[1]["lanes"] if r["name"] == "stand"]
                row["middle16_trajectories_bitwise100"] = {k: np.array_equal(old[3][k][:, lanes], e[3][k][:, lanes])
                    for k in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")}
                assert all(row["middle16_trajectories_bitwise100"].values())
            result["static"].append(row)
            writer = SummaryWriter(str(e[0].parent), filename_suffix=".standing_recenter115")
            for case, metrics in row["standing"].items():
                writer.add_scalar(f"StandingEnvelope/{case}/limit_m", metrics["excursion_limit_m"], 115)
                writer.add_scalar(f"StandingEnvelope/{case}/max_excursion_m", max(x["full_window_excursion_m"] for x in metrics["episodes"]), 115)
                if case == "stand_high":
                    writer.add_scalar("StandingRecenter/stand_high/tail_mean_radius_max_m", max(x["recentering"]["tail_mean_radius_m"] for x in metrics["episodes"]), 115)
                    writer.add_scalar("StandingRecenter/stand_high/tail_outward_slope_max_m_s", max(x["recentering"]["tail_outward_radial_slope_m_s"] for x in metrics["episodes"]), 115)
            writer.close()
            static_dirs.append(e[0])
            seal_progress()
            print("115 STATIC", json.dumps(dict(name=job["name"], adequate=row["adequate"], failures=row["failures"], standing=row["standing_summary"])), flush=True)
            if row["failures"]:
                break
        static_ok = len(result["static"]) == 2 and all(r["adequate"] for r in result["static"])
        if static_ok:
            result["extended_standing"] = assess_extended_standing(static_dirs, sha(CHECKPOINT), protocol)
            assert result["extended_standing"]["adequate"]
            pipeline["phase"] = "randomized_full140"
            write(REPORTS / "iteration115_pipeline.json", pipeline)
            e = evaluate(value["random_job"], ["robust"])["robust"]
            result["randomized_full"] = common_gates(e, protocol)
            result["randomized_full"]["initial_pairing079"] = paired(read("dynamic_recovery_full079"), e)
            seal_progress()
            if not result["randomized_full"]["failures"]:
                reports, baselines, nominal_full = [], [], {}
                for job in value["nominal_jobs"]:
                    pipeline["phase"] = job["name"]
                    write(REPORTS / "iteration115_pipeline.json", pipeline)
                    e = evaluate(job, ["parent", "robust"])
                    paired(e["parent"], e["robust"])
                    reports.append(e["robust"][0])
                    baselines.append(e["parent"][0])
                    nominal_full[str(job["seed"])] = static_failures(e["robust"], protocol, actual_hf=False)
                result["nominal"] = assess(reports, baselines, CHECKPOINT, protocol, standing_evaluations=static_dirs)
                result["nominal_full_window"] = nominal_full
                for seed, (rows, failures) in nominal_full.items():
                    result["nominal"]["failures"].extend(f"seed{seed}:{x}" for x in failures)
                result["nominal"]["adequate"] = not result["nominal"]["failures"]
                seal_progress()
                if result["nominal"]["adequate"]:
                    pipeline["phase"] = "stationary_lifecycle"
                    write(REPORTS / "iteration115_pipeline.json", pipeline)
                    e = evaluate(value["lifecycle_job"], ["robust"], stationary=True)["robust"]
                    task = json.loads((e[0].parent / "task_contract.json").read_text())["settings"]
                    result["lifecycle"] = lifecycle(e, task)
                    result["full_training_admitted"] = not result["lifecycle"]["failures"]
        result["status"] = "simulation_admitted_for_long_training" if result["full_training_admitted"] else "needs_iteration"
        write(REPORTS / "iteration115_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time())
        print("115 RESULT", json.dumps({k: result[k] for k in ("status", "checkpoint_sha256", "full_training_admitted")}), flush=True)
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration115_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
