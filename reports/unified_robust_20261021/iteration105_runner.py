"""Frozen existing protocol for composition096+097, without new learning."""
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

CHECKPOINT = RUNS / "composed_getup105/model_migrated.pt"
PROTOCOL = REPORTS / "acceptance_protocol003.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    migration = json.loads((REPORTS / "iteration105_migration.json").read_text())
    assert migration["checkpoint_sha256"] == sha(CHECKPOINT)
    return dict(schema="composed105-admission-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        migration_sha256=sha(REPORTS / "iteration105_migration.json"), protocol_sha256=sha(PROTOCOL),
        prerequisite="100 quiet30s and fresh48 static pass; exact097 quiet tensors/decoder and recorded-function equivalence",
        order=["randomized_full140", "nominal210_seed211131", "nominal210_seed211132", "stationary_lifecycle32"],
        random_job=dict(name="composed_full105", replicas=4, environments=140, seconds=12., seed=211322,
                        domain_scale=.35, friction_min=.3),
        nominal_jobs=[dict(name=f"composed_nominal105_{seed}", replicas=3, environments=210, seconds=12., seed=seed,
                           domain_scale=0., variants=["parent", "robust"]) for seed in (211131, 211132)],
        lifecycle_job=dict(name="composed_stationary105", replicas=16, environments=32, seconds=12.,
                           seed=211322, domain_scale=.35, friction_min=.3, stationary_push_cases=True),
        stop_after_failed_stage=True, new_macro_updates=0, new_offline_steps=0, cumulative_offline_steps=64,
        robust_consumed=1384, candidate_ppo_lineage=569, full_training_admitted=False,
        known_failed_stress="096 nominal strong moving right0/4 at27.324Ns; disclose regardless of aggregate pass",
        random_gates="036 six actualHF/full5cm/air3mm/allstaticalive;protocol003 family minimum2 and25%,arc errors,survival85%,one actual moving fall practical",
        lifecycle_gates="Each stationary side>=1 actual severe fall, valid reset/recapture, final2s95%stable with vx<.1/yaw<.2 and5cm sway;closure3mm/passive0",
        interpretation="Existing formal gates allow imperfect behavior; no requirement from the retired099/098 all12 diagnostic is silently rewritten",
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "scripts/analyze_robust_getup.py", ROOT / "src/wheeled_algo/quiet_output_actor.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py",
            Path(__file__))})


def evaluate(job, variants, stationary=False):
    output = RUNS / job["name"]
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
        "--output", str(output), "--variants", *variants, "--replicas", str(job["replicas"]), "--seed", str(job["seed"]),
        "--domain-scale", str(job["domain_scale"]), "--friction-min", str(job.get("friction_min", .5)),
        "--seconds", str(job["seconds"]), "--isolated-origins"]
    if stationary:
        command.append("--stationary-push-cases")
    native(job["name"], command, output)
    evidence = {}
    import numpy as np
    for variant in variants:
        directory = output / variant
        with (REPORTS / (job["name"] + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        metrics = json.loads((directory / "analysis.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        evidence[variant] = (directory, report, metrics, np.load(directory / "trajectory.npz"))
    return evidence


def common_gates(evidence, protocol, require_static_hf):
    from iteration036_compare import read, standing
    from iteration048_runner import static_failures
    directory, report, metrics, data = evidence
    rows = standing(evidence)
    failures = static_failures(rows, standing(read("quiet_velocity_baseline036"))) if require_static_hf else []
    if metrics["max_gap_m"] > .003 or metrics["passive_motor_peak_nm"] != 0.:
        failures.append("closed-chain or passive torque")
    for family, rate in protocol["family_min_complete_rate"].items():
        row = metrics["actual_settled_attitudes"].get(family, {})
        count = row.get(protocol["complete_metric"], 0)
        if count < 2 or count < rate * row.get("episodes", 0):
            failures.append(f"actual {family} practical recovery")
    if metrics["motion_survived"] < .85 * metrics["motion_episodes"]:
        failures.append("ordinary motion survival")
    for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right"):
        row = metrics["cases"][name]
        if row["vx_mae_m_s"] > .35 or row["yaw_mae_rad_s"] > .5:
            failures.append(name + " tracking")
    if metrics["practical_moving_fall_recoveries"] < 1:
        failures.append("actual moving fall practical recovery")
    return dict(evaluation=str(directory), trajectory_sha256=report["trajectory_sha256"], failures=failures,
        standing=rows, settled=metrics["actual_settled_attitudes"],
        actual_moving_fall_practical=metrics["practical_moving_fall_recoveries"],
        motion_survived=metrics["motion_survived"], motion_episodes=metrics["motion_episodes"],
        closure_peak_m=metrics["max_gap_m"], passive_motor_peak_nm=metrics["passive_motor_peak_nm"])


def lifecycle(evidence, task):
    import numpy as np
    from iteration065_analyze import reference_trace_violations
    directory, report, metrics, data = evidence
    rows, failures = [], []
    for row in report["lanes"]:
        lane = row["lane"]
        obs, alive, time = data["observations"][:, lane], data["alive"][:, lane], data["time"][:, lane]
        active = data["standing_reference_active"][:, lane]
        count = data["standing_reference_severe_reset_count"][:, lane]
        violations = reference_trace_violations(obs, active, count,
            data["standing_reference_age_s"][:, lane], data["standing_reference_filtered_m"][:, lane],
            data["standing_reference_anchor_m"][:, lane], alive, task)
        failures.extend(f"lane{lane}:{message}" for message in violations)
        captures = active & ~np.r_[False, active[:-1]] & (count > 0) & alive
        last = alive & (time > 10.)
        good = data["stable"][last, lane] & (np.abs(data["velocity"][last, lane, 0]) < .1)
        good &= np.abs(data["omega"][last, lane, 2]) < .2
        xy = data["local_position"][last, lane, :2]
        excursion = float(np.linalg.norm(xy - xy[:1], axis=-1).max()) if len(xy) else None
        before = active[980:1000].all()
        valid = bool(row["survived"] and row["in_motion_fall"] and before and captures.any()
                     and len(good) >= 399 and good.mean() >= .95 and excursion <= .05)
        rows.append(dict(lane=lane, name=row["name"], survived=row["survived"],
                         actual_fall=row["in_motion_fall"], reset_recapture_practical=valid))
    for name in ("stationary_push_left", "stationary_push_right"):
        if sum(row["reset_recapture_practical"] for row in rows if row["name"] == name) < 1:
            failures.append(name + " valid reset/recapture/stand")
    if metrics["max_gap_m"] > .003 or metrics["passive_motor_peak_nm"] != 0.:
        failures.append("lifecycle closed-chain or passive torque")
    return dict(rows=rows, failures=failures, trajectory_sha256=report["trajectory_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    plan_path = REPORTS / "iteration105_plan.json"
    if args.prepare:
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == value
        else:
            write(plan_path, value)
        print(json.dumps(value), flush=True)
        return
    assert json.loads(plan_path.read_text()) == value
    assert json.loads((REPORTS / "iteration100_pipeline.json").read_text())["status"] == "completed"
    assert json.loads((REPORTS / "iteration100_final.json").read_text())["static_confirmation_pass"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *value["sources"], str(plan_path.relative_to(ROOT))], cwd=ROOT, check=True)
    protocol = json.loads(PROTOCOL.read_text())
    pipeline = dict(status="running", phase="randomized_full140", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration105_pipeline.json", pipeline)
    result = dict(schema="composed105-admission-result-v1", checkpoint_sha256=sha(CHECKPOINT),
        full_training_admitted=False, candidate_adopted=False, new_macro_updates=0,
        new_offline_steps=0, cumulative_offline_steps=64, robust_consumed=1384,
        protocol_sha256=sha(PROTOCOL), inherited_static100_sha256=sha(REPORTS / "iteration100_final.json"))
    try:
        evidence = evaluate(value["random_job"], ["robust"])
        result["randomized_full"] = common_gates(evidence["robust"], protocol, True)
        if not result["randomized_full"]["failures"]:
            from export_robust_getup import assess
            from iteration036_compare import standing, paired
            reports, baselines, full = [], [], {}
            for job in value["nominal_jobs"]:
                pipeline["phase"] = job["name"]
                write(REPORTS / "iteration105_pipeline.json", pipeline)
                evidence = evaluate(job, ["parent", "robust"])
                paired(evidence["parent"], evidence["robust"])
                reports.append(evidence["robust"][0])
                baselines.append(evidence["parent"][0])
                full[str(job["seed"])] = standing(evidence["robust"])
            result["nominal"] = assess(reports, baselines, CHECKPOINT, protocol)
            result["nominal_full_window"] = full
            for seed, rows in full.items():
                for name, row in rows.items():
                    if row["full_window_max_excursion_m"] > .05 or row["wheel_clearance_after_02s_peak_m"] > .003:
                        result["nominal"]["failures"].append(f"seed{seed}:{name} full5cm/air3mm")
            result["nominal"]["adequate"] = not result["nominal"]["failures"]
            if result["nominal"]["adequate"]:
                pipeline["phase"] = "stationary_lifecycle"
                write(REPORTS / "iteration105_pipeline.json", pipeline)
                evidence = evaluate(value["lifecycle_job"], ["robust"], stationary=True)
                task = json.loads((RUNS / value["lifecycle_job"]["name"] / "task_contract.json").read_text())["settings"]
                result["lifecycle"] = lifecycle(evidence["robust"], task)
                result["full_training_admitted"] = not result["lifecycle"]["failures"]
        result["status"] = "simulation_admitted_for_long_training" if result["full_training_admitted"] else "needs_iteration"
        write(REPORTS / "iteration105_final.json", result)
        pipeline.update(status="completed", phase=result["status"], finished=time.time())
        print("105 RESULT", json.dumps(result), flush=True)
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration105_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
