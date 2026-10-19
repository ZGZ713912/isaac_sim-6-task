#!/usr/bin/env python3
"""Run bounded 64-update pilots with serialized native behavior comparisons.

This development loop stops by update 512. It never starts a long training
block, changes the plan, retries failed workers, or accesses robot hardware.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compare_flat_base import compare
from train_flat_base import validate_large_block_evidence
from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan


def assess(checkpoint, basic, envelope, deployment, sources, source_sha, previous=None):
    checkpoint_sha = digest(checkpoint)
    for candidate in (basic, envelope, deployment):
        if candidate.get("status") != "evaluated" or candidate.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Development assessment must belong to the exact completed checkpoint")
    comparisons = [compare(source, candidate) for source, candidate in zip(sources, (basic, envelope, deployment))]
    retained = all(item["no_lost_baseline_checks"] for item in comparisons)
    original_tracking_regressions = []
    # The original may miss an absolute precision target. That does not permit
    # a worse high-speed policy to pass preservation simply because both fail.
    for cohort, source, candidate in zip(("basic", "envelope", "deployment"), sources,
                                         (basic, envelope, deployment)):
        for name, case in candidate["result"]["cases"].items():
            vx, yaw, _ = case["command"]
            if abs(vx) < .75 and abs(yaw) < 1.5:
                continue
            before = source["result"]["cases"][name]
            for metric, tolerance in (("vx_mae_m_s", .02), ("yaw_mae_rad_s", .05)):
                if case[metric] > before[metric] + tolerance:
                    original_tracking_regressions.append(dict(cohort=cohort, case=name, metric=metric,
                        before=before[metric], after=case[metric], tolerance=tolerance))
    regressions = []
    if previous is not None:
        for cohort, old, new in zip(("basic", "envelope", "deployment"), previous, (basic, envelope, deployment)):
            paired = compare(old, new)
            regressions.extend(dict(cohort=cohort, **item) for item in paired["lost_baseline_checks"])
            for name, case in new["result"]["cases"].items():
                before = old["result"]["cases"][name]
                tolerances = dict(vx_mae_m_s=.02, yaw_mae_rad_s=.05)
                if case["command"][:2] == [0., 0.]:
                    tolerances["stand_drift_max_m"] = .03
                for metric, tolerance in tolerances.items():
                    if case[metric] > before[metric] + tolerance:
                        regressions.append(dict(cohort=cohort, case=name, metric=metric,
                            before=before[metric], after=case[metric], tolerance=tolerance))
    try:
        validate_large_block_evidence(basic, envelope, sources[1], checkpoint_sha, source_sha)
        scale_reason = None if deployment["result"]["passed"] else "1000Hz basic behavior targets remain incomplete"
    except ValueError as error:
        scale_reason = str(error)
    return dict(no_lost_original_checks=retained,
        no_original_tracking_regressions=not original_tracking_regressions,
        original_tracking_regressions=original_tracking_regressions,
        no_previous_regressions=not regressions, previous_regressions=regressions,
        passed_cases=[item["passed_cases"] for item in comparisons],
        scale_candidate=retained and not original_tracking_regressions and not regressions and scale_reason is None,
        scale_block_reason=scale_reason,
        comparisons=comparisons, hardware_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--basic", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ceiling", type=int, default=512)
    parser.add_argument("--reference-basic", type=Path)
    parser.add_argument("--reference-envelope", type=Path)
    parser.add_argument("--reference-deployment", type=Path)
    args = parser.parse_args()
    if not 1 <= args.ceiling <= 512:
        parser.error("The short-pilot development ceiling cannot exceed 512")
    references = [args.reference_basic, args.reference_envelope, args.reference_deployment]
    if any(references) and not all(references):
        parser.error("A reference candidate requires all three matched evaluation cohorts")
    plan = read_plan(ROOT, args.plan)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (ROOT / "runs/flat_base_20261019/.iteration.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = dict(schema="flat-bounded-pilot-iteration-v1", status="assessing", pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), ceiling=args.ceiling, plan_sha256=digest(ROOT / args.plan),
        history=[], training_release=False, hardware_motion_started=False)

    def save():
        temporary = args.output / "campaign.tmp"
        temporary.write_bytes(encode(report))
        temporary.replace(args.output / "campaign.json")

    def run(command, label):
        with (args.output / (label + ".log")).open("w") as log:
            env = os.environ.copy()
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            report.update(worker_pid=process.pid, phase=label)
            save()
            result = process.wait()
        report.pop("worker_pid", None)
        if result:
            raise RuntimeError(f"{label} exited {result}; the failed worker will not be retried")

    source_paths = [ROOT / "runs/flat_base_20261019" / name / "evaluation.json" for name in
                    ("baseline_native002", "envelope_source002", "deployment_source001")]
    sources = [json.loads(path.read_text()) for path in source_paths]
    report["source_evaluation_sha256"] = {str(path): digest(path) for path in source_paths}
    checkpoint, evaluations = args.checkpoint.resolve(), [args.basic.resolve(), args.envelope.resolve(), args.deployment.resolve()]
    previous = [json.loads(path.read_text()) for path in references] if all(references) else None
    report["initial_reference_evaluation_sha256"] = {str(path): digest(path) for path in references if path}
    try:
        import torch
        while True:
            info = torch.load(checkpoint, map_location="cpu", weights_only=True)["infos"]
            if info["plan_sha256"] != report["plan_sha256"]:
                raise ValueError("Iteration cannot change the candidate's training plan")
            current = [json.loads(path.read_text()) for path in evaluations]
            assessment = assess(checkpoint, *current, sources, plan["source_onnx_sha256"], previous=previous)
            report["history"].append(dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                successful_updates_total=info["successful_updates_total"], actor_updates_total=info["actor_updates_total"],
                evaluations=[str(p) for p in evaluations], assessment=assessment))
            save()
            if (not assessment["no_lost_original_checks"] or not assessment["no_original_tracking_regressions"]
                    or not assessment["no_previous_regressions"]):
                report["status"] = "regression_stopped"
                break
            if assessment["scale_candidate"]:
                report["status"] = "ready_for_scale_review"
                break
            count = info["successful_updates_total"]
            if count >= args.ceiling:
                report["status"] = "small_budget_complete"
                break
            additional = min(64, args.ceiling - count)
            previous = current
            run_dir = ROOT / "runs/flat_base_20261019" / f"iter_{count + additional:06d}"
            report["status"] = "training"
            run(["scripts/train_flat_base.py", "--plan", args.plan, "--resume", str(checkpoint),
                 "--output", str(run_dir), "--updates", str(additional)], run_dir.name)
            progress = json.loads((run_dir / "progress.json").read_text())
            if progress["status"] != "block_complete":
                raise ValueError("Training did not seal a complete checkpoint")
            checkpoint = Path(progress["final_checkpoint"])
            evaluations = []
            report["status"] = "evaluating"
            for suffix, options in (("basic", []), ("envelope", ["--envelope"]), ("deployment", ["--deployment-pd"])):
                directory = run_dir.parent / (run_dir.name + "_" + suffix)
                run(["scripts/evaluate_flat_base.py", "--checkpoint", str(checkpoint), "--output", str(directory), *options],
                    run_dir.name + "_" + suffix)
                evaluations.append(directory / "evaluation.json")
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(json.dumps(dict(status=report["status"], manifest=str(args.output / "campaign.json"))), flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
