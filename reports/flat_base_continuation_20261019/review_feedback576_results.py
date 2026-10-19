"""Read completed cohorts, including regressions when both absolute grades fail."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_flat_base import compare

RUN = ROOT / "runs/flat_base_20261019"
load = lambda name: json.loads((RUN / name / "evaluation.json").read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checkpoint = RUN / "pilot012/model_000576.pt"
checkpoint_sha = sha(checkpoint)
rows = {}
for name, source_name, previous_name, older_name in (
        ("feedback576_basic001", "baseline_native002", "feedback512_basic001", "pilot010_eval"),
        ("feedback576_envelope001", "envelope_source002", "feedback512_envelope001", "envelope448_001"),
        ("rmcs576_basic001", "rmcs_source_basic001", "rmcs512_basic001", "rmcs448_basic001"),
        ("rmcs576_envelope001", "rmcs_source_envelope001", "rmcs512_envelope001", "rmcs448_envelope001")):
    path = RUN / name / "evaluation.json"
    if not path.exists() or load(name)["status"] != "evaluated":
        rows[name] = dict(status="awaiting_completion")
        continue
    data = load(name)
    if data["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Cohort belongs to another checkpoint")
    cases = data["result"]["cases"]
    comparisons = {}
    for label, refname in (("source", source_name), ("parent512", previous_name), ("command_parent448", older_name)):
        baseline = load(refname)
        paired = compare(baseline, data)
        relative = []
        for case_name, case in cases.items():
            vx, yaw, _ = case["command"]
            if label == "source" and abs(vx) < .75 and abs(yaw) < 1.5:
                continue
            metrics = dict(vx_mae_m_s=.02, yaw_mae_rad_s=.05)
            if label != "source" and case_name.startswith("stand_"):
                metrics["stand_drift_max_m"] = .03
            old = baseline["result"]["cases"][case_name]
            for key, tolerance in metrics.items():
                if case[key] > old[key] + tolerance:
                    relative.append(dict(case=case_name, metric=key, before=old[key], after=case[key], tolerance=tolerance))
        comparisons[label] = dict(reference_sha256=sha(RUN / refname / "evaluation.json"),
            no_lost_checks=paired["no_lost_baseline_checks"], lost_checks=paired["lost_baseline_checks"],
            no_relative_regressions=not relative, relative_regressions=relative)
    rows[name] = dict(status="evaluated", evaluation_sha256=sha(path),
        passed_cases=sum(c["passed"] for c in cases.values()), requested_cases=len(cases),
        all_targets_passed=data["result"]["passed"], comparisons=comparisons,
        metrics={n:{key:c[key] for key in ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m",
            "stand_drift_max_m", "full_horizon_episodes", "termination_reasons")} for n,c in cases.items()})
progress = json.loads((RUN / "pilot012/progress.json").read_text())
report = dict(schema="flat-feedback576-behavior-review-v1", recorded_at=datetime.now(timezone.utc).isoformat(),
    checkpoint_sha256=checkpoint_sha, training_status=progress["status"],
    successful_updates_total=progress["successful_updates_total"], actor_updates_total=progress["actor_updates_total"],
    command_branch_actor_updates=progress["active_branch_actor_updates"],
    feedback_branch_actor_updates=progress["feedback_branch_actor_updates"],
    consumed_optimizer_updates_total=progress["consumed_optimizer_updates_total"],
    source_training_updates="unknown", long_training_started=False, next_PPO_block_queued=False,
    behavior_qualified=False, hardware_motion_started=False, cohorts=rows)
out = Path(__file__).with_name("feedback576_behavior_review001.json")
temporary = out.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
temporary.replace(out)
print(json.dumps({name:dict(status=row["status"], passed_cases=row.get("passed_cases"),
    relative_regressions={k:len(v["relative_regressions"]) for k,v in row.get("comparisons",{}).items()})
    for name,row in rows.items()}))
