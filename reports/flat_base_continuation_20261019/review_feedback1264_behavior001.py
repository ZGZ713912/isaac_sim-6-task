"""Summarize the completed fixed cohorts and both bound regression references."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "runs/flat_base_20261019"
OUT = HERE / "feedback1264_behavior_review001.json"
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists():
    raise SystemExit("Preserve the completed behavior review")
seal_path = HERE / "feedback1264_population_completion001.json"
seal = read(seal_path)
queue_path = RUN / "feedback1264_population_queue001.json"
queue = read(queue_path)
if not seal["passed"] or queue["status"] != "complete" or seal["queue_sha256"] != sha(queue_path):
    raise SystemExit("Wait for all fixed cohorts and their source/event seal")
cohorts = {}
for name, summary in seal["cohorts"].items():
    evaluation_path = RUN / name / "evaluation.json"
    data = read(evaluation_path)
    if sha(evaluation_path) != summary["evaluation_sha256"]:
        raise ValueError("Fixed evaluation changed after sealing")
    cases = {}
    for case_name, case in data["result"]["cases"].items():
        cases[case_name] = {k: case[k] for k in (
            "passed", "vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m",
            "planar_speed_mean_m_s", "stand_drift_max_m", "full_horizon_episodes",
            "requested_episodes", "termination_reasons", "checks") if k in case}
        # Stopping peak, rather than a late average, is part of the existing verdict.
        for key, value in case.items():
            if "settled" in key:
                cases[case_name][key] = value
    cohorts[name] = dict(summary=summary, cases=cases)
result = dict(schema="flat-feedback1264-fixed-behavior-review-v1",
    recorded_at=datetime.now(timezone.utc).isoformat(), completion_sha256=sha(seal_path),
    checkpoint_sha256=seal["checkpoint_sha256"], cohorts=cohorts,
    gate_counterfactual_indicated=cohorts["deployment1264_basic001"]["cases"]["backward_05"]["vx_mae_m_s"] > .05,
    further_long_training_eligible=False, behavior_qualified=False,
    heldout_seed_used=False, hardware_motion_started=False,
    scope="Same1264 checkpoint, three completed development cohorts, original acceptance thresholds. Passing case counts do not override lost checks or relative regressions.")
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({name: value["summary"] for name, value in cohorts.items()}, indent=2))
