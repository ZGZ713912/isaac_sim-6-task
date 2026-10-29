#!/usr/bin/env python3
"""Compare equal-condition deterministic flat trials without reward ranking."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(baseline, candidate):
    for key in ("seed", "source_onnx_sha256", "asset_manifest_sha256", "model_spec_sha256", "settings",
                "policy_hz", "physics_hz"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Paired evaluation differs in {key}")
    if baseline["status"] != candidate["status"] or candidate["status"] != "evaluated":
        raise ValueError("Both trials must have completed")
    before, after = baseline["result"]["cases"], candidate["result"]["cases"]
    if before.keys() != after.keys() or any(c["episodes"] != 4 for c in (*before.values(), *after.values())):
        raise ValueError("All cases require four accounted first episodes")
    changes, regressions = {}, []
    for name in before:
        old, new = before[name], after[name]
        metrics = ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "stand_drift_max_m", "closure_gap_max_m")
        changes[name] = {key: dict(before=old[key], after=new[key], difference=new[key] - old[key]) for key in metrics}
        lost = [key for key, passed in old["checks"].items() if passed and not new["checks"][key]]
        # Crossing an absolute user target matters even when the old model was
        # much better. Harmless changes within the same target are recorded.
        if lost:
            regressions.append(dict(case=name, lost_checks=lost))
    return dict(schema="flat-base-paired-behavior-comparison-v1", no_lost_baseline_checks=not regressions,
        lost_baseline_checks=regressions, passed_cases=sum(c["passed"] for c in after.values()),
        requested_cases=len(after), all_behavior_targets_passed=candidate["result"]["passed"], changes=changes,
        eligible_for_large_training=not regressions and candidate["result"]["passed"],
        hardware_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    report.update(baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
