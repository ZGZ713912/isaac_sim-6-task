#!/usr/bin/env python3
"""Rebind an unchanged standard posture to measured passive timing evidence."""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, evidence, output):
    source, evidence, output = (p.resolve() for p in (source, evidence, output))
    if output.exists():
        raise ValueError("Preserve the existing reset bank")
    bank = json.loads(source.read_text()); probe = json.loads(evidence.read_text())
    if (probe.get("status") != "completed" or probe.get("all_pass") is not True
            or probe.get("new_PPO") != 0 or probe.get("checkpoint_reads") != []
            or probe.get("bank_sha256") != digest(source) or probe.get("height_shift_m") != 0.
            or probe.get("seconds", 0) < .15
            or (probe.get("physics_hz"), probe.get("policy_hz"), probe.get("feedback_pd_hz")) != (1000., 100., 200.)
            or not probe.get("runtime_source_hashes") or probe.get("source_mismatches_at_exit") != []):
        raise ValueError("Need original geometry, actor-free measured1000/100/200Hz passive evidence")
    for path, value in probe["runtime_source_hashes"].items():
        if digest(ROOT/path) != value:
            raise ValueError("Measured passive runtime source changed: " + path)
    indices = [i for i,p in enumerate(bank["poses"]) if p.get("recovery")]
    if len(indices) != 1 or bank["multi_pose_recovery"].get("reset_scope") != "standard_prone_only":
        raise ValueError("Only the unchanged single standard posture may be rebound")
    rows = probe["cases"]
    if (len(rows) != 2 or {r["pose_index"] for r in rows} != set(indices)
            or {r["heading_rad"] for r in rows} != {0., math.pi/2}
            or any(not r["passed"] or r["terminated"] or r["mechanical"]
                   or r["closure_peak_m"] > .0028 or r["passive_motor_peak_nm"] != 0. for r in rows)):
        raise ValueError("Need both measured headings with original passive admission margin")
    result = deepcopy(bank)
    declaration = dict(schema="native-passive-reset-admission-v1", new_PPO=0, checkpoint_reads=[],
        source_bank_sha256=digest(source), source_evidence_sha256=digest(evidence),
        source_evidence_file=str(evidence.relative_to(ROOT)), runtime_source_hashes=probe["runtime_source_hashes"],
        physics_hz=1000, policy_hz=100, feedback_pd_hz=200, seconds=probe["seconds"],
        guard_m=.003, admission_margin_m=.0002, cases=deepcopy(rows), all_pass=True,
        control_timing=deepcopy(probe["control_timing"]),
        scope="nominal_disabled_placement_only_not_recovery_standing_DR_or_hardware_qualification")
    output.parent.mkdir(parents=True, exist_ok=True)
    admission = output.parent/"native_admission.json"
    if admission.exists():
        raise ValueError("Preserve the existing native admission")
    admission.write_text(json.dumps(declaration, indent=2)+"\n")
    result["multi_pose_recovery"].update(
        native_reset_admission=dict(file=str(admission.relative_to(ROOT)), sha256=digest(admission)),
        admission="unchanged standard posture measured at physics1000/policy100/PD200Hz",
        source_pose_bank_sha256=digest(source))
    assert result["poses"] == bank["poses"]
    output.write_text(json.dumps(result, indent=2)+"\n")
    output.with_suffix(".build.json").write_text(json.dumps(dict(
        bank_sha256=digest(output), admission_sha256=digest(admission), builder_sha256=digest(Path(__file__)),
        source_bank_sha256=digest(source), source_evidence_sha256=digest(evidence),
        poses_exact=True, reset_scope="standard_prone_only", new_PPO=0, hardware_ready=False), indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); build(a.source, a.evidence, a.output)
