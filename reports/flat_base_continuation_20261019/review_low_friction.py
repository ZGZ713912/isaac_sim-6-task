"""Seal separate-material implementation and retain all negative recovery evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("low_friction_candidate001")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    path = ROOT / name
    return json.loads(path.read_text()), dict(file=name, sha256=sha(path))


def main():
    checkpoint = ROOT / "runs/flat_base_20261019/pilot028_wheel_rate1328/model_001328.pt"
    checkpoint_sha = "5e2320c50b8fd7982b278dfb4d21cc4763cf12be3cf10193bbf630c648bf672b"
    assert sha(checkpoint) == checkpoint_sha
    low, low_source = read("runs/flat_base_20261019/low_friction1328_basic001/evaluation.json")
    assert low["status"] == "evaluated" and low["checkpoint_sha256"] == checkpoint_sha
    assert low["new_optimizer_updates"] == 0
    assert sha(ROOT / "models/wheel_leg_flat_base_20261018/source.onnx") == low["source_onnx_sha256"]
    material = low["startup"]["contact_material"]
    assert material["native_readback"]["usd_collision_bindings_verified"]
    assert material["native_readback"]["physx_robot_shape_coefficients_verified"]
    assert material["wheel_static_friction"] == .1
    assert (low["policy_hz"], low["physics_hz"]) == (50, 1000)
    cases = low["result"]["cases"]
    recovery = {}
    for variant in ("legacy001", "slow001", "slow002", "slow003", "slow004", "slow005"):
        report, source = read(f"runs/flat_base_20261019/low_friction_recovery_{variant}/report.json")
        assert report["status"] == "evaluated" and report["source_unchanged"]
        assert report["checkpoint_sha256"] == checkpoint_sha and report["new_optimizer_updates"] == 0
        normal = {name: row for name, row in report["cases"].items() if not name.endswith("_immediate_control")}
        immediate = {name: row for name, row in report["cases"].items() if name.endswith("_immediate_control")}
        recovery[variant] = dict(source=source,
            normal_successes=sum(row["successful_recovery_episodes"] for row in normal.values()),
            normal_episodes=sum(row["requested_episodes"] for row in normal.values()),
            immediate_successes=sum(row["successful_recovery_episodes"] for row in immediate.values()),
            immediate_episodes=sum(row["requested_episodes"] for row in immediate.values()),
            closure_gap_max_m=max(row["closure_gap_max_m"] for row in report["cases"].values()),
            cases=report["cases"])
    randomized, randomized_source = read("runs/flat_base_20261019/random_friction1328_material002/evaluation.json")
    assert randomized["status"] == "evaluated" and randomized["new_optimizer_updates"] == 0
    assert randomized["checkpoint_sha256"] == checkpoint_sha
    random_material = randomized["startup"]["contact_material"]
    values = random_material["ground_static_friction_per_environment"]
    random_counts = [sum(abs(v - .1) < 1e-6 for v in values),
        sum(.2 <= v < .3 for v in values), sum(.3 <= v <= .400001 for v in values)]
    assert min(random_counts) > 0
    assert random_material["native_readback"]["usd_collision_bindings_verified"]
    assert random_material["native_readback"]["physx_robot_shape_coefficients_verified"]
    tests = ET.parse(OUT / "cpu_tests001.xml").getroot()
    suites = [tests] if tests.tag == "testsuite" else list(tests)
    assert all(int(row.attrib.get("failures", 0)) == int(row.attrib.get("errors", 0)) == 0 for row in suites)
    diagnostic_tests = ET.parse(OUT / "diagnostic_cpu001.xml").getroot()
    diagnostic_suites = list(diagnostic_tests)
    assert sum(int(row.attrib["tests"]) for row in diagnostic_suites) == 4
    assert all(int(row.attrib.get("failures", 0)) == int(row.attrib.get("errors", 0)) == 0
               for row in diagnostic_suites)
    report = dict(schema="flat-low-friction-implementation-review-v1", status="implemented_behavior_incomplete",
        recorded_at=datetime.now(timezone.utc).isoformat(), checkpoint_sha256=checkpoint_sha,
        original_onnx_weights_preserved=True, new_optimizer_updates=0, hardware_motion_started=False,
        qualified=False, automatic_randomized_training_started=False, full_training_admitted=False,
        current_recovery_schema="compact-sagittal-recovery-v5", recovery_production_default_changed=False,
        new_training_blocked_by="low static drift exceeds 3 cm; existing trainable wheel head has zero gate in all 6000 static inputs",
        basic=dict(source=low_source, passed_cases=sum(row["passed"] for row in cases.values()),
            requested_cases=len(cases), failed_cases=[name for name, row in cases.items() if not row["passed"]],
            cases=cases, native_material=material),
        randomized_material_probe=dict(source=randomized_source, band_counts=random_counts,
            diagnostic_seconds=2., scope="Material binding and sampling only; no trained randomized-stage result",
            native_material=random_material), recovery=recovery,
        cpu_tests=dict(passed=sum(int(row.attrib["tests"]) for row in suites),
            junit_sha256=sha(OUT / "cpu_tests001.xml")),
        evaluator_noop_tests=dict(passed=4, junit_sha256=sha(OUT / "diagnostic_cpu001.xml")),
        source_final_sha256={str(path.relative_to(ROOT)): sha(path) for directory in
            (ROOT / "src/wheeled_tasks/chassis", ROOT / "src/wheeled_algo") for path in directory.glob("*.py")})
    target = OUT / "review001.json"
    if target.exists():
        raise ValueError("Preserve existing review receipt")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], basic=report["basic"]["passed_cases"],
        randomized_counts=random_counts, cpu_tests=report["cpu_tests"]["passed"],
        recovery={key: [row["normal_successes"], row["immediate_successes"]] for key, row in recovery.items()})))


if __name__ == "__main__":
    main()
