"""Reproduce CPU-only profile/design bindings, without creating a native env."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.randomization_study import resolve


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


directory = ROOT / "reports/robust_policy_repair_20261024"
spec_path = ROOT / "contracts/wheel_leg_randomization_study_v1.json"
spec = json.loads(spec_path.read_text())
rows = []
for name in ("domain_sensor_first_preflight002", "domain_combined_light_preflight001"):
    run = ROOT / "runs/scratch_standing_20261026" / name
    files = {name: run / name for name in ("task_contract.json", "config/contract.json",
        "config/design.json", "randomization_study.resolved.json", "randomization_study.input.json",
        "source_hashes.json")}
    task = json.loads(files["task_contract.json"].read_text())
    cfg = json.loads(files["config/contract.json"].read_text())
    study = json.loads(files["randomization_study.resolved.json"].read_text())
    design = json.loads(files["config/design.json"].read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    sources = json.loads(files["source_hashes.json"].read_text())
    checks = dict(input_hash_matches=study["input_sha256"] == sha(spec_path) == sha(files["randomization_study.input.json"]),
        ranges_resolve_exact={k: v for k, v in study.items() if k not in ("input_file", "input_sha256")}
            == resolve(spec, study["profile"], study["strength"]),
        saved_task_ranges_exact=task["settings"]["randomization_study"] == study,
        no_legacy_scale="domain_scale" not in task["settings"],
        no_legacy_overlay="gap_robustness_version" not in task["settings"],
        mechanism_hash_exact=task["mechanism_config_sha256"] == sha(files["config/contract.json"]),
        design_hash_exact=cfg["design_preflight"]["sha256"] == sha(files["config/design.json"]),
        runtime_dynamics_flag_exact=row["runtime"]["modules"]["dynamics_randomization"] == cfg["dynamics_randomization"]["enabled"],
        design_rigid_ranges_exact=row["contract"]["dynamics_randomization"] == cfg["dynamics_randomization"] == study["dynamics_randomization"],
        design_material_exact=row["contract"]["contact_material"] == row["runtime"]["contact_material"] == cfg["contact_material"],
        clocks_unchanged=cfg["physics_dt"] == .001 and cfg["policy_dt"] == .005,
        frozen_sources_exact=all(sha(run / "source_snapshot" / path) == expected for path, expected in sources.items()),
        resolver_source_current=sha(run / "source_snapshot/src/wheeled_tasks/chassis/randomization_study.py")
            == sha(ROOT / "src/wheeled_tasks/chassis/randomization_study.py"),
        fresh_origin_declared=task["training_origin"] == "seeded random actor and critic; empty Adam; no checkpoint reads",
        no_model_artifacts=not list(run.glob("*.pt")), native_not_started=not (run / "progress.json").exists())
    if not all(checks.values()):
        raise ValueError(f"Profile preflight failed: {name}: {checks}")
    rows.append(dict(run=str(run), profile=study["profile"], checks=checks,
        source_count=len(sources), files_sha256={key: sha(value) for key, value in files.items()}))
xml = directory / "randomization_study_CPU002.xml"
suites = ET.parse(xml).getroot().findall("testsuite")
cpu = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
if cpu != dict(tests=40, failures=0, errors=0, skipped=0):
    raise ValueError("Required final CPU tests did not pass")
output = directory / "randomization_study_preflight001.json"
if output.exists():
    raise ValueError("Preserve previous preflight receipts")
report = dict(schema="wheel-leg-domain-study-CPU-preflight-v1", runs=rows, all_checks_pass=True,
    CPU_test_summary=cpu, CPU_junit_sha256=sha(xml), reader_sha256=sha(__file__),
    source_resolver_sha256=sha(ROOT / "src/wheeled_tasks/chassis/randomization_study.py"),
    new_PPO=0, new_native_steps=0, checkpoint_loads=0, hardware_ready=False,
    scope="Configuration bindings, actual CPU sensor/PD sampler units and fresh initialization tests only. No native behavior or optimal-intensity claim.")
output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
print(json.dumps(dict(all_checks_pass=True, runs=len(rows), CPU_tests=cpu["tests"],
                     checks=sum(len(r["checks"]) for r in rows))))
