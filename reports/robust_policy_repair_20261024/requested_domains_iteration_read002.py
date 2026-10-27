"""Seal bounded progress/CPU evidence without changing the live training source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree

WORK=Path(__file__).resolve().parents[2]
ORIGINAL=Path("/home/noir/Documents/workspace/isaac_wheeled_rl_train")
REPORT=ORIGINAL/"reports/robust_policy_repair_20261024"
RUN=ORIGINAL/"runs/scratch_standing_20261026/pilot010_requested_domains_retry001"
NAME="requested_domains_iteration_read002"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


output=REPORT/(NAME+".json")
assert not output.exists()
raw=(RUN/"progress.json").read_bytes()
progress=json.loads(raw)
pid=progress["pid"]
assert pid==280546 and progress["start_ticks"]==1391023
assert int(Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19])==1391023
assert b"scripts/train_scratch_standing.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
assert progress["status"]=="training" and progress["checkpoint_reads"]==[]
hashes=read(RUN/"source_hashes.json")
assert len(hashes)==265
assert all(sha(WORK/name)==value==sha(RUN/"source_snapshot"/name) for name,value in hashes.items())
assert progress["committed_source"]["commit"]=="6cdeda1db836c6933d7e6a657eef20431477e300"

preflights=[]
for name in ("full_suite_preflight003_nominal","full_suite_preflight004_requested"):
    folder=REPORT/name
    receipt=read(folder/"receipt.json")
    assert receipt["status"]=="CPU_preflight_passed_unqualified"
    assert receipt["case_count"]==198 and receipt["native_policy_steps"]==0
    assert receipt["new_PPO"]==0 and receipt["optimizer_constructed"] is False
    assert receipt["spec_sha256"]==sha(folder/"spec.json")
    assert read(folder/"spec.json")==read(WORK/"contracts/wheel_leg_full_suite_v1.json")
    for helper,value in receipt["diagnostic_source_hashes"].items():
        path=WORK/("src/wheeled_tasks/chassis" if helper=="randomization_study.py" else "scripts")/helper
        assert sha(path)==value
    geometry=read(folder/"floor_geometry.json")
    assert geometry["factory_modified"] is False and geometry["native_simulation"] is False
    assert geometry["half_length_m"]==64 and geometry["width_m"]==128
    preflights.append(dict(path=str(folder),checkpoint_sha256=receipt["checkpoint_sha256"],
        seed=receipt["seed"],case_count=198,native_steps=0,
        evidence_sha256={n:sha(folder/n) for n in ("receipt.json","cases.json","preflight.json","floor_geometry.json")}))

test_file=REPORT/"full_suite_CPU004.xml"
tests=ElementTree.parse(test_file).getroot().find("testsuite")
assert tests is not None and tests.attrib["tests"]=="16"
assert tests.attrib["failures"]=="0" and tests.attrib["errors"]=="0"
comparison=read(REPORT/"requested_domains_first200002.json")
assert len(comparison["windows"])==2
assert comparison["windows"][1]["PPO_update_numbers_inclusive"]==[101,200]
moment=datetime.now(timezone.utc).isoformat()
snapshot=REPORT/(NAME+".progress.json")
snapshot.write_bytes(raw)
report=dict(schema="requested-domain-v2-progress-and-full-suite-preparation-v1",observed_utc=moment,
    active_run=str(RUN),active_pid=pid,active_start_ticks=1391023,
    completed_fresh_updates=progress["completed_updates"],requested_updates=512,
    frozen_training_commit=progress["committed_source"]["commit"],all265_live_and_archived_sources_unchanged=True,
    progress_snapshot_sha256=sha(snapshot),checkpoint_reads=[],optimizer_attempts=progress["optimizer_attempts"],
    RSL_comparison=str(REPORT/"requested_domains_first200002.json"),
    RSL_comparison_sha256=sha(REPORT/"requested_domains_first200002.json"),
    full_suite_CPU_tests=16,CPU_test_evidence_sha256=sha(test_file),full_suite_CPU_preflights=preflights,
    full_suite_native_steps=0,full_suite_native_verified=False,
    superseded_design_only_preflights=["full_suite_preflight001_nominal","full_suite_preflight002_requested"],
    preflight_correction="Exercise the archived floor factory too. Its64m half-length bound requires declared18s high-speed straight; other cases30s. No factory/controller patch or posthoc test window.",
    existing_capabilities_still_required=[c["id"] for c in read(WORK/"contracts/wheel_leg_acceptance_requirements_v2.json")["capabilities"]],
    next_actions=["Complete512 bounded fresh updates; seal/check exact process exit and final checkpoint.",
        "Evaluate the same frozen final actor in nominal/requested held-out domains; keep all failed trials and apply<=2ms contact review.",
        "If degraded, compare declared without-encoder-bias, without-policy-delay and without-alignment profiles; verify actual nonintervened draws before causal attribution.",
        "Train a fresh unified curriculum covering all permanent tasks, without loading an old policy/teacher; complete full behavior/seed/domain gates before long training, ONNX and RMCS integration."],
    training_continues=True,long_training_started=False,ONNX_exported=False,RMCS_deployed=False,
    native_behavior_qualified=False,hardware_ready=False,verifier_sha256=sha(Path(__file__)))
write(output,report)
shutil.copyfile(Path(__file__),REPORT/(NAME+".py"))

ledger_file=REPORT/"scratch_campaign001.json"
ledger_raw=ledger_file.read_bytes()
ledger=json.loads(ledger_raw)
assert ledger["active_run"]==str(RUN) and ledger["active_pid"]==pid and ledger["active_start_ticks"]==1391023
assert ledger["prior_closed_fresh_updates"]==2945
ledger.setdefault("historical_accounting_before_nominal009",dict(prior_closed_fresh_updates=2433,nominal009_completed_updates=512))
ledger.update(status="training_unqualified",completed_updates_observed=progress["completed_updates"],
    active_completed_updates_observed=progress["completed_updates"],source_snapshot_count=265,
    prior_from_zero_completed=2945,prior_completed_fresh_updates=2945,
    total_from_zero_completed_observed=2945+progress["completed_updates"],
    latest_live_read=str(output),latest_launch=str(REPORT/"scratch_pilot010_launch_identity001.json"),
    latest_native_launch=str(REPORT/"scratch_pilot010_launch_identity001.json"),
    last_observed_utc=moment,observed_utc=moment,
    latest_event_comparison=str(REPORT/"requested_domains_first200002.json"),
    latest_active_run_event_comparison=str(REPORT/"requested_domains_first200002.json"),
    latest_training_plot_runs=["pilot009_leg_velocity_separated","pilot010_requested_domains_retry001"],
    prepared_randomization_study=str(WORK/"contracts/wheel_leg_randomization_study_v2.json"),
    latest_full_suite_preflight=[entry["path"] for entry in preflights],full_suite_native_verified=False,
    next_action="完成512更新的从零指定随机域短训并封存，同权重名义/随机留出域复测；必要时严格配对定位偏置/延迟/IMU。完整能力仍需从零统一训练并逐项通过，不能以基础静站或CPU预检准入长训/ONNX/实车。",
    qualified=False,hardware_ready=False)
assert ledger_file.read_bytes()==ledger_raw
temporary=ledger_file.with_suffix(".read002.tmp")
write(temporary,ledger)
temporary.replace(ledger_file)
print(json.dumps(dict(completed_updates=progress["completed_updates"],live_sources_verified=265,
    CPU_tests=16,full_suite_cases=198,new_native_steps=0,total_fresh_updates_observed=ledger["total_from_zero_completed_observed"])))
