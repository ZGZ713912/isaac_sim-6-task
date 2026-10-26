"""Read the original live008 process, complete event windows and pose provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUN = ROOT / "runs/scratch_standing_20261026/pilot008_leg_target_rate020"
OUTPUT = REPORTS / "scratch_training008_read008.json"
assert not OUTPUT.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


identity_path = REPORTS / "scratch_pilot008_launch_identity001.json"
identity = json.loads(identity_path.read_text())
pid = identity["pid"]
stat = Path(f"/proc/{pid}/stat").read_text()
fields = stat[stat.rfind(")") + 2:].split()
assert int(fields[19]) == identity["start_ticks"] and fields[0] != "Z"
actual_command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
assert [value.decode() for value in actual_command if value] == identity["command"]
gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                     text=True, capture_output=True, check=True)
assert str(pid) in gpu.stdout.splitlines()
progress = json.loads((RUN / "progress.json").read_text())
assert progress["status"] == "training" and progress["checkpoint_reads"] == []
assert progress["requested_updates"] == 512 and progress["completed_updates"] >= 384
sources = json.loads((RUN / "source_hashes.json").read_text())
assert len(sources) == 263
for name, expected in sources.items():
    assert sha(ROOT / name) == expected, name
    assert sha(RUN / "source_snapshot" / name) == expected, name

comparison_path = REPORTS / "scratch_training007_008_window032_read011.json"
comparison = json.loads(comparison_path.read_text())
for item in comparison["sources"].values():
    with Path(item["path"]).open("rb") as stream:
        prefix = stream.read(item["observed_prefix_bytes"])
    assert hashlib.sha256(prefix).hexdigest() == item["observed_prefix_sha256"]
assert comparison["windows"][-1]["PPO_update_numbers_inclusive"] == [353, 384]

bank_path = REPORTS / "pose_bank_height_matched073.json"
prior_path = REPORTS / "pose_bank_rmcs003.json"
bank = json.loads(bank_path.read_text())
prior = json.loads(prior_path.read_text())
assert sha(bank_path) == sha(RUN / "pose_bank.json")
assert sha(prior_path) == "1d07f64a75a9b683995b3ca270c3a79eb3564e75c149dd1115ac5902cc4a222e"
assert len(bank["poses"]) == 432 and len(prior["poses"]) == 429
assert bank["poses"][:429] == prior["poses"]
assert bank["diagnostic_height_starts"]["mapping"] == {"stand_low": 429, "stand": 430, "stand_high": 431}
rmcs_poses = [item for item in bank["poses"] if "rmcs" in item["name"].lower()]
assert len(rmcs_poses) == 72
reference_path = REPORTS / "rmcs_reference/reference_receipt.json"
reference = json.loads(reference_path.read_text())
assert reference["branch"] == "dev/wheel_leg_rl"
assert reference["commit"] == "0fcc9b45cd80e4dba69c03a1fbed2b848ab5092a"
assert not reference["dynamic_recovery_validated"]
assert progress["args"]["all_heights_after"] == 512
sampler = "src/wheeled_tasks/chassis/scratch_standing_env.py"
assert sampler in sources

now = datetime.now(timezone.utc).isoformat()
report = dict(schema="scratch-standing-live-read-v1", observed_utc=now, run=str(RUN), pid=pid,
    start_ticks=identity["start_ticks"], original_process=True, original_PID_in_GPU_compute=True,
    status=progress["status"], completed_updates=progress["completed_updates"], requested_updates=512,
    source_count=len(sources), all_live_and_frozen_sources_match=True, training_checkpoint_reads=0,
    prior_completed_fresh_updates=1921, total_from_zero_completed_observed=1921 + progress["completed_updates"],
    event_comparison=str(comparison_path), event_comparison_sha256=sha(comparison_path),
    last_complete_window=comparison["windows"][-1],
    RMCS_pose_provenance=dict(reference=str(reference_path), reference_sha256=sha(reference_path),
        reference_branch=reference["branch"], reference_commit=reference["commit"],
        original429_poses_and_indices_exact=True, preserved_rmcs_pose_count=72,
        actual_run_bank_sha256=sha(bank_path), sampler_path=sampler, sampler_sha256=sources[sampler],
        all_rollouts_before_update512_mid_height_only=True, recovery_poses_used_in_this_training=False,
        dynamic_recovery_validated=False),
    reader_sha256=sha(Path(__file__)), new_PPO_by_reader=0, own_native_evaluation_finished=False,
    behavior_qualification=False, hardware_ready=False)
margin_path = REPORTS / "scratch_contact_margin007_audit001.json"
margin = json.loads(margin_path.read_text())
assert margin["all_trace_and_formula_checks_pass"]
assert margin["aggregate"]["interruption_onsets"] == 216
assert margin["aggregate"]["Fz_before_break_min_N"] > 59.
assert margin["onset_substep_counts_1_to_5"] == [47, 46, 37, 40, 46]
excitation_path = REPORTS / "scratch_physical_excitation005_006_007_read001.json"
excitation = json.loads(excitation_path.read_text())
calibration_path = REPORTS / "scratch_physical_excitation_validation001.json"
calibration = json.loads(calibration_path.read_text())
assert calibration["all_checks_pass"] and calibration["verified_actual_RSL_values"] == 24
assert calibration["native_report_sha256"] == sha(excitation_path)
assert calibration["source_sha256"] == sha(ROOT / "scripts/inspect_scratch_physical_excitation.py")
assert excitation["all_original_metrics_and_parseval_checks_pass"]
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
verified = 0
for info, expected, step, prefix in [(margin["tensorboard"], margin["aggregate"], 512, "DiagnosticScratchContactMargin/")]:
    event, = Path(info["directory"]).glob("events.out.tfevents.*")
    assert sha(event) == info["files_sha256"][event.name]
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
    for key, value in expected.items():
        entry, = accumulator.Scalars(prefix + key)
        assert entry.step == step and abs(entry.value - value) < max(1e-10, abs(value) * 1e-6)
        verified += 1
assert verified == 11
for info in excitation["tensorboard"].values():
    event, = Path(info["directory"]).glob("events.out.tfevents.*")
    assert sha(event) == info["files_sha256"][event.name]
report.update(contact_margin_audit=str(margin_path), contact_margin_audit_sha256=sha(margin_path),
    matched_contact_margin_RSL_values=verified, physical_excitation_report=str(excitation_path),
    physical_excitation_report_sha256=sha(excitation_path), physical_excitation_calibration=str(calibration_path),
    calibrated_signal_cases=len(calibration["known_signal_cases"]),
    invalid_physical_signal_cases_rejected=calibration["invalid_cases_rejected"],
    matched_physical_excitation_RSL_values=24)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
campaign_path = REPORTS / "scratch_campaign001.json"
campaign = json.loads(campaign_path.read_text())
assert campaign["active_pid"] == pid and campaign["active_start_ticks"] == identity["start_ticks"]
campaign.update(completed_updates_observed=progress["completed_updates"],
    total_from_zero_completed=1921 + progress["completed_updates"],
    total_from_zero_completed_observed=1921 + progress["completed_updates"],
    latest_live_read=str(OUTPUT), last_observed_utc=now, latest_event_comparison=str(comparison_path),
    latest_RMCS_pose_provenance=str(OUTPUT), latest_contact_margin_audit=str(margin_path),
    latest_physical_excitation_report=str(excitation_path),
    latest_physical_excitation_calibration=str(calibration_path),
    next_action="Continue original fresh008 PID709399/start31429889 and263 frozen sources to512 plus own strict21-case8s "
        "evaluation. Latest353-384 stochastic support is slightly better than007 but leg RMS remains worse; improvement is not "
        "sustained. Seal only after original process and GPU exit, then inspect every1ms contact and vibration. "
        "All512 rollouts are mid-height only; preserved72 RMCS poses were not sampled here. No restart, resumed "
        "training or qualification. Later training must start from zero and use a verified committed source version.")
temporary = campaign_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(campaign, indent=2, allow_nan=False) + "\n")
temporary.replace(campaign_path)
print(json.dumps(dict(completed_updates=progress["completed_updates"], pid=pid,
    all_live_and_frozen_sources_match=True, preserved_rmcs_pose_count=72, output=str(OUTPUT))))
