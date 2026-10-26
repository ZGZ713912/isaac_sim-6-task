"""Seal the target-variation candidate's actual failure and current serial diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
OUTPUT = REPORTS / "scratch_pilot008_review001.json"
assert not OUTPUT.exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())
names = ("pilot007_contact_cost_zero", "pilot008_leg_target_rate020")
runs = {name: ROOT / "runs/scratch_standing_20261026" / name for name in names}
seals = [read(REPORTS / ("scratch_pilot00" + str(number) + "_closed_seal001.json")) for number in (7, 8)]
assert all(item["all_checks_pass"] and item["completed_updates"] == 512 for item in seals)
gpu = [int(value) for value in subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).splitlines()
    if value.strip().isdigit()]
assert not Path("/proc/709399").exists() and 709399 not in gpu
exit_path = REPORTS / "scratch_pilot008_actual_exit001.json"
assert not exit_path.exists()
exit_path.write_text(json.dumps(dict(schema="scratch-native-process-exit-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), pid=709399, start_ticks=31429889,
    original_proc_absent=True, original_GPU_compute_absent=True, remaining_GPU_compute_pids=gpu,
    scope="Original008 absence rechecked after its seal; the serial probe launch independently verified this before starting."),
    indent=2, allow_nan=False) + "\n")
sources = [read(run / "source_hashes.json") for run in runs.values()]
assert [len(value) for value in sources] == [262, 263]
assert [name for name in sources[0].keys() & sources[1].keys() if sources[0][name] != sources[1][name]] == ["scripts/train_scratch_standing.py"]
assert sources[1].keys() - sources[0].keys() == {"src/wheeled_tasks/chassis/standing_leg_target_rate.py"}
for name, expected in sources[1].items():
    assert sha(ROOT / name) == expected and sha(runs[names[1]] / "source_snapshot" / name) == expected
tasks = [read(run / "task_contract.json") for run in runs.values()]
settings = [value["settings"] for value in tasks]
assert [key for key in settings[0].keys() | settings[1].keys() if settings[0].get(key) != settings[1].get(key)] == ["version"]
configs = [read(run / "config/contract.json") for run in runs.values()]
assert {k: v for k, v in configs[0].items() if k != "design_preflight"} == {
    k: v for k, v in configs[1].items() if k != "design_preflight"}
assert sha(runs[names[0]] / "agent_config.json") == sha(runs[names[1]] / "agent_config.json")
initial = [read(run / "initialization.json") for run in runs.values()]
assert all(initial[0][role + "_tensor_sha256"] == initial[1][role + "_tensor_sha256"] for role in ("actor", "critic"))
assert all(value["optimizer_states"] == 0 and not value["checkpoint_loaded"] and not value["teacher_loaded"] for value in initial)
behavior_paths = [REPORTS / ("scratch_pilot00" + str(number) + "_behavior001.json") for number in (7, 8)]
behaviors = [read(path) for path in behavior_paths]
groups = [next(row for row in value["groups"] if row["height_m"] == .305) for value in behaviors]
assert [row["total_missing_ms"] for row in groups] == [217, 194]
assert all(row["standing_pass"] == 0 and row["cases"] == 7 for row in groups)
break_path = REPORTS / "scratch_support_breaks008_read001.json"
breaks = read(break_path)
assert breaks["total_missing_physical_ms"] == 194 and breaks["total_interruption_events"] == 131
assert [row["missing_physical_ms"] for row in breaks["cases"]] == [131, 21, 15, 12, 6, 5, 4]
evaluations = [read(run / "evaluation/result.json") for run in runs.values()]
verified_native_values = 0
for run, evaluation in zip(runs.values(), evaluations):
    assert sha(run / "evaluation/native_trace.npz") == evaluation["trace_sha256"]
    event, = run.glob("events.out.tfevents.*")
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
    cases = [case for case in evaluation["cases"] if case["height_m"] == .305]
    for metric in ("dual_support_fraction", "missing_physical_ms", "leg_speed_peak_rad_s",
                   "leg_vibration_3_95hz_rms_mrad", "height_mae_m", "forward_speed_rms_m_s"):
        value, = accumulator.Scalars("EvalScratchStanding/0.305m/" + metric)
        expected = sum(case[metric] for case in cases) / 7
        assert value.step == 512 and abs(value.value - expected) < max(1e-10, abs(expected) * 1e-6)
        verified_native_values += 1
excitation_path = REPORTS / "scratch_physical_excitation007_008_read001.json"
excitation = read(excitation_path)
assert excitation["all_original_metrics_and_parseval_checks_pass"]
verified_physical_values = 0
for row in excitation["evaluations"]:
    info = excitation["tensorboard"][Path(row["evaluation"]).parent.name]
    event, = Path(info["directory"]).glob("events.out.tfevents.*")
    assert sha(event) == info["files_sha256"][event.name]
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
    for key, expected in dict(row["means_of_sufficient_cases"], sufficient_late_cases=row["sufficient_late_cases"]).items():
        value, = accumulator.Scalars("DiagnosticScratchPhysicalExcitation/0.305m/" + key)
        assert value.step == 512 and abs(value.value - expected) < max(1e-10, abs(expected) * 1e-6)
        verified_physical_values += 1
assert verified_native_values == 12 and verified_physical_values == 16
plot_path = REPORTS / "scratch_training007_008_window032_read013_compactticks.png"
plot_receipt = read(plot_path.with_suffix(".receipt.json"))
assert plot_receipt["every_plotted_window_verified_from_original_events"] and sha(plot_path) == plot_receipt["plot_sha256"]
assert sha(plot_path.with_suffix(".receipt.plot_source.py")) == plot_receipt["source_sha256"]
ratios = {name: groups[1][name] / groups[0][name] for name in (
    "mean_leg_vibration_mrad", "maximum_leg_speed_rad_s", "total_missing_ms")}
probe_identity_path = REPORTS / "scratch_eval007_leg_velocity_zero_launch001.json"
probe_identity = read(probe_identity_path)
probe_proc = Path(f"/proc/{probe_identity['pid']}/stat")
probe_live = False
if probe_proc.exists():
    stat = probe_proc.read_text()
    probe_live = int(stat[stat.rfind(")") + 2:].split()[19]) == probe_identity["start_ticks"]
report = dict(schema="scratch-target-variation-candidate-review-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), run=str(runs[names[1]]), completed_updates=512,
    prior_completed_fresh_updates=1921, total_from_zero_completed=2433,
    initial_actor_and_critic_hashes_exact=True, initial_Adam_empty_both=True,
    common_mechanism_agent_bank_settings_preserved=True,
    intended_training_change="Applied four-leg target-variation cost0 to0.2/s; same fresh seed and512 budget.",
    baseline_mid=groups[0], candidate_mid=groups[1], candidate_to_baseline_ratios=ratios,
    same_initial_root_positions= evaluations[0]["initial_root_positions"] == evaluations[1]["initial_root_positions"],
    initial_full_native_state_pairing_available=False,
    matched_original_native_RSL_values=verified_native_values,
    matched_physical_excitation_RSL_values=verified_physical_values,
    physical_excitation_report=str(excitation_path), physical_excitation_report_sha256=sha(excitation_path),
    full21_standing_pass=0, strict_mid_standing_pass=0,
    native_exit_receipt=str(exit_path), seal=str(REPORTS / "scratch_pilot008_closed_seal001.json"),
    plot=str(plot_path), training_sources_unchanged=263,
    current_readonly_probe_identity=str(probe_identity_path), current_probe_live=probe_live,
    reader_sha256=sha(Path(__file__)), new_PPO_by_reader=0, optimizer_constructed=False,
    adopt_candidate=False, qualification=False, hardware_ready=False,
    decision="Reject008 as a standing repair. Slightly fewer missing milliseconds cannot offset larger vibration, "
        "higher speed peaks and a severe zero-pitch regression. Do not continue or adopt its weights. "
        "The serial zero-PPO007 leg-speed diagnostic tests a feedback hypothesis against its fresh matched baseline; "
        "it is not an008 ablation or a replacement trained policy. All later training remains from zero.")
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
campaign_path = REPORTS / "scratch_campaign001.json"
campaign = read(campaign_path)
campaign.update(status="readonly_feedback_diagnostic", active_run=None, active_pid=None, active_start_ticks=None,
    completed_updates_observed=512, total_from_zero_completed=2433, total_from_zero_completed_observed=2433,
    last_observed_utc=report["observed_utc"], latest_live_read=str(OUTPUT), latest_candidate_review=str(OUTPUT),
    latest_behavior_analysis=str(behavior_paths[1]), latest_support_break_evidence=str(break_path),
    latest_training_plot=str(plot_path), latest_event_comparison=str(REPORTS / "scratch_training007_008_window032_read013.json"),
    latest_physical_excitation_report=str(excitation_path), latest_training_launch=str(REPORTS / "scratch_pilot008_launch_identity001.json"),
    latest_native_launch=str(probe_identity_path), active_evaluation=probe_identity["run"] if probe_live else None,
    active_evaluation_pid=probe_identity["pid"] if probe_live else None,
    active_evaluation_start_ticks=probe_identity["start_ticks"] if probe_live else None,
    latest_behavior=dict(mid_height_standing_pass=0, mid_height_cases=7, mid_height_total_missing_ms=194,
        mid_height_mean_leg_vibration_mrad=groups[1]["mean_leg_vibration_mrad"],
        mid_height_maximum_leg_speed_rad_s=groups[1]["maximum_leg_speed_rad_s"],
        full_evaluation_standing_pass=0, full_evaluation_cases=21, qualified=False),
    next_action="Wait for original serial007 zero-leg-speed probe PID746147/start32122474 to finish and actually exit. "
        "Authenticate all matched baseline, native first-state and untouched-population checks before attribution. "
        "008 is rejected and sealed at512; no continuation or budget extension. All future training must start from zero "
        "and launch from a verified committed source. Heights, long horizon, domains, motion/recovery and hardware remain unqualified.")
temporary = campaign_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(campaign, indent=2, allow_nan=False) + "\n")
temporary.replace(campaign_path)
print(json.dumps(dict(candidate_rejected=True, candidate_mid=groups[1], ratios=ratios,
    total_from_zero_completed=2433, native_RSL_values_verified=12, physical_RSL_values_verified=16,
    current_probe_live=probe_live)))
