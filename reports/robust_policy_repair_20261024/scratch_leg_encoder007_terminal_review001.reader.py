"""Independent terminal checks for the matched native leg-speed intervention."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_events(info, expected):
    directory = Path(info["directory"])
    assert {p.name: sha(p) for p in directory.glob("events.out.tfevents.*")} == info["files_sha256"]
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    for tag, value in expected.items():
        row, = events.Scalars(tag)
        assert row.step == 512 and math.isclose(row.value, value, rel_tol=2e-6, abs_tol=1e-8), tag
    return len(expected)


comparison = read(REPORTS / "scratch_leg_encoder007_zero_comparison001.json")
excitation = read(REPORTS / "scratch_leg_encoder007_physical_excitation001.json")
assert comparison["all_checks_pass"] and len(comparison["checks"]) == 23 and all(comparison["checks"].values())
counts = {}
expected = {"DiagnosticLegEncoderNative/" + group + "/" + key: value
            for group in ("baseline_mid", "intervention_mid") for key, value in comparison[group].items()}
expected["DiagnosticLegEncoderNative/all_provenance_checks_pass"] = 1
counts["matched_comparison"] = verify_events(comparison["tensorboard"], expected)
for row in excitation["evaluations"]:
    name = Path(row["evaluation"]).parent.name
    expected = {"DiagnosticScratchPhysicalExcitation/0.305m/" + key: value
                for key, value in row["means_of_sufficient_cases"].items()}
    expected["DiagnosticScratchPhysicalExcitation/0.305m/sufficient_late_cases"] = row["sufficient_late_cases"]
    counts[name] = verify_events(excitation["tensorboard"][name], expected)
run = Path(comparison["intervention"])
receipt = read(run / "receipt.json")
identity = read(REPORTS / "scratch_eval007_leg_velocity_zero_launch001.json")
stat = Path("/proc") / str(identity["pid"]) / "stat"
original_live = stat.exists() and int(stat.read_text().rsplit(") ", 1)[1].split()[19]) == identity["start_ticks"]
gpu_pids = [int(p.strip()) for p in subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines() if p.strip()]
events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
metrics = dict(actor_calls=1600, selected_original_cases=7, response_scale=0.,
               native_wheel_output_max_error=0., first_action_native=1.)
for name, value in metrics.items():
    row, = events.Scalars("LegEncoderProbe/" + name)
    assert row.step == 512 and row.value == value
counts["native_intervention"] = len(metrics)
assert not original_live and identity["pid"] not in gpu_pids
assert receipt["status"] == "completed_unqualified" and receipt["new_PPO"] == 0
assert receipt["optimizer_constructed"] is False
result = dict(schema="scratch-leg-speed-terminal-review-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    original_process_exited=True, original_GPU_absent=True, identity=identity,
    matched_native_checks=23, independently_verified_RSL_values=counts,
    total_verified_RSL_values=sum(counts.values()), comparison_sha256=sha(REPORTS / "scratch_leg_encoder007_zero_comparison001.json"),
    excitation_sha256=sha(REPORTS / "scratch_leg_encoder007_physical_excitation001.json"),
    baseline=comparison["baseline_mid"], intervention=comparison["intervention_mid"],
    conclusion="Direct neural leg-speed response contributes to measured excitation. Isolation reduces leg speed and excitation but still fails strict continuous support; no unique-cause or deployable-policy claim.",
    new_PPO=0, optimizer_constructed=False, adopted=False, qualified=False, hardware_ready=False,
    reader_sha256=sha(__file__))
output = REPORTS / "scratch_leg_encoder007_terminal_review001.json"
assert not output.exists()
output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps(dict(checks=23, verified_RSL_values=counts, actual_process_exit=True)))
