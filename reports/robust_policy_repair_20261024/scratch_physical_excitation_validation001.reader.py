"""Calibrate physical-frequency units and verify actual RSL event values."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
SOURCE = ROOT / "scripts/inspect_scratch_physical_excitation.py"
OUTPUT = REPORTS / "scratch_physical_excitation_validation001.json"
assert not OUTPUT.exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location("physical_excitation_reader", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = []
for frequency in (2., 3., 94., 95., 200., 450., 451., 500.):
    time = np.arange(1000) * .001
    # Use cosine so the excluded Nyquist test has nonzero sampled amplitude.
    samples = np.repeat((2. * np.cos(2. * np.pi * frequency * time))[:, None], 4, axis=1)
    low = module.band_rms(samples, 3., 95., upper_inclusive=False)
    high = module.band_rms(samples, 95., 450., upper_inclusive=True)
    expected_low = np.sqrt(2.) if 3. <= frequency < 95. else 0.
    expected_high = np.sqrt(2.) if 95. <= frequency <= 450. else 0.
    assert abs(low - expected_low) < 1e-11 and abs(high - expected_high) < 1e-11
    rows.append(dict(frequency_Hz=frequency, physical_samples=1000,
        expected_low=expected_low, expected_high=expected_high, observed_low=low, observed_high=high))
for length in (1000, 1001):
    ticks = np.arange(length) / length
    mixed = 2. * np.cos(2. * np.pi * 10 * ticks) + 3. * np.cos(2. * np.pi * 200 * ticks) + 7.
    samples = np.repeat(mixed[:, None], 4, axis=1)
    low = module.band_rms(samples, 3., 95., upper_inclusive=False)
    high = module.band_rms(samples, 95., 450., upper_inclusive=True)
    assert abs(low - np.sqrt(2.)) < 1e-11 and abs(high - 3. / np.sqrt(2.)) < 1e-11
    rows.append(dict(physical_samples=length, mixed_signal_with_DC=True, observed_low=low, observed_high=high))
bad_inputs = (np.ones((999, 4)), np.ones((1000, 5)), np.ones(1000), np.full((1000, 4), np.nan))
for samples in bad_inputs:
    try:
        module.band_rms(samples, 95., 450., upper_inclusive=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid physical samples accepted")
report_path = REPORTS / "scratch_physical_excitation005_006_007_read001.json"
report = json.loads(report_path.read_text())
assert report["reader_sha256"] == sha(SOURCE) == sha(report_path.with_suffix(".source.py"))
assert report["all_original_metrics_and_parseval_checks_pass"]
verified = 0
for row in report["evaluations"]:
    name = Path(row["evaluation"]).parent.name
    info = report["tensorboard"][name]
    event, = Path(info["directory"]).glob("events.out.tfevents.*")
    assert sha(event) == info["files_sha256"][event.name]
    accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0}).Reload()
    expected = dict(row["means_of_sufficient_cases"], sufficient_late_cases=row["sufficient_late_cases"])
    for key, value in expected.items():
        entry, = accumulator.Scalars(f"DiagnosticScratchPhysicalExcitation/{row['height_m']:.3f}m/{key}")
        assert entry.step == row["completed_updates"]
        assert abs(entry.value - value) < max(1e-10, abs(value) * 1e-6)
        verified += 1
assert verified == 24
OUTPUT.write_text(json.dumps(dict(schema="scratch-physical-excitation-calibration-v1",
    physical_sample_dt_s=.001, known_signal_cases=rows, invalid_cases_rejected=len(bad_inputs),
    verified_actual_RSL_values=verified, all_checks_pass=True, source_sha256=sha(SOURCE),
    native_report=str(report_path), native_report_sha256=sha(report_path), reader_sha256=sha(Path(__file__)),
    new_PPO=0, native_steps=0, behavior_qualification=False, hardware_ready=False), indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(known_signal_cases=len(rows), invalid_cases_rejected=len(bad_inputs),
    verified_actual_RSL_values=verified, all_checks_pass=True)))
