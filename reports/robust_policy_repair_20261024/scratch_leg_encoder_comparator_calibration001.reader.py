"""Calibrate native pair rejection on real preserved positive/negative controls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from rsl_rl.utils.logger import Logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_scratch_leg_encoder_probe import physical_pair_checks, process_matches
from inspect_scratch_support_breaks import inspect as support_audit

REPORTS = Path(__file__).parent
OUTPUT = REPORTS / "scratch_leg_encoder_comparator_calibration001.json"
EVENTS = OUTPUT.with_suffix(".events")
assert not OUTPUT.exists() and not EVENTS.exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
runs = ROOT / "runs/scratch_standing_20261026"
names = ("eval007_final512_feedback002", "eval007_final512_contact_state001", "pilot007_contact_cost_zero")
traces, sources, audits = {}, {}, {}
for name in names:
    directory = runs / name / "evaluation"
    result = json.loads((directory / "result.json").read_text())
    path = directory / "native_trace.npz"
    assert sha(path) == result["trace_sha256"]
    sources[str(path)] = sha(path)
    sources[str(directory / "result.json")] = sha(directory / "result.json")
    with np.load(path) as values:
        traces[name] = {key: values[key] for key in values.files}
    audits[name] = support_audit(directory, .305)
positive = physical_pair_checks(traces[names[0]], traces[names[1]])
negative = physical_pair_checks(traces[names[0]], traces[names[2]])
assert all(positive.values()) and not any(negative.values())
assert [audits[name]["total_missing_physical_ms"] for name in names] == [190, 190, 217]
bad = dict(traces[names[1]])
bad.pop("force")
try:
    physical_pair_checks(traces[names[0]], bad)
except ValueError:
    pass
else:
    raise AssertionError("Missing native channel was accepted")
live_identity_path = REPORTS / "scratch_pilot008_launch_identity001.json"
live_identity = json.loads(live_identity_path.read_text())
ended_identity_path = REPORTS / "scratch_pilot007_launch_identity001.json"
ended_identity = json.loads(ended_identity_path.read_text())
assert process_matches(live_identity) and not process_matches(ended_identity)
checks = dict(real_matched_fresh_observers_accepted=True,
    real_warm_vs_fresh_known_mismatch_rejected=True, missing_channel_rejected=True,
    actual_live008_identity_detected=True, actual_ended007_identity_absent=True,
    all_original_support_outcomes_recomputed=True)
run = runs / "pilot007_contact_cost_zero"
logger = Logger(log_dir=str(EVENTS), cfg=json.loads((run / "agent_config.json").read_text()),
    env_cfg=json.loads((run / "config/contract.json").read_text()), num_envs=1,
    is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
logger.init_logging_writer()
try:
    for key, value in checks.items():
        logger.writer.add_scalar("DiagnosticLegEncoderComparatorCalibration/" + key, int(value), 512)
    logger.writer.flush()
finally:
    logger.writer.close()
comparator = ROOT / "scripts/compare_scratch_leg_encoder_probe.py"
support_reader = ROOT / "scripts/inspect_scratch_support_breaks.py"
OUTPUT.with_suffix(".comparator_source.py").write_bytes(comparator.read_bytes())
OUTPUT.write_text(json.dumps(dict(schema="scratch-leg-encoder-comparator-calibration-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), checks=checks, all_checks_pass=all(checks.values()),
    matched_fresh_pair=positive, rejected_warm_vs_fresh_pair=negative,
    original_missing_ms={name: audits[name]["total_missing_physical_ms"] for name in names},
    sources_sha256=dict(sources, **{str(comparator): sha(comparator), str(support_reader): sha(support_reader),
        str(live_identity_path): sha(live_identity_path), str(ended_identity_path): sha(ended_identity_path)}),
    tensorboard=dict(directory=str(EVENTS), files_sha256={path.name: sha(path)
        for path in EVENTS.glob("events.out.tfevents.*")}), reader_sha256=sha(Path(__file__)),
    pair_check_calibrated_on_existing_native_data=True, native_leg_encoder_probe_validation=False,
    new_PPO=0, native_steps=0, optimizer_constructed=False, behavior_qualification=False, hardware_ready=False,
    scope="Read-only actual positive/negative controls for the comparator. The leg-speed intervention has not run; "
        "a working comparison tool is not evidence of a physical repair."), indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(checks=checks, all_checks_pass=True, native_leg_encoder_probe_validation=False)))
