#!/usr/bin/env python3
"""Write authenticated, existing behavior measurements through the RSL TensorBoard logger."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/robust_policy_repair_20261024"
SOURCE = REPORTS / "capability_status_20261026_read001.json"
EVENTS = REPORTS / "capability_status_20261026_events001"
RECEIPT = REPORTS / "capability_status_20261026_events001.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if EVENTS.exists() or RECEIPT.exists():
    raise FileExistsError("Preserve existing events and receipt")
data = json.loads(SOURCE.read_text())
if not data["all_checks_pass"] or sha(SOURCE.with_suffix(".reader.py")) != data["reader_sha256"]:
    raise ValueError("Unauthenticated capability observation")
values = {}
for role, label in [("current", "Scratch009"), ("previous007", "Scratch007")]:
    for group in data[role]["groups"]:
        prefix = f"CapabilityStatus/{label}/{group['height_m']:.3f}m/"
        for key, value in group.items():
            if key != "height_m":
                values[prefix + key] = (512, float(value))
for name, group in data["historical_115"]["motion"].items():
    for key in ["vx_mean_m_s", "yaw_mean_rad_s", "actual_abs_yaw_rpm", "vx_mae_m_s", "yaw_mae_rad_s",
                "episodes", "survived", "worst_max_xy_from_first_active_m", "worst_max_absolute_world_y_m"]:
        values[f"CapabilityStatus/Historical115/{name}/{key}"] = (121, float(group[key]))

run = ROOT / data["current"]["run"]
logger = Logger(log_dir=str(EVENTS), cfg=json.loads((run / "agent_config.json").read_text()),
    env_cfg=json.loads((run / "config/contract.json").read_text()), num_envs=1,
    is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
logger.init_logging_writer()
try:
    for tag, (step, value) in values.items():
        logger.writer.add_scalar(tag, value, step)
    logger.writer.flush()
finally:
    logger.writer.close()

events = EventAccumulator(str(EVENTS), size_guidance={"scalars": 0}).Reload()
if set(events.Tags()["scalars"]) != set(values):
    raise ValueError("TensorBoard tags differ from the authenticated measurements")
for tag, (step, value) in values.items():
    rows = events.Scalars(tag)
    if len(rows) != 1 or rows[0].step != step or not math.isclose(rows[0].value, value, rel_tol=2e-6, abs_tol=2e-8):
        raise ValueError(f"TensorBoard independent readback failed: {tag}")
receipt = dict(schema="capability-status-RSL-event-receipt-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), source=str(SOURCE.resolve()),
    source_sha256=sha(SOURCE), source_reader_sha256=data["reader_sha256"],
    exact_script_sha256=sha(Path(__file__)),
    step_definition={"Scratch007_and009": "actual completed fresh PPO update512",
                     "Historical115": "fixed native behavior review revision121; not a training clock"},
    verified_scalar_count=len(values), all_values_independently_read_back=True,
    event_files_sha256={str(p.relative_to(ROOT)): sha(p) for p in EVENTS.glob("events.out.tfevents.*")},
    original_training_and_evaluation_events_modified=False, tensorboard_server_started=False,
    new_native_steps=0, new_PPO=0, optimizer_constructed=False, qualified=False, hardware_ready=False)
RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(receipt, ensure_ascii=False))
