"""Inspect existing strict contact traces; never alter reward or acceptance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.utils.logger import Logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.standing_contact_reward import support_density

REPORTS = Path(__file__).parent
RUN = ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero"
EVALUATION = RUN / "evaluation"
OUTPUT = REPORTS / "scratch_contact_margin007_audit001.json"
EVENTS = REPORTS / "scratch_contact_margin007_audit001.events"
assert not OUTPUT.exists() and not EVENTS.exists()
torch.set_num_threads(2)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result_path = EVALUATION / "result.json"
result = json.loads(result_path.read_text())
trace_path = EVALUATION / "native_trace.npz"
assert sha(trace_path) == result["trace_sha256"]
sources = json.loads((RUN / "source_hashes.json").read_text())
reward_source = "src/wheeled_tasks/chassis/standing_contact_reward.py"
assert sha(ROOT / reward_source) == sources[reward_source]
assert sha(RUN / "source_snapshot" / reward_source) == sources[reward_source]
breaks_path = REPORTS / "scratch_support_breaks007_read001.json"
breaks = json.loads(breaks_path.read_text())
assert breaks["native_trace_sha256"] == sha(trace_path)
quantiles = lambda values: np.quantile(values, [0., .1, .5, .9, 1.]).tolist()
rows, previous_force, previous_clearance, phases = [], [], [], np.zeros(5, dtype=np.int64)
with np.load(trace_path) as trace:
    for lane, case in enumerate(result["cases"]):
        if case["height_m"] != .305:
            continue
        active = trace["valid"][:, lane]
        assert active.all() and active.size == 1600
        force = trace["force"][active, :, lane].reshape(-1, 2, 3)
        clearance = trace["clearance"][active, :, lane].reshape(-1, 2)
        present = (np.linalg.norm(force, axis=-1) > 2.).all(-1)
        starts = np.flatnonzero(np.diff(np.r_[True, present].astype(np.int8)) == -1)
        assert (starts > 0).all() and present[:20].all()
        assert int((~present).sum()) == case["missing_physical_ms"]
        original = next(row for row in breaks["cases"] if row["case_index"] == lane)
        assert (starts + 1).tolist() == [event["first_missing_physical_ms"] for event in original["events"]]
        failing_wheel = np.argmin(np.linalg.norm(force[starts], axis=-1), axis=-1)
        before = force[starts - 1, failing_wheel, 2]
        at_onset = force[starts, failing_wheel, 2]
        gap_before = clearance[starts - 1, failing_wheel]
        assert (before > 20.).all() and (at_onset == 0.).all()
        counts = np.bincount(starts % 5, minlength=5)
        phases += counts
        previous_force.extend(before.tolist())
        previous_clearance.extend(gap_before.tolist())
        # Independent NumPy evaluation against the actual authenticated formula.
        density_np = ((1. - np.clip(force[..., 2] / 20., 0., 1.)) ** 2
            + np.log1p((np.maximum(clearance, 0.) / .0002) ** 2)).max(-1).mean()
        actual = support_density(torch.from_numpy(force[:, None]), torch.from_numpy(clearance[:, None]))
        assert np.isclose(float(actual[0]), density_np, atol=2e-7, rtol=2e-6)
        rows.append(dict(pitch_deg=case["pitch_deg"], original_physical_ms=case["physical_ms"],
            missing_physical_ms=int((~present).sum()), interruption_onsets=len(starts),
            onset_substep_counts_1_to_5=counts.tolist(),
            failing_wheel_Fz_previous1ms_quantiles_N=quantiles(before),
            failing_wheel_Fz_at_onset_quantiles_N=quantiles(at_onset),
            failing_wheel_clearance_previous1ms_quantiles_m=quantiles(gap_before),
            force_margin_satisfied_fraction=float((force[..., 2] >= 20.).all(-1).mean()),
            mean_original_support_density=float(actual[0])))
assert len(rows) == 7 and len(previous_force) == 216
assert sum(row["missing_physical_ms"] for row in rows) == 217
metrics = dict(interruption_onsets=216., missing_physical_ms=217.,
    Fz_before_break_min_N=float(min(previous_force)),
    Fz_before_break_median_N=float(np.median(previous_force)),
    force_deficit_term_zero_before_onset_fraction=1.,
    failing_wheel_clearance_before_break_median_m=float(np.median(previous_clearance)))
metrics.update({f"onset_count_substep{index + 1}": float(count) for index, count in enumerate(phases)})
agent = json.loads((RUN / "agent_config.json").read_text())
cfg = json.loads((RUN / "config/contract.json").read_text())
logger = Logger(log_dir=str(EVENTS), cfg=agent, env_cfg=cfg, num_envs=1,
    is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
logger.init_logging_writer()
try:
    for key, value in metrics.items():
        logger.writer.add_scalar("DiagnosticScratchContactMargin/" + key, value, result["completed_updates"])
    logger.writer.flush()
finally:
    logger.writer.close()
report = dict(schema="scratch-contact-margin-evidence-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    evaluation=str(EVALUATION), completed_updates=result["completed_updates"],
    original_force_threshold_N=2., original_reward_force_margin_N=20., original_reward_clearance_scale_m=.0002,
    cases=rows, aggregate=metrics, onset_substep_counts_1_to_5=phases.tolist(),
    quantile_probabilities=[0., .1, .5, .9, 1.],
    sources_sha256={str(result_path): sha(result_path), str(trace_path): sha(trace_path),
        str(breaks_path): sha(breaks_path), reward_source: sources[reward_source]},
    reader_sha256=sha(Path(__file__)), tensorboard=dict(directory=str(EVENTS),
        files_sha256={event.name: sha(event) for event in EVENTS.glob("events.out.tfevents.*")}),
    all_trace_and_formula_checks_pass=True, new_PPO=0, native_steps=0, optimizer_constructed=False,
    behavior_qualification=False, hardware_ready=False,
    scope="Original007 warm-environment fixed evaluation, all1ms samples. The20N force-deficit term was zero "
        "one sample before every observed onset; the clearance term is separate and was not generally zero. "
        "Onsets occupy all five substeps. This neither proves a unique solver cause nor disproves neural feedback "
        "excitation. Do not change the2N acceptance, remove missing samples or blindly equate a larger weight with a fix.")
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(aggregate=metrics, onset_substep_counts_1_to_5=phases.tolist(),
    output=str(OUTPUT), all_trace_and_formula_checks_pass=True)))
