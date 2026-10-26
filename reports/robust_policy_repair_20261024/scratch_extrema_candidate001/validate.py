"""Validate the unapplied observer against real RSL reduction and frozen traces."""
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import rsl_rl.utils.logger as logger_module
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "runs/scratch_standing_20261026/pilot009_leg_velocity_separated"
spec = importlib.util.spec_from_file_location("candidate_rollout_extrema", HERE / "standing_rollout_extrema.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logger_for(name, num_envs):
    path = HERE / name
    assert not path.exists()
    logger = Logger(log_dir=str(path), cfg=read(RUN / "agent_config.json"), env_cfg=read(RUN / "config/contract.json"),
                    num_envs=num_envs, is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device="cpu")
    logger.init_logging_writer()
    return logger, path


def verify_events(path, expected, step):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
    for key, value in expected.items():
        row, = accumulator.Scalars(key)
        assert row.step == step and row.value == value, key
    return dict(directory=str(path), files_sha256={p.name: sha(p) for p in path.glob("events.out.tfevents.*")},
                verified_scalar_count=len(expected))


logger, synthetic_path = logger_for("synthetic_logger_control002.events", 2)
observer = module.StandingRolloutExtrema(2, "cpu")
native_rng = torch.get_rng_state().clone()
for step in range(128):
    dq = torch.zeros(5, 2, 4)
    if step == 0:
        dq[0, 0, 3] = 50.
    before = dq.clone()
    logger.ep_extras.append({"ScratchStanding/leg_speed_peak_rad_s": dq.abs().max()})
    observer.observe(dq)
    assert torch.equal(dq, before)
assert torch.equal(native_rng, torch.get_rng_state())
synthetic = observer.write(logger.writer, 1, 128)
logger.log(it=0, start_it=0, total_it=1, collect_time=1., learn_time=.1, loss_dict={},
           learning_rate=3e-4, action_std=torch.ones(6), rnd_weight=None, print_minimal=True)
logger.writer.flush()
logger.writer.close()
synthetic_expected = {"ScratchStandingExtrema/" + k: v for k, v in synthetic.items()}
synthetic_expected["ScratchStanding/leg_speed_peak_rad_s"] = 50. / 128
synthetic_events = verify_events(synthetic_path, synthetic_expected, 0)
assert synthetic["peak_leg_speed_rad_s"] == 50. and synthetic["leg_motor_samples_at_or_above_50_rad_s"] == 1

evaluation = ROOT / "runs/scratch_standing_20261026/eval007_final512_feedback002/evaluation"
result = read(evaluation / "result.json")
trace = evaluation / "native_trace.npz"
assert sha(trace) == result["trace_sha256"]
with np.load(trace) as data:
    assert data["valid"][:, 7:14].all()
    dq = data["dq"][:, :, 7:14, :].copy()
assert dq.shape == (1600, 5, 7, 4) and np.isfinite(dq).all()
observer = module.StandingRolloutExtrema(7, "cpu")
for frame in dq:
    observer.observe(torch.from_numpy(frame))
native = observer.snapshot(1600)
assert native["peak_leg_speed_rad_s"] == float(np.abs(dq).max())
assert native["peak_leg_speed_rad_s"] == max(c["leg_speed_peak_rad_s"] for c in result["cases"][7:14])
for axis in range(4):
    assert native[f"peak_leg{axis}_speed_rad_s"] == float(np.abs(dq[..., axis]).max())
for threshold in (3, 10, 50):
    assert native[f"leg_motor_samples_at_or_above_{threshold}_rad_s"] == int((np.abs(dq) >= threshold).sum())
logger, native_path = logger_for("native007_trace_replay002.events", 7)
written = observer.write(logger.writer, result["completed_updates"] + 1, 1600)
assert written == native
logger.writer.flush()
logger.writer.close()
native_events = verify_events(native_path, {"ScratchStandingExtrema/" + k: v for k, v in native.items()}, result["completed_updates"])

reward_name = "src/wheeled_tasks/chassis/physical_standing_reward.py"
entry_name = "scripts/train_scratch_standing.py"
original_reward = (ROOT / reward_name).read_text()
reward = original_reward.replace("    monitor = StandingContactMonitor(env)\n",
    "    from .standing_rollout_extrema import StandingRolloutExtrema\n"
    "    env.standing_rollout_extrema = StandingRolloutExtrema(env.num_envs, env.device)\n"
    "    monitor = StandingContactMonitor(env)\n")
reward = reward.replace("        speed, slew = motion_density(dq, delta)\n",
                        "        speed, slew = motion_density(dq, delta)\n        env.standing_rollout_extrema.observe(dq)\n")
original_entry = (ROOT / entry_name).read_text()
entry = original_entry.replace("            env.actor_updates = report[\"completed_updates\"]\n",
    "            report[\"last_rollout_extrema\"] = env.standing_rollout_extrema.write(\n"
    "                runner.logger.writer, report[\"completed_updates\"], args.rollout_steps)\n"
    "            env.actor_updates = report[\"completed_updates\"]\n")
assert reward != original_reward and entry != original_entry
patch = "".join(difflib.unified_diff(original_reward.splitlines(True), reward.splitlines(True),
                                  fromfile="a/" + reward_name, tofile="b/" + reward_name))
patch += "".join(difflib.unified_diff(original_entry.splitlines(True), entry.splitlines(True),
                                   fromfile="a/" + entry_name, tofile="b/" + entry_name))
patch += "".join(difflib.unified_diff([], (HERE / "standing_rollout_extrema.py").read_text().splitlines(True),
                                   fromfile="/dev/null", tofile="b/src/wheeled_tasks/chassis/standing_rollout_extrema.py"))
(HERE / "integration.patch").write_text(patch)
compile(reward, reward_name, "exec")
compile(entry, entry_name, "exec")
manifest = read(RUN / "source_hashes.json")
assert all(sha(ROOT / name) == expected for name, expected in manifest.items())
installed_logger = Path(logger_module.__file__)
(HERE / "installed_rsl_logger.py").write_bytes(installed_logger.read_bytes())
receipt = dict(schema="scratch-rollout-extrema-unapplied-candidate-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    synthetic_input="128 decisions,5 physical samples per decision,2 environments,4 leg motors; one50rad/s impulse at the first physical sample",
    actual_existing_RSL_logged_mean_rad_s=50./128, candidate_true_peak_rad_s=synthetic["peak_leg_speed_rad_s"],
    synthetic_RNG_and_input_unchanged=True, synthetic_events=synthetic_events,
    native_source=str(evaluation), native_trace_sha256=sha(trace), native_observer_result=native,
    original_actual_1ms_peak_reproduced=True, native_events=native_events,
    installed_RSL_logger_sha256=sha(installed_logger), installed_RSL_logger_copy_sha256=sha(HERE / "installed_rsl_logger.py"),
    observer_sha256=sha(HERE / "standing_rollout_extrema.py"), integration_patch_sha256=sha(HERE / "integration.patch"),
    patch_basis_sha256={reward_name: sha(ROOT / reward_name), entry_name: sha(ROOT / entry_name)},
    all264_live_training_source_bytes_preserved=True, new_native_steps=0, new_PPO=0,
    applied_to_current_training=False, native_validation=False, qualified=False, hardware_ready=False,
    scope="True rollout peak and threshold sample counts only. No action, reward, mask, model, physics, RNG or acceptance changes. Apply only after009 actually exits, validate/commit before any future fresh training.",
    reader_sha256=sha(__file__))
assert not (HERE / "validation.json").exists()
(HERE / "validation.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
print(json.dumps(receipt))
