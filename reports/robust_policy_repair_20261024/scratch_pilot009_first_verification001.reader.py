"""Read-only authentication of009 initialization and its first real PPO update."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tensordict import TensorDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


torch.set_num_threads(2)
identity = read(REPORTS / "scratch_pilot009_launch_identity001.json")
run = Path(identity["run"])
progress = read(run / "progress.json")
initial = torch.load(run / "model_initial.pt", map_location="cpu", weights_only=False)
first = torch.load(run / "model_0.pt", map_location="cpu", weights_only=False)
old_initial = torch.load(ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero/model_initial.pt", map_location="cpu", weights_only=False)
audit = read(run / "first_update_audit.json")
revision = read(run / "source_revision.json")
manifest = read(run / "source_hashes.json")
init = read(run / "initialization.json")
stat = (Path("/proc") / str(identity["pid"]) / "stat").read_text().rsplit(") ", 1)[1].split()
gpu_pids = [int(p.strip()) for p in subprocess.check_output([
    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).splitlines() if p.strip()]
agent = read(run / "agent_config.json")
spec = importlib.util.spec_from_file_location("archived_actor_reader009", ROOT / "scripts/inspect_scratch_feedback.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
cls = reader.frozen_actor_class(agent["actor"]["class_name"], run, initial["infos"])
torch.manual_seed(261901)
frame = torch.randn(8, 35) * .1
obs = TensorDict(dict(policy=frame), batch_size=[8])
model_cfg = dict(agent["actor"])
model_cfg.pop("class_name")
actor = cls(obs, agent["obs_groups"], "actor", 6, **model_cfg)
actor.load_state_dict(first["actor_state_dict"], strict=True)
actor.eval()
changed = obs.clone()
changed["policy"][:, 16:20] += .5
with torch.no_grad():
    expected, shifted = actor(obs), actor(changed)
    native_wheels = actor.mlp(actor.get_latent(changed))[:, 4:] + changed["policy"][:, 20:22]
    jit = torch.jit.script(actor.as_jit())(frame)
    exported_forward = actor.as_onnx()(frame)
event, = list(run.glob("events.out.tfevents.*"))
raw_event = event.read_bytes()
with tempfile.TemporaryDirectory() as directory:
    p = Path(directory) / event.name
    p.write_bytes(raw_event)
    events = EventAccumulator(str(p), size_guidance={"scalars": 0}).Reload()
    tags = ["Train/mean_episode_length", "ScratchStanding/dual_support_physical_fraction",
            "ScratchStanding/leg_speed_rms_rad_s", "PrecisionStanding/height_mae_m",
            "PrecisionStanding/tilt_mean_deg", "PrecisionStanding/planar_speed_rms_m_s"]
    event_rows = {tag: [dict(step=row.step, value=row.value) for row in events.Scalars(tag)] for tag in tags}
checks = dict(
    original_native_process_live=int(stat[19]) == identity["start_ticks"],
    original_native_GPU_present=identity["pid"] in gpu_pids,
    committed_source_and_archive_match_launch=revision["commit"] == identity["commit"] and revision["archive_sha256"] == identity["source_archive_sha256"] == sha(run / "committed_source.tar"),
    all264_frozen_source_hashes_authentic=len(manifest) == 264 and all(sha(run / "source_snapshot" / name) == expected for name, expected in manifest.items()),
    checkpoint_embeds_actual_committed_source=initial["infos"]["committed_source"] == revision == first["infos"]["committed_source"],
    all_initial_update_clocks_zero=initial["iter"] == initial["infos"]["completed_updates"] == init["actor_updates"] == init["critic_updates"] == 0,
    initial_Adam_empty=not initial["optimizer_state_dict"]["state"] and init["optimizer_states"] == 0,
    checkpoint_and_teacher_initialization_forbidden=init["checkpoint_loaded"] is False and init["teacher_loaded"] is False and initial["infos"]["continuation_forbidden"] is True,
    seed_matched_fresh_actor_common_tensors_exact=all(torch.equal(value, initial["actor_state_dict"][key]) for key, value in old_initial["actor_state_dict"].items()),
    seed_matched_fresh_critic_exact=all(torch.equal(value, initial["critic_state_dict"][key]) for key, value in old_initial["critic_state_dict"].items()),
    only_extra_actor_ABI_state=set(initial["actor_state_dict"]) - set(old_initial["actor_state_dict"]) == {"leg_velocity_separated_abi"},
    initial_state_has_no_additional_learned_parameters=init["actor_trainable_parameters"] == 29772 and init["critic_trainable_parameters"] == 35585,
    first_saved_checkpoint_is_one_update=first["iter"] == first["infos"]["completed_updates"] == 1,
    all17_first_Adam_states_have12_steps=len(first["optimizer_state_dict"]["state"]) == 17 and {int(v["step"]) for v in first["optimizer_state_dict"]["state"].values()} == {12},
    first_native_audit_has12_committed_finite_calls=audit["committed_optimizer_calls"] == 12 and audit["all_recorded_values_finite"] is True and len(audit["rows"]) == 12 and all(row["optimizer_committed"] for row in audit["rows"]),
    first_native_update_changes_all_actor_critic_parameter_tensors=audit["actor_changed_parameter_tensors"] == 9 and audit["critic_changed_parameter_tensors"] == 8,
    first_native_minibatches_have8192_samples=all(row["samples"] == 8192 for row in audit["rows"]),
    actual_trained_leg_means_have_zero_direct_leg_speed_response=torch.equal(expected[:, :4], shifted[:, :4]),
    actual_trained_wheels_keep_full_current_observations=torch.equal(shifted[:, 4:], native_wheels) and not torch.equal(expected[:, 4:], shifted[:, 4:]),
    actual_trained_JIT_and_export_forward_exact=torch.equal(expected, jit) and torch.equal(expected, exported_forward),
    actual_initial_and_first_model_Adam_finite=finite(initial) and finite(first),
    training_process_checkpoint_reads_still_empty=progress["checkpoint_reads"] == [],
    original_six_training_event_tags_present_finite=all(rows and [row["step"] for row in rows] == list(range(len(rows))) and all(torch.isfinite(torch.tensor(row["value"])) for row in rows) for rows in event_rows.values()),
    original_fixed_budget_and_from_zero_launch=progress["requested_updates"] == 512 and progress["args"]["separate_leg_velocity_feedback"] is True,
)
result = dict(schema="scratch-native-first-update-verification-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    identity=identity, observed_completed_updates=progress["completed_updates"], checks=checks, all_checks_pass=all(checks.values()),
    model_initial_sha256=sha(run / "model_initial.pt"), first_model_sha256=sha(run / "model_0.pt"),
    first_audit_sha256=sha(run / "first_update_audit.json"), event_prefix_bytes=len(raw_event),
    event_prefix_sha256=hashlib.sha256(raw_event).hexdigest(), event_rows=event_rows,
    verification_checkpoint_reads_only_in_this_separate_CPU_reader=True, new_native_steps_by_reader=0,
    qualified=False, hardware_ready=False, reader_sha256=sha(__file__))
output = Path(__file__).with_name("scratch_pilot009_first_verification001.json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(all_checks_pass=result["all_checks_pass"], observed_updates=progress["completed_updates"], checks=checks)))
assert result["all_checks_pass"]
