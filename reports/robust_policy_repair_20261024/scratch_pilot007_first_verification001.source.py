"""Independent CPU-only check of the already-running fresh cost-zero arm."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/robust_policy_repair_20261024"
RUN = ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero"
BASE = ROOT / "runs/scratch_standing_20261026/pilot006_contact_events"
OUTPUT = REPORTS / "scratch_pilot007_first_verification001.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(prefix + "/" + key)
            else:
                result.extend(differences(left[key], right[key], prefix + "/" + key))
        return result
    return [] if left == right else [prefix]


if OUTPUT.exists():
    raise ValueError("Preserve the earlier observation")
torch.set_num_threads(2)
progress, previous = [read(folder / "progress.json") for folder in (RUN, BASE)]
identity = read(REPORTS / "scratch_pilot007_launch_identity001.json")
pid = identity["pid"]
proc = Path("/proc") / str(pid)
start_ticks = int((proc / "stat").read_text().rsplit(") ", 1)[1].split()[19])
initial, control = [torch.load(folder / "model_initial.pt", map_location="cpu", weights_only=False)
                    for folder in (RUN, BASE)]
init = read(RUN / "initialization.json")
audit = read(RUN / "first_update_audit.json")
manifest = read(RUN / "source_hashes.json")
mismatches = [str(folder / name) for name, value in manifest.items()
              for folder in (ROOT, RUN / "source_snapshot") if sha(folder / name) != value]
same_initial = {name: initial[name].keys() == control[name].keys()
                and all(torch.equal(value, control[name][key]) for key, value in initial[name].items())
                for name in ("actor_state_dict", "critic_state_dict")}
same_hashes = {}
from wheeled_algo.scratch_standing import tensor_hash
for role in ("actor", "critic"):
    same_hashes[role] = tensor_hash(initial[role + "_state_dict"]) == init[role + "_tensor_sha256"]
task_diff = differences(read(BASE / "task_contract.json"), read(RUN / "task_contract.json"))
arg_diff = differences(previous["args"], progress["args"])
event, = RUN.glob("events.out.tfevents.*")
raw = event.read_bytes()
with tempfile.TemporaryDirectory() as directory:
    prefix = Path(directory) / event.name
    prefix.write_bytes(raw)
    events = EventAccumulator(str(prefix), size_guidance={"scalars": 0}).Reload()
    names = ["StandingContinuity/" + name for name in
             ("support_break_events_per_policy_step", "paid_penalty_per_policy_step", "reward_cost_per_break")]
    scalar = {name: {row.step: row.value for row in events.Scalars(name)} for name in names}
    steps = sorted(set.intersection(*(set(values) for values in scalar.values())))
    assert steps
    zero_paid = all(scalar[names[1]][step] == scalar[names[2]][step] == 0. for step in steps)
    events_present = any(scalar[names[0]][step] > 0 for step in steps)

checks = dict(
    original_native_process=start_ticks == identity["start_ticks"],
    fresh_clock_zero=initial["iter"] == init["actor_updates"] == init["critic_updates"] == 0,
    empty_initial_Adam=not initial["optimizer_state_dict"]["state"] and init["optimizer_states"] == 0,
    training_checkpoint_reads_zero=progress["checkpoint_reads"] == [],
    no_checkpoint_or_teacher_loaded=not init["checkpoint_loaded"] and not init["teacher_loaded"],
    same_fresh_actor_and_critic_as006_initial=all(same_initial.values()),
    initial_tensor_hashes_match=all(same_hashes.values()),
    all_262_frozen_and_live_sources_match=len(manifest) == 262 and not mismatches
        and manifest == read(BASE / "source_hashes.json"),
    first_update_12_Adam_calls_finite=audit["committed_optimizer_calls"] == 12
        and audit["all_recorded_values_finite"] and audit["closed"],
    actor_and_critic_parameters_updated=audit["actor_changed_parameter_tensors"] == 9
        and audit["critic_changed_parameter_tensors"] == 8,
    only_reward_cost_and_output_args_change=arg_diff == ["/contact_break_cost", "/output"],
    actual_task_only_cost_and_provenance_change=task_diff == ["/mechanism_config_sha256",
        "/standing_contact_continuity_contract/reward_cost_per_break"],
    agent_config_unchanged=read(BASE / "agent_config.json") == read(RUN / "agent_config.json"),
    actual_event_cost_zero_with_events_observed=zero_paid and events_present,
    event_contract_matches=progress["standing_contact_continuity_contract"]
        == read(RUN / "task_contract.json")["standing_contact_continuity_contract"],
)
report = dict(schema="scratch-standing-first-native-verification-v3",
    observed_utc=datetime.now(timezone.utc).isoformat(), run=str(RUN), pid=pid, start_ticks=start_ticks,
    completed_updates_observed=progress["completed_updates"], source_count=len(manifest),
    checks=checks, all_checks_pass=all(checks.values()), source_mismatches=mismatches,
    source_sha256=sha(Path(__file__)), task_differences=task_diff, argument_differences=arg_diff,
    actor_initial_sha256=init["actor_tensor_sha256"], critic_initial_sha256=init["critic_tensor_sha256"],
    independent_initial_file_comparison_only=True, comparison_did_not_assign_weights_into_training=True,
    training_reward_payment_event_check=dict(logged_steps=steps, event_prefix_bytes=len(raw),
        event_prefix_sha256=hashlib.sha256(raw).hexdigest(), zero_penalty_confirmed=zero_paid,
        break_events_still_measured=events_present, new_tags=names),
    artifacts_sha256={name: sha(RUN / name) for name in ("model_initial.pt", "first_update_audit.json",
        "initialization.json", "source_hashes.json", "task_contract.json", "agent_config.json")},
    new_PPO_by_reader=0, optimizer_constructed_by_reader=False,
    hardware_ready=False, behavior_qualification=False)
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(completed_updates_observed=report["completed_updates_observed"], checks=checks),
                 ensure_ascii=False))
if not report["all_checks_pass"]:
    raise SystemExit(1)
