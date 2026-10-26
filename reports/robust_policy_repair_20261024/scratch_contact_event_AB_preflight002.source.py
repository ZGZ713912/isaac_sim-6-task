"""CPU-only paired contract audit; no actor, optimizer or simulator construction."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.scratch_standing import agent_config

REPORTS = ROOT / "reports/robust_policy_repair_20261024"
ACTIVE = ROOT / "runs/scratch_standing_20261026/pilot006_contact_events"
CANDIDATE = REPORTS / "scratch_contact_event_zero_preflight002"
OUTPUT = REPORTS / "scratch_contact_event_AB_preflight002.json"


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
    raise ValueError("Keep the original audit immutable")

progress = read(ACTIVE / "progress.json")
args = progress["args"]
active_task, zero_task = [read(folder / "task_contract.json") for folder in (ACTIVE, CANDIDATE)]
task_differences = differences(active_task, zero_task)
active_cfg, zero_cfg = [read(folder / "config/contract.json") for folder in (ACTIVE, CANDIDATE)]
config_differences = differences(active_cfg, zero_cfg)
active_design, zero_design = [read(folder / "config/design.json") for folder in (ACTIVE, CANDIDATE)]
design_differences = differences(active_design, zero_design)
manifest, zero_manifest = [read(folder / "source_hashes.json") for folder in (ACTIVE, CANDIDATE)]
source_mismatches = []
for name, expected in manifest.items():
    for folder in (ROOT, ACTIVE / "source_snapshot", CANDIDATE / "source_snapshot"):
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            source_mismatches.append(str(path))

generated_agent = agent_config(args["seed"], args["device"], args["rollout_steps"], args["learning_rate"],
    normalize_observations=args["normalize_observations"],
    wheel_velocity_increment=args["wheel_velocity_increment"])
checks = dict(
    exact_task_difference_is_cost_and_provenance=task_differences == [
        "/mechanism_config_sha256", "/standing_contact_continuity_contract/reward_cost_per_break"],
    event_cost_values_correct=active_task["standing_contact_continuity_contract"]["reward_cost_per_break"] == .1
        and zero_task["standing_contact_continuity_contract"]["reward_cost_per_break"] == 0.,
    config_only_design_provenance_differs=config_differences == [
        "/design_preflight/file", "/design_preflight/sha256"],
    design_only_output_contract_path_differs=design_differences == ["/review_basis/task_contract"],
    design_hashes_authenticated=all(sha(folder / "config/design.json") == cfg["design_preflight"]["sha256"]
        for folder, cfg in ((ACTIVE, active_cfg), (CANDIDATE, zero_cfg))),
    equal_262_source_manifests=manifest == zero_manifest and len(manifest) == 262,
    live_and_both_frozen_sources_match=not source_mismatches,
    pure_agent_configuration_matches_active=generated_agent == read(ACTIVE / "agent_config.json"),
    seed_and_hyperparameters_preserved=args["seed"] == 261001 and args["updates"] == 512
        and args["all_heights_after"] == 512 and args["num_envs"] == 256 and args["rollout_steps"] == 128,
    bank_and_static_prior_copies_equal=all(sha(ACTIVE / name) == sha(CANDIDATE / name)
        for name in ("pose_bank.json", "static_load_reference.json", "static_load_native_pair.json")),
    active_training_has_no_checkpoint_reads=progress["checkpoint_reads"] == [],
    candidate_has_no_weights_or_live_progress=not list(CANDIDATE.glob("*.pt"))
        and not (CANDIDATE / "progress.json").exists() and not (CANDIDATE / "initialization.json").exists(),
)

report = dict(schema="scratch-contact-event-paired-preflight-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), status="prepared_not_started",
    active_run=str(ACTIVE), candidate_preflight=str(CANDIDATE),
    task_differences=task_differences, mechanism_differences=config_differences,
    design_differences=design_differences, source_mismatches=source_mismatches,
    checks=checks, all_checks_pass=all(checks.values()),
    planned_fresh_configuration=dict(seed=args["seed"], updates=512, agent=deepcopy(generated_agent)),
    source_sha256=sha(Path(__file__)), native_validation=False,
    actor_constructed=False, optimizer_constructed=False, new_PPO=0, native_steps=0,
    hardware_ready=False, adopted=False,
    prerequisite="006 completes512 updates plus its own final evaluation and actually exits; then replay005 "
        "with contact-state audit. Choose this fresh cost0 control only if the contact audit does not reveal "
        "an input or physics defect requiring repair first. No automatic launch or checkpoint continuation.",
    scope="Only the new per-break cost changes; dense support and leg-speed costs, strict first-ms100% "
        "support acceptance and fixed evaluation remain identical. Seed/architecture are planned inputs; "
        "fresh actor tensors and empty Adam still require an actual future launch audit.")
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(status=report["status"], checks=checks), ensure_ascii=False))
if not report["all_checks_pass"]:
    raise SystemExit(1)
