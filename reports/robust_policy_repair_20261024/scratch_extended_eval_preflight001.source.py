"""Prepare the stronger horizon from007's frozen evaluation; no simulator or model."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/robust_policy_repair_20261024"
RUN = ROOT / "runs/scratch_standing_20261026/pilot007_contact_cost_zero"
OUTPUT = REPORTS / "scratch_extended_eval_preflight001.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


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


assert not OUTPUT.exists()
manifest = read(RUN / "source_hashes.json")
assert len(manifest) == 262
assert all(sha(folder / name) == value for name, value in manifest.items()
    for folder in (ROOT, RUN / "source_snapshot"))
frozen_training = RUN / "source_snapshot/scripts/train_scratch_standing.py"
training = load("long_frozen_scratch", frozen_training)
helper_path = ROOT / "scripts/extended_scratch_standing_evaluation.py"
helper = load("long_current_helper", helper_path)
config = read(RUN / "config/contract.json")
design_path = Path(config["design_preflight"]["file"])
assert sha(design_path) == config["design_preflight"]["sha256"]
design = read(design_path)
settings = read(RUN / "task_contract.json")["settings"]
before = deepcopy((config, design, settings))
contract = helper.configure(config, design, settings, 30)
changes = [differences(a, b) for a, b in zip(before, (config, design, settings))]
directory = REPORTS / "scratch_extended_eval_preflight001"
directory.mkdir(exist_ok=False)
source = inspect.getsource(training.evaluate)
(directory / "original_evaluate.py").write_text(source)
evaluate, metadata = helper.extend_function(source, 30, vars(training), directory / "extended_evaluate.py")
contract.update(metadata)
restored = ast.parse((directory / "extended_evaluate.py").read_text())
assignment, = [node for node in restored.body[0].body if isinstance(node, ast.Assign)
    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "horizon"]
assert assignment.value.args[0].left.value == 30.
assignment.value.args[0].left.value = 8.
checks = dict(
    only_three_timeout_fields_change=changes == [["/episode_seconds"],
        ["/stages/speed/roles/train/contract/episode_seconds"], ["/episode_seconds"]],
    exact_archived_function_AST_restored=ast.dump(restored, include_attributes=False)
        == ast.dump(ast.parse(source), include_attributes=False),
    original_function_signature_preserved=inspect.signature(evaluate) == inspect.signature(training.evaluate),
    original_frozen_source_unchanged=sha(frozen_training) == manifest["scripts/train_scratch_standing.py"],
    required_native_frames=contract["policy_frames"] == 6000 and contract["physical_frames"] == 30000,
    first_millisecond_no_grace=contract["first_ms_included"] and contract["startup_grace_s"] == 0.,
    original_contact_threshold=contract["contact_threshold_n"] == 2.,
    CPU_tests_completed="42 passed" in (REPORTS / "scratch_extended_eval_cpu_tests001.log").read_text(),
    all5_invalid_CLI_configurations_rejected=all(row["exit_code"] == 2 and not row["output_created"]
        for row in read(REPORTS / "scratch_extended_eval_cli_rejections001.json")),
)
paths = [ROOT / name for name in ("scripts/evaluate_scratch_standing.py",
    "scripts/extended_scratch_standing_evaluation.py", "tests/test_extended_scratch_standing_evaluation.py")]
for path in paths:
    (directory / path.name).write_bytes(path.read_bytes())
report = dict(schema="scratch-extended-standing-preflight-v1", status="prepared_not_started",
    observed_utc=datetime.now(timezone.utc).isoformat(), source_run=str(RUN),
    source_archive_count=262, frozen_training_sha256=sha(frozen_training),
    source_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths},
    original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
    config_changes=changes[0], design_changes=changes[1], settings_changes=changes[2],
    checks=checks, all_checks_pass=all(checks.values()), duration_contract=contract,
    new_RSL_protocol_tags=["EvalScratchProtocol/duration_s", "EvalScratchProtocol/startup_grace_s",
        "EvalScratchProtocol/contact_threshold_n"],
    CPU_tests=dict(total=42, new_extended_protocol=18,
        log=str(REPORTS / "scratch_extended_eval_cpu_tests001.log")),
    CLI_rejections=str(REPORTS / "scratch_extended_eval_cli_rejections001.json"),
    native_validation=False, native_steps=0, new_PPO=0, actor_constructed=False,
    optimizer_constructed=False, checkpoint_reads=0, hardware_ready=False, behavior_qualification=False,
    prerequisite="Do not interrupt/restart007 or run native jobs in parallel. Finish its512 updates and "
        "original8s fixed evaluation, verify actual exit/seal. Use30s only after a native8s candidate passes "
        "the standing/contact foundation; source checks alone are not admission or behavior evidence.",
    scope="Explicit stronger evaluation protocol only. No train code/weights/reward/control/physics "
        "parameters are modified. Restoring the one horizon literal restores the full archived AST.",
    preflight_reader_sha256=sha(Path(__file__)))
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(checks=checks, all_checks_pass=report["all_checks_pass"]), ensure_ascii=False))
if not report["all_checks_pass"]:
    raise SystemExit(1)
