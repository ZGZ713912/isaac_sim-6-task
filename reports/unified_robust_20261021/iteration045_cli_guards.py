"""Exercise actual startup refusals before creating any native output."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
PARENT = "runs/unified_robust_20261022/quiet_velocity036/model_migrated.pt"
MIGRATED = "runs/unified_robust_20261022/quiet_coupled_state045/model_migrated.pt"
CASES = [
    ("alias_profile_requirement", "scripts/train_robust_getup.py", ["--quiet-feedback-only"],
     "Velocity-only training requires a resumed quiet support/early-position/drift profile"),
    ("mode_without_migration", "scripts/migrate_robust_getup_contract.py",
     ["--checkpoint", PARENT, "--quiet-velocity-feedback-mode", "coupled_state"],
     "requires explicit --quiet-velocity-feedback"),
    ("repeat_migration", "scripts/migrate_robust_getup_contract.py",
     ["--checkpoint", MIGRATED, "--quiet-velocity-feedback", "--quiet-velocity-feedback-mode", "coupled_state"],
     "Add velocity feedback once"),
    ("unknown_mode", "scripts/migrate_robust_getup_contract.py",
     ["--checkpoint", PARENT, "--quiet-velocity-feedback", "--quiet-velocity-feedback-mode", "arbitrary"],
     "invalid choice"),
    ("old_lr_cap", "scripts/train_robust_getup.py", ["--actor-lr", ".001"],
     "Use bounded domain scale, friction and actor learning rate"),
]
results = []
for name, script, args, expected in CASES:
    output = Path("/tmp") / ("wheeled_coupled045_guard_" + name)
    assert not output.exists()
    process = subprocess.run([sys.executable, script, *args, "--output", str(output)],
                             cwd=ROOT, capture_output=True, text=True)
    log = process.stdout + process.stderr
    (REPORTS / f"iteration045_guard_{name}.log").write_text(log)
    result = dict(name=name, returncode=process.returncode,
                  expected_error_found=expected in log, output_absent=not output.exists())
    results.append(result)
    assert process.returncode != 0 and result["expected_error_found"] and result["output_absent"], result
receipt = dict(cases=results, optimizer_updates=0,
    harness_provenance="Uses the corrected038 error-message assertion and the new feedback-only CLI alias")
(REPORTS / "iteration045_cli_guards.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
