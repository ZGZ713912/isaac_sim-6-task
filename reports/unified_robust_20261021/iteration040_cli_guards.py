"""Exercise actual startup refusals before creating any native output."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
PARENT = "runs/unified_robust_20261022/quiet_velocity036/model_migrated.pt"
MIGRATED = "runs/unified_robust_20261022/quiet_wheel_memory040/model_migrated.pt"
CASES = [
    ("mode_without_migration", "scripts/migrate_robust_getup_contract.py",
     ["--checkpoint", PARENT, "--quiet-velocity-feedback-mode", "wheel_memory"],
     "requires explicit --quiet-velocity-feedback"),
    ("repeat_migration", "scripts/migrate_robust_getup_contract.py",
     ["--checkpoint", MIGRATED, "--quiet-velocity-feedback", "--quiet-velocity-feedback-mode", "wheel_memory"],
     "Add velocity feedback once"),
    ("unknown_mode", "scripts/migrate_robust_getup_contract.py",
     ["--checkpoint", PARENT, "--quiet-velocity-feedback", "--quiet-velocity-feedback-mode", "arbitrary"],
     "invalid choice"),
    ("old_lr_cap", "scripts/train_robust_getup.py", ["--actor-lr", ".001"],
     "Use bounded domain scale, friction and actor learning rate"),
]
results = []
for name, script, args, expected in CASES:
    output = Path("/tmp") / ("wheeled_memory040_guard_" + name)
    assert not output.exists()
    process = subprocess.run([sys.executable, script, *args, "--output", str(output)],
                             cwd=ROOT, capture_output=True, text=True)
    log = process.stdout + process.stderr
    (REPORTS / f"iteration040_guard_{name}.log").write_text(log)
    result = dict(name=name, returncode=process.returncode,
                  expected_error_found=expected in log, output_absent=not output.exists())
    results.append(result)
    assert process.returncode != 0 and result["expected_error_found"] and result["output_absent"], result
receipt = dict(cases=results, optimizer_updates=0,
    preflight_harness_correction="First inline guard expected the numeric LR cap in a generic error message; corrected only the assertion, no training was started")
(REPORTS / "iteration040_cli_guards.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
