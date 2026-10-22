"""Recheck the authenticated original ONNX on the same48 near-origin static cases."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(REPORTS)]
from iteration040_runner import native
from iteration036_compare import read, paired, standing

output = RUNS / "deployed_static041"
command = [sys.executable, "scripts/evaluate_robust_getup_batch.py",
    "--checkpoint", str(RUNS / "pilot025_wheel_memory040/model_final.pt"),
    "--output", str(output), "--variants", "deployed", "--replicas", "16", "--seed", "211322",
    "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
    "--case-names", "stand_low", "stand", "stand_high"]
native("deployed_static041", command, output)
directory = output / "deployed"
with (REPORTS / "deployed_static041_analysis.log").open("w") as log:
    subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
report = json.loads((directory / "report.json").read_text())
analysis = json.loads((directory / "analysis.json").read_text())
trajectory = directory / "trajectory.npz"
assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == report["trajectory_sha256"]
evidence = directory, report, analysis, np.load(trajectory)
reference = read("quiet_velocity_baseline036")
match = paired(reference, evidence)
result = dict(schema="original-onnx-static041-v1", optimizer_updates=0, robust_consumed=929,
    source_onnx_sha256=hashlib.sha256((ROOT / "models/wheel_leg_flat_base_20261018/source.onnx").read_bytes()).hexdigest(),
    scope="Authenticated original ONNX under the existing native35D/200Hz simulator wrapper, including its original-comparison3raw-leg/9wheel clips; no new hardware or RMCS-path validation",
    pairing=match, original_onnx=standing(evidence), retained_quiet_baseline=standing(reference),
    trajectory_sha256=report["trajectory_sha256"], native_origin_mode=report["native_origin_mode"],
    native_completed=True, long_training_admitted=False, deployment_package_replaced=False)
(REPORTS / "iteration041_final.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result))
