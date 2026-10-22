"""Check functional retention of the selected high-height feedback, no PPO."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(REPORTS)]
from iteration040_runner import native
from iteration036_compare import read, paired, standing

selection = json.loads((REPORTS / "iteration044_selection.json").read_text())
checkpoint = ROOT / selection["checkpoint"]
output = RUNS / "selective_memory_full044"
command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
    "--output", str(output), "--variants", "robust", "--replicas", "4", "--seed", "211322",
    "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins"]
native("selective_memory_full044", command, output)
with (REPORTS / "selective_memory_full044_analysis.log").open("w") as log:
    subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
baseline, candidate = read("side_confirmation032"), read("selective_memory_full044")
match = paired(baseline, candidate)
result = dict(schema="quiet044-selective-native-v1", selection=selection, pairing=match,
    baseline_standing=standing(baseline), standing=standing(candidate),
    recovery=candidate[2]["actual_settled_attitudes"],
    motion=dict(survived=candidate[2]["motion_survived"], episodes=candidate[2]["motion_episodes"],
                practical_moving_fall_recoveries=candidate[2]["practical_moving_fall_recoveries"]),
    motion_cases={name: {k: row[k] for k in ("vx_mean_m_s", "yaw_mean_rad_s", "vx_mae_m_s", "yaw_mae_rad_s")}
        for name, row in candidate[2]["cases"].items() if name.startswith(("arc_", "forward", "reverse", "fast_"))},
    passive_motor_peak_nm=candidate[2]["passive_motor_peak_nm"], closure_peak_m=candidate[2]["max_gap_m"],
    new_optimizer_updates=0, robust_consumed=929, native_completed=True,
    long_training_admitted=False, deployment_package_replaced=False,
    scope="Functional retention experiment only. Expanded48-episode static target remains unmet and formal two-seed admission is separate.")
(REPORTS / "iteration044_final.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result), flush=True)
