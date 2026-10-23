"""Matched static evaluation of a zero direct position-to-leg ablation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write
from iteration036_compare import read, paired, standing
from iteration048_runner import static_failures
from wheeled_algo.standing_reference_lifecycle import validate_observation_contract


def main():
    checkpoint = RUNS / "quiet_position_wheels063/model_migrated.pt"
    migration = json.loads((REPORTS / "iteration063_migration.json").read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == migration["checkpoint_sha256"]
    output = RUNS / "position_wheels_static063"
    progress = native("position_wheels_static063", [sys.executable, "scripts/evaluate_robust_getup_batch.py",
        "--checkpoint", str(checkpoint), "--output", str(output), "--variants", "robust", "--replicas", "16",
        "--seed", "211322", "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12",
        "--isolated-origins", "--case-names", "stand_low", "stand", "stand_high"], output)
    assert progress["status"] == "completed"
    task = json.loads((output / "task_contract.json").read_text())["settings"]
    assert validate_observation_contract(task) and task["standing_position_filter_seconds"] == .15
    assert task["standing_position_reference"] == dict(capture_delay_seconds=.05, reset_on_severe_tilt=False)
    with (REPORTS / "position_wheels_static063_analysis.log").open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    baseline, parent, after = read("quiet_velocity_baseline036"), read("position_reference_static062"), read("position_wheels_static063")
    rows = standing(after)
    failures = static_failures(rows, standing(baseline))
    result = dict(schema="position063-wheel-position-native-v1", pairing_to062=paired(parent, after),
                  pairing_to036=paired(baseline, after), before=standing(parent), after=rows,
                  baseline=standing(baseline), static_failures=failures, static_target_passed=not failures,
                  native_filter_seconds=task["standing_position_filter_seconds"],
                  native_reference=task["standing_position_reference"], evaluation_overrides=after[1]["evaluation_overrides"],
                  checkpoint_sha256=after[1]["checkpoint_sha256"], trajectory_sha256=after[1]["trajectory_sha256"],
                  new_macro_updates=0, robust_consumed=1128, lineage_clock=473, long_training_admitted=False,
                  scope="Direct position-to-leg ablation; old RL leg control and wheel position feedback remain active. Full behavior and long standing still required.")
    write(REPORTS / "iteration063_final.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
