"""One native static bias ablation; preserve failed evidence without PPO retries."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import write


def native(name, command, output):
    assert not output.exists(), "Refuse to overwrite completed or partial evidence"
    receipt = dict(schema="quiet047-native-process-v1", command=command,
                   status="starting", started=time.time(), returncode=None)
    path = REPORTS / (name + "_process.json")
    with (REPORTS / (name + ".log")).open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(pid=child.pid, status="running")
        write(path, receipt)
        print(name, "started", child.pid, flush=True)
        code = child.wait()
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    receipt.update(returncode=code, native_status=progress.get("status"), finished=time.time(),
                   status="completed" if code == 0 and progress.get("status") == "completed" else "failed")
    write(path, receipt)
    if receipt["status"] != "completed":
        raise RuntimeError(str(receipt))


def main():
    output = ROOT / "runs/unified_robust_20261022/bias_ablation_static047"
    checkpoint = ROOT / "runs/unified_robust_20261022/quiet_bias_ablation047/model_ablated.pt"
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
        "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
        "--case-names", "stand_low", "stand", "stand_high"]
    native("bias_ablation_static047", command, output)
    with (REPORTS / "bias_ablation_static047_analysis.log").open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    from iteration036_compare import read, paired, standing
    before, after = read("coupled_state_static045"), read("bias_ablation_static047")
    clean = read("quiet_velocity_baseline036")
    rows, original = standing(after), standing(clean)
    failures = [name for name, row in rows.items() if row["survived"] != row["episodes"]
                or row["full_window_max_excursion_m"] > .05
                or row["wheel_clearance_after_02s_peak_m"] > .003
                or row["leg_motion_fast_rms_rad_3_95"] > original[name]["leg_motion_fast_rms_rad_3_95"]
                or row["wheel_motion_fast_rms_rad_s_3_95"] > original[name]["wheel_motion_fast_rms_rad_s_3_95"]]
    result = dict(schema="quiet047-bias-ablation-results-v1", learned045_pairing=paired(before, after),
        clean_baseline_pairing=paired(clean, after), before=standing(before), after=rows, clean_baseline=original,
        static_failures=failures, optimizer_updates=0, robust_consumed=961,
        status="static_target_not_met" if failures else "static_only_passed",
        full_suite_complete=False, long_training_admitted=False, hardware_ready=False,
        limits="Zeroing learned affine biases is one causal ablation, not a general calibration or proof of a unique cause. No new recovery/motion evidence or deployment admission.")
    write(REPORTS / "iteration047_final.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
