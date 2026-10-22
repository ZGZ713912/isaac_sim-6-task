"""Validate single-frame rolling kinematics, then one native zero-PPO test."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write, native
from iteration036_compare import read, paired, standing
from iteration048_runner import static_failures
from rolling_velocity049_probe import RollingGeometry, TABLE
from wheeled_algo.rolling_velocity import RollingBaseVelocity
from wheeled_algo.evaluation_command_feedback import StandingCommandFeedbackParent


def main():
    torch.set_num_threads(1)
    table = json.loads(TABLE.read_text())
    estimator = RollingBaseVelocity(table)
    geometry = RollingGeometry()
    baseline, old_probe = read("quiet_velocity_baseline036"), read("command_feedback_positive048")
    x = torch.from_numpy(baseline[3]["observations"][::20].reshape(-1, 35).copy())
    q = np.zeros((len(x), 6))
    q[:, [0, 1, 3, 4]] = x[:, 10:14].numpy()
    q += geometry.nominal
    dq = x[:, 16:22].numpy()[:, [0, 1, 4, 2, 3, 5]] * 10.
    expected, valid = geometry.estimate(q, dq, x[:, 4:7].numpy() * 2., x[:, 7:10].numpy())
    with torch.inference_mode():
        actual, actual_valid = estimator(x)
    assert np.array_equal(actual_valid[:, 0].numpy(), valid)
    error = float(np.abs(actual[:, 0].numpy()[valid] - expected[valid]).max())
    assert error < 2e-5
    checkpoint = RUNS / "quiet_coupled_state045/model_migrated.pt"
    mean, _ = load_mean(checkpoint)
    zero = deepcopy(mean)
    zero.parent = StandingCommandFeedbackParent(zero.parent, 0., deepcopy(estimator))
    with torch.inference_mode():
        assert torch.equal(mean(x), zero(x))
        inactive = x.clone()
        inactive[:, 0] = .5
        probe = deepcopy(mean)
        probe.parent = StandingCommandFeedbackParent(probe.parent, 2., deepcopy(estimator))
        assert torch.equal(mean(inactive), probe(inactive))
        inactive[:, 0] = 0.
        inactive[:, 9] = 1.
        assert torch.equal(mean(inactive), probe(inactive))
    plan = dict(schema="rolling049-native-plan-v1", optimizer_updates=0, gain=2.,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        geometry_table_sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest(),
        numpy_torch_real_observations=len(x), maximum_velocity_estimate_error_m_s=error,
        zero_gain_and_inactive_actions_exact=True, policy_hz=200, physics_hz=1000,
        seed=211322, replicas=16, seconds=12, domain_scale=.35, ground_friction=[.3,.5],
        native_origin_mode="isolated_overlap_v1", robust_consumed=961, long_training_admitted=False,
        hypothesis="Wheel rotation alone includes articulated balancing motion. Model-derived leg/IMU cancellation may allow slow forward-request feedback without amplifying that motion.",
        limits="No-slip estimate from existing35D only; no truth velocity, contact, history or new runtime input. Single randomized-seed diagnostic, not deployment admission.")
    write(REPORTS / "iteration049_plan.json", plan)
    name = "rolling_command_feedback049"
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
        "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
        "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
        "--standing-command-feedback-gain", "2", "--standing-command-feedback-estimator", "rolling",
        "--case-names", "stand_low", "stand", "stand_high"]
    native(name, command, output)
    with (REPORTS / (name + "_analysis.log")).open("w") as log:
        subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    evidence = read(name)
    rows = standing(evidence)
    failures = static_failures(rows, standing(baseline))
    result = dict(schema="rolling049-native-results-v1", baseline_pairing=paired(baseline, evidence),
        wheel_proxy_pairing=paired(old_probe, evidence), baseline=standing(baseline),
        wheel_proxy=standing(old_probe), rolling_estimate=rows, static_failures=failures,
        status="static_target_not_met" if failures else "diagnostic_static_passed", optimizer_updates=0,
        robust_consumed=961, long_training_admitted=False, hardware_ready=False,
        trajectory_sha256=evidence[1]["trajectory_sha256"],
        evaluation_overrides=evidence[1]["evaluation_overrides"])
    write(REPORTS / "iteration049_final.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
