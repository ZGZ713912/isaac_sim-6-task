"""Serial zero-PPO tests of coordinated forward-request feedback."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
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
from wheeled_algo.evaluation_command_feedback import StandingCommandFeedbackParent
from wheeled_algo.robust_getup import quiet_gate


def static_failures(rows, baseline):
    failures = []
    for name, row in rows.items():
        if row["survived"] != row["episodes"]:
            failures.append(name + ": termination")
        if row["full_window_max_excursion_m"] > .05:
            failures.append(name + ": displacement exceeds5cm")
        if row["wheel_clearance_after_02s_peak_m"] > .003:
            failures.append(name + ": clearance exceeds3mm")
        for key in ("leg_motion_fast_rms_rad_3_95", "wheel_motion_fast_rms_rad_s_3_95"):
            if row[key] > baseline[name][key]:
                failures.append(name + ": actual vibration regressed " + key)
    return failures


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "quiet_coupled_state045/model_migrated.pt"
    mean, _ = load_mean(checkpoint)
    old_path = RUNS / "quiet_velocity036/model_migrated.pt"
    old_mean, _ = load_mean(old_path, migration_source_root=old_path.parent / "source")
    baseline = read("quiet_velocity_baseline036")
    x = torch.from_numpy(baseline[3]["observations"][::20].reshape(-1, 35).copy())
    zero = deepcopy(mean)
    zero.parent = StandingCommandFeedbackParent(zero.parent, 0.)
    with torch.inference_mode():
        assert torch.equal(old_mean(x), mean(x))
        assert torch.equal(mean(x), zero(x))
        inactive = x.clone()
        inactive[:, 0] = .5
        probe = deepcopy(mean)
        probe.parent = StandingCommandFeedbackParent(probe.parent, 2.)
        assert torch.equal(probe(inactive), mean(inactive))
        inactive[:, 0] = 0.
        inactive[:, 9] = 1.
        assert torch.equal(probe(inactive), mean(inactive))
        delta = probe.parent.command_delta(x)
    source = ROOT.parent / "RMCS/rmcs_ws/src/rmcs_rl/models/wheel_leg/recovery_candidate.onnx"
    local = ROOT / "models/wheel_leg_flat_base_20261018/source.onnx"
    reference_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert reference_sha == hashlib.sha256(local.read_bytes()).hexdigest() == "a00d13df1d4113fe949c2a6a15b508e5c491fd1a5e6881570bf025a8d3c29c29"
    plan = dict(schema="quiet048-command-feedback-plan-v1", optimizer_updates=0,
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        reference_onnx_sha256=reference_sha, reference_current_path=str(source),
        gains=[-2., 2.], command_delta_bound_m_s=.15, real_cpu_observations=len(x),
        clean036_actions_exact=True, zero_gain_actions_exact=True, moving_and_fallen_actions_exact=True,
        observed_positive_gain_request_delta_max_abs_m_s=float(delta.abs().max()),
        native=dict(seed=211322, replicas=16, seconds=12, domain_scale=.35,
                    ground_friction=[.3,.5], policy_hz=200, physics_hz=1000,
                    origin="isolated_overlap_v1", baseline="quiet_velocity_baseline036"),
        scope="Frozen locomotion parent receives a corrected forward request; original outer request, quiet/recovery gates and quiet heads retained. Fixed gain diagnostic, not a trained candidate or deployable admission.",
        robust_consumed=961, long_training_updates=0, long_training_admitted=False)
    write(REPORTS / "iteration048_plan.json", plan)
    result = dict(schema="quiet048-command-feedback-results-v1", baseline=standing(baseline),
                  tests={}, optimizer_updates=0, robust_consumed=961, long_training_admitted=False)
    for label, gain in (("negative", -2.), ("positive", 2.)):
        name = f"command_feedback_{label}048"
        output = RUNS / name
        command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(checkpoint),
            "--output", str(output), "--variants", "robust", "--replicas", "16", "--seed", "211322",
            "--domain-scale", ".35", "--friction-min", ".3", "--seconds", "12", "--isolated-origins",
            "--standing-command-feedback-gain", str(gain), "--case-names", "stand_low", "stand", "stand_high"]
        native(name, command, output)
        with (REPORTS / (name + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / "robust"), "--tensorboard"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        evidence = read(name)
        result["tests"][label] = dict(gain=gain, paired=paired(baseline, evidence), standing=standing(evidence),
            evaluation_overrides=evidence[1]["evaluation_overrides"],
            report_sha256=hashlib.sha256((evidence[0] / "report.json").read_bytes()).hexdigest(),
            trajectory_sha256=evidence[1]["trajectory_sha256"])
        result["tests"][label]["static_failures"] = static_failures(
            result["tests"][label]["standing"], result["baseline"])
        result["tests"][label]["candidate_accepted"] = False
        write(REPORTS / "iteration048_partial.json", result)
        print(label, json.dumps(result["tests"][label]["standing"]), flush=True)
    result.update(status="completed_both_fixed_gains_rejected", new_policy_delivered=False,
        limits="Two fixed gains on one matched randomized seed; not formal recovery/motion evidence or long-training admission")
    write(REPORTS / "iteration048_final.json", result)


if __name__ == "__main__":
    main()
