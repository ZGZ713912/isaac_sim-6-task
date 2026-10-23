"""Read-only reachability bounds for the sampled successful applied targets."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import tilt_gate
from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.recovery_coordinates import smooth_recovery_legs
from wheeled_algo.paired_recovery import DELTA_BOUNDS, NOMINAL_DELTA, wrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
    mean, _ = load_mean(checkpoint)
    directory = RUNS / "recovery_sample_exposure092/robust"
    report = json.loads((directory / "report.json").read_text())
    analysis = json.loads((directory / "analysis.json").read_text())
    assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
    row, = [row for row in report["lanes"] if row["name"] == "moving_push_right"
            and analysis["practical_followup_by_lane"][row["lane"]]]
    lane = row["lane"]
    with np.load(directory / "trajectory.npz") as data:
        obs = torch.from_numpy(data["observations"][:, lane].copy())
        gate = tilt_gate(obs, mean.stance_aware_recovery, mean.recovery_shape_aware).ravel()
        keep = (torch.from_numpy(data["alive"][:, lane].copy()) & (gate >= .999999)
                & torch.from_numpy(data["in_motion_fall"][:, lane].copy())
                & ~torch.from_numpy(data["recovered_after_push"][:, lane].copy()))
        x = obs[keep]
        target = torch.from_numpy(data["actions"][:, lane][:, [0, 1, 3, 4, 2, 5]].copy())[keep]
    assert len(x) > 100
    measured = x[:, 10:14].unflatten(-1, (2, 2))
    measured_common = measured[..., 0] + .5 * wrap(measured[..., 1] - measured[..., 0])
    previous = .25 * x[:, 22:26].unflatten(-1, (2, 2)).mean(-1)
    prior_lead = wrap(previous - measured_common)
    assert float(prior_lead.abs().max()) < 3.
    common_lo = (prior_lead - mean.recovery_leg_step * .25).clamp(-.35, .35)
    common_hi = (prior_lead + mean.recovery_leg_step * .25).clamp(-.35, .35)
    sign = x.new_tensor([1., -1.])
    bounds, nominal = x.new_tensor(DELTA_BOUNDS), x.new_tensor(NOMINAL_DELTA)
    actual_knee = (sign * (nominal + wrap(measured[..., 1] - measured[..., 0]))).clamp(bounds[:, 0], bounds[:, 1])
    low_error, high_error = bounds[:, 0] - actual_knee, bounds[:, 1] - actual_knee
    knee_lo = actual_knee + .15 * low_error / (.15 + low_error.abs())
    knee_hi = actual_knee + .15 * high_error / (.15 + high_error.abs())

    def coordinates(action):
        pair = (.25 * action[:, :4]).unflatten(-1, (2, 2))
        return wrap(pair.mean(-1) - measured_common), sign * (nominal + pair[..., 1] - pair[..., 0])

    def distances(action):
        common, knee = coordinates(action)
        return torch.maximum(common_lo - common, common - common_hi).clamp_min(0.), torch.maximum(knee_lo - knee, knee - knee_hi).clamp_min(0.)

    with torch.inference_mode():
        legacy = x.clone(); legacy[:, 14:16] = 0.
        raw = canonical_recovery_logits(mean.recovery, legacy, mean.feature_scale)
        decoded = smooth_recovery_legs(raw[:, :4], legacy[:, 22:26], legacy[:, 10:14],
            mean.recovery_leg_step, mean.recovery_knee_center)
        decoder_action = torch.cat((decoded, 9. * raw[:, 4:].tanh()), -1)
        own_common, own_knee = distances(decoder_action)
        assert float(own_common.max()) < 2e-6 and float(own_knee.max()) < 2e-6
        assert float((mean(x) - decoder_action).abs().max()) < 4e-6
        common_violation, knee_violation = distances(target)
    unreachable = ((common_violation > 2e-6) | (knee_violation > 2e-6)).any(-1)
    side = ((-x[:, 8] - .15) / .35).clamp(0., 1.)
    value = dict(schema="recovery109-decoder-reachability-v1", checkpoint_sha256=sha(checkpoint),
        source_trajectory_sha256=report["trajectory_sha256"], lane=lane,
        scope="Actual postfall full-recovery-gate frames before recovered-after-push only",
        full_gate_frames=len(x), targets_outside_decoder_intervals=int(unreachable.sum()),
        common_limit_violation_peak_rad=float(common_violation.max()),
        knee_limit_violation_peak_rad=float(knee_violation.max()),
        common_violation_frames=int((common_violation > 2e-6).any(-1).sum()),
        knee_violation_frames=int((knee_violation > 2e-6).any(-1).sum()),
        decoder_self_bound_error_peak_rad=max(float(own_common.max()), float(own_knee.max())),
        asymmetric104_side_active_frames=int((side > 0.).sum()),
        asymmetric104_side_full_frames=int((side >= 1.).sum()),
        scope_caveat="Interval violations prove some sampled targets cannot be represented by this mean decoder on the same state; do not prove the cause of closed-loop failures or justify changing safety/mechanics",
        new_optimizer_steps=0, new_macro_updates=0, robust_consumed=1384)
    path = REPORTS / "recovery109_decoder_scope.json"
    assert not path.exists()
    write(path, value)
    writer = SummaryWriter(str(RUNS / "recovery_sample_exposure092"), filename_suffix=".recovery109")
    for key in ("full_gate_frames", "targets_outside_decoder_intervals", "common_limit_violation_peak_rad", "knee_limit_violation_peak_rad"):
        writer.add_scalar("Diagnosis/RecoveryDecoderReachability/" + key, value[key], 109)
    writer.close()
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
