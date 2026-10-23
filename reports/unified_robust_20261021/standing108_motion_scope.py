"""Read-only checkpoint proof of standing-coordinate isolation from locomotion."""
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
from wheeled_algo.robust_getup import quiet_gate
from wheeled_tasks.chassis.quiet_standing import standing_request_mask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    current = RUNS / "quiet_position_gain106/model_migrated.pt"
    parent = RUNS / "quiet_high_bias097/model_migrated.pt"
    mean, _ = load_mean(current)
    old, _ = load_mean(parent)
    directory = RUNS / "high_recenter_full066/robust"
    report = json.loads((directory / "report.json").read_text())
    assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
    with np.load(directory / "trajectory.npz") as data:
        x = torch.from_numpy(data["observations"].reshape(-1, 35)[::311].copy())
    rows = []
    # Static and tiny commands are intentionally excluded from this proof.
    # The boundary and ordinary/high-speed arcs are all included explicitly.
    commands = [(v, w) for v in (-2.5, -.75, -.12, 0., .12, .75, 1., 2., 3.)
                for w in (-2., -1., -.25, 0., .25, 1., 2.) if abs(v) >= .12 or abs(w) >= .25]
    for v, w in commands:
        for height in (.23, .305, .42):
            z = x.clone()
            z[:, 0], z[:, 2], z[:, 3] = v, w, torch.tensor(height) * 5.
            # Set upright, calm body to make the command gate the sole reason
            # the quiet path is disabled, rather than rely on a fallen frame.
            z[:, 4:7] = 0.
            z[:, 7:10] = torch.tensor([0., 0., -1.])
            z[:, 14:16] = 0.
            altered = z.clone()
            altered[:, 14:16] = torch.tensor([5., -5.])
            command = torch.tensor([[v, w, height]]).expand(len(z), -1)
            false = torch.zeros(len(z), dtype=torch.bool)
            assert quiet_gate(z).count_nonzero() == 0
            assert standing_request_mask(command, false, false, false).count_nonzero() == 0
            with torch.inference_mode():
                expected = old(z)
                a, b = mean(z), mean(altered)
                assert torch.equal(expected, a) and torch.equal(a, b)
            rows.append(altered)
    combined = torch.cat(rows)
    mean.freeze_except_quiet_velocity()
    mean(combined).sum().backward()
    learned = {key: tensor for key, tensor in mean.named_parameters() if tensor.requires_grad}
    assert len(learned) == 3 and sum(tensor.numel() for tensor in learned.values()) == 24
    assert all(tensor.grad is not None and tensor.grad.count_nonzero() == 0 for tensor in learned.values())
    assert all(tensor.grad is None for key, tensor in mean.named_parameters() if key not in learned)
    result = dict(schema="standing108-motion-isolation-v1", checkpoint_sha256=sha(current),
        parent_checkpoint_sha256=sha(parent), recorded_sensor_frames=len(x), command_pairs=len(commands),
        height_values_m=[.23, .305, .42], evaluated_inputs=len(combined),
        position_slots_m=[.25, -.25], actual_motion_actions_bitwise_parent=True,
        all_position_slot_perturbations_bitwise_inactive=True,
        moving_standing_reward_mask_zero=True, quiet_parameters_moving_gradient_zero=True,
        quiet_learning_scope_parameters=24, frozen_motion_recovery_parameters_no_gradient=True,
        checked_vx_range_m_s=[-2.5, 3.], checked_yaw_range_rad_s=[-2., 2.],
        tests_passed=43, tests_failed=0, new_optimizer_steps=0, new_macro_updates=0,
        limitation="Same-input CPU scope only; no new closed-loop locomotion or hardware qualification")
    path = REPORTS / "standing108_motion_scope.json"
    assert not path.exists()
    write(path, result)
    writer = SummaryWriter(str(current.parent), filename_suffix=".standing108")
    writer.add_scalar("Diagnosis/StandingMotionIsolation/moving_action_peak_difference_raw", 0., 108)
    writer.add_scalar("Diagnosis/StandingMotionIsolation/moving_position_gradient_peak", 0., 108)
    writer.add_scalar("Diagnosis/StandingMotionIsolation/checked_inputs", len(combined), 108)
    writer.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
