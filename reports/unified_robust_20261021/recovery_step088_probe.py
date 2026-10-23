"""CPU scope probe for a smaller learned recovery common-angle increment."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import tilt_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
    source = RUNS / "dynamic_recovery_full079/robust/trajectory.npz"
    mean, _ = load_mean(checkpoint)
    assert mean.recovery_leg_step == .3 and mean.smooth_knee_targets
    with np.load(source) as data:
        obs = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    rng = torch.get_rng_state()
    with torch.inference_mode():
        original = mean(obs)
        gate = tilt_gate(obs, mean.stance_aware_recovery, mean.recovery_shape_aware).ravel()
        outside = gate == 0.
        checks = {}
        for step in [.3, .2, .15]:
            candidate = deepcopy(mean)
            candidate.recovery_leg_step = step
            actual = candidate(obs)
            raw_delta = actual[:, :4] - original[:, :4]
            circular_delta = torch.atan2((.25 * raw_delta).sin(), (.25 * raw_delta).cos()).abs()
            branch = raw_delta.abs().amax(-1) > 12.
            partial_branch = branch & (gate > 0.) & (gate < 1.)
            worst = int(circular_delta.amax(-1).argmax())
            item = dict(zero_recovery_authority_exact=torch.equal(original[outside], actual[outside]),
                wheel_outputs_exact=torch.equal(original[:, 4:], actual[:, 4:]),
                original_step_exact=torch.equal(original, actual) if step == .3 else None,
                weights_exact=all(torch.equal(v, candidate.state_dict()[k]) for k,v in mean.state_dict().items()),
                finite=bool(torch.isfinite(actual).all()), maximum_changed_raw=float((actual-original).abs().max()),
                maximum_circular_target_delta_rad=float(circular_delta.max()),
                branch_wrap_rows=int(branch.sum()), partial_authority_branch_wrap_rows=int(partial_branch.sum()),
                worst_recorded_input_row=worst, worst_recovery_authority=float(gate[worst]),
                common_increment_before_projection_rad=.25*step,
                common_increment_before_projection_rad_s=.25*step*200.)
            assert item["zero_recovery_authority_exact"] and item["wheel_outputs_exact"] and item["weights_exact"] and item["finite"]
            if step == .3:
                assert item["original_step_exact"]
            checks[str(step)] = item
    assert torch.equal(rng, torch.get_rng_state()) and mean.recovery_leg_step == .3
    write(REPORTS / "recovery_step088_cpu.json", dict(schema="recovery088-cpu-scope-v1", status="prototype_only",
        checkpoint_sha256=sha(checkpoint), input_trajectory_sha256=sha(source), frames=len(obs),
        zero_recovery_frames=int(outside.sum()), checks=checks, original_not_mutated=True, rng_unchanged=True,
        reason="083 broader exposure regressed random recovery;079 nominal right impact occurs after full recovery authority and fast leg rotation",
        hypothesis="Smaller learned common-angle increments may reduce active collision impulse while retaining knee-coordinate and wheel feedback",
        native_behavior_proven=False, installed_in_actor_or_cli=False, full_training_admitted=False, new_macro_updates=0))
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
