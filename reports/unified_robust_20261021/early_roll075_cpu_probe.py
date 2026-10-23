"""Gate/namespace/tensor scope proof only; no changed-action native result."""
import hashlib
import json
from pathlib import Path
import sys
import math
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from early_roll075_prototype import EarlyRollPrototype, roll_gate
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import quiet_gate
from wheeled_algo.getup_symmetry import canonical_recovery_logits
from wheeled_algo.recovery_coordinates import smooth_recovery_legs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "quiet_whole_wheel074/model_migrated.pt"
    mean, _ = load_mean(checkpoint)
    directory = RUNS / "high_recenter_full066/robust"
    report = json.loads((directory / "report.json").read_text())
    path = directory / "trajectory.npz"
    assert sha(path) == report["trajectory_sha256"]
    with np.load(path) as data:
        lanes = [row["lane"] for row in report["lanes"] if not row["recovery"] and not row.get("push")]
        ordinary = torch.from_numpy(data["observations"][:, lanes].reshape(-1, 35).copy())
        alive = torch.from_numpy(data["alive"][:, lanes].reshape(-1).copy())
        sampled = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    results = {}
    assert math.sqrt(1. - .94 ** 2) < math.sin(math.radians(20.))
    with torch.inference_mode():
        reference = mean(sampled)
        for degrees in ((20., 45.), (20., 25.)):
            model = EarlyRollPrototype(mean, *degrees)
            old, new = roll_gate(mean, ordinary, *degrees)
            assert torch.equal(old[alive], new[alive])
            original, proposed = roll_gate(mean, sampled, *degrees)
            changed = proposed.ravel() > original.ravel()
            actual = model(sampled)
            assert torch.equal(actual[~changed], reference[~changed])
            assert not bool(((quiet_gate(sampled).ravel() > 0.) & changed).any())
            gravity = sampled[:, 7:10]
            norm = gravity.norm(dim=-1)
            coordinate_valid = (norm >= .5) & (norm <= 1.5) & (gravity[:, 2] / norm < -.94)
            assert not bool((coordinate_valid & changed).any())
            selected = sampled[changed]
            legacy = torch.cat((selected[:, :14], torch.zeros_like(selected[:, 14:16]), selected[:, 16:]), -1)
            raw = canonical_recovery_logits(mean.recovery, legacy, mean.feature_scale)
            legs = smooth_recovery_legs(raw[:, :4], legacy[:, 22:26], legacy[:, 10:14],
                                       mean.recovery_leg_step, mean.recovery_knee_center)
            fallen = torch.cat((legs, 9. * torch.tanh(raw[:, 4:6])), -1)
            # Independent full motion expression with only the old recovery
            # blend disabled. Changed-domain quiet is identically zero.
            with patch("wheeled_algo.robust_getup.tilt_gate", lambda obs, *_: torch.zeros_like(obs[:, :1])):
                motion = mean(selected)
            expected = motion + proposed[changed] * (fallen - motion)
            error = float((expected - actual[changed]).abs().max())
            assert error < 4e-6
            results[str(degrees)] = dict(ordinary_alive_frames=int(alive.sum()), ordinary_changed_gate_frames=0,
                sampled_frames=len(sampled), unchanged_outputs_exact=int((~changed).sum()),
                changed_gate_frames=int(changed.sum()), quiet_outputs_unchanged=True,
                reference_coordinate_valid_domain_unchanged=True,
                maximum_action_magnitude=float(actual.abs().max()), explicit_convex_blend_max_error=error)
    write(REPORTS / "early_roll075_cpu.json", dict(schema="roll075-recorded-input-proof-v1",
        checkpoint_sha256=sha(checkpoint), source_trajectory_sha256=sha(path), variants=results,
        prototype_sha256=sha(REPORTS / "early_roll075_prototype.py"), new_macro_updates=0,
        changed_action_native_behavior_proven=False,
        limitation="Only unchanged-domain and algebraic scope proven; all changed-action trajectories remain untested"))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
