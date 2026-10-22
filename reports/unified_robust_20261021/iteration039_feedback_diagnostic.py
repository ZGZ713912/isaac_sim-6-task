"""Local, read-only evidence of learned velocity dependence, not stability proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_robust_getup import load_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    torch.set_num_threads(1)
    initial_path = args.checkpoint.parent / "model_initial.pt"
    initial, _ = load_mean(initial_path)
    candidate, info = load_mean(args.checkpoint)
    data_dir = ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust"
    report = json.loads((data_dir / "report.json").read_text())
    data = np.load(data_dir / "trajectory.npz")
    x = torch.from_numpy(data["observations"][[599, 1199, 1799]].reshape(-1, 35).copy())
    zero_speed = x.clone()
    zero_speed[:, 16:22] = 0.
    results = []
    with torch.inference_mode():
        assert torch.equal(initial(zero_speed), candidate(zero_speed))
        assert torch.equal(initial(x)[:, :4], candidate(x)[:, :4])
        for name, model in (("initial", initial), ("candidate", candidate)):
            axes = {}
            for axis, direction in (("encoder_sum", [1., 1.]), ("opposed_forward_odometry_proxy", [1., -1.])):
                plus, minus = x.clone(), x.clone()
                delta = x.new_tensor(direction) * .001
                plus[:, 20:22] += delta
                minus[:, 20:22] -= delta
                derivative = (model(plus) - model(minus)) / .02
                rows = {}
                for case in ("stand_low", "stand", "stand_high"):
                    ids = [i for i in range(len(x)) if report["lanes"][i % 48]["name"] == case]
                    rows[case] = derivative[ids].mean(0).tolist()
                axes[axis] = rows
            results.append(dict(model=name, raw_action_derivative_per_encoder_rad_s=axes))
        weights = {name: dict(max_abs=float(head.weight.abs().max()), norm=float(head.weight.norm()),
                             weights=head.weight.tolist()) for name, head in candidate.quiet_velocity.items()}
    receipt = dict(schema="quiet039-local-feedback-v2", checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint=str(args.checkpoint.relative_to(ROOT)), completed_updates=info["completed_updates"],
        observations=len(x), new_optimizer_updates=0, zero_encoder_velocity_function_exact=True, fixed_input_leg_actions_exact=True,
        scope="Raw action / encoder rad/s finite differences. Forward rolling has opposite encoder signs; odometry is a proxy, not privileged base velocity or a closed-loop stability proof",
        derivatives=results, velocity_weights=weights)
    path = ROOT / "reports/unified_robust_20261021" / f"iteration039_feedback_{info['completed_updates']}_axes002.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed_updates": info["completed_updates"], "weights": {k: v["max_abs"] for k, v in weights.items()},
                      "derivatives": results}, indent=2))


if __name__ == "__main__":
    main()
