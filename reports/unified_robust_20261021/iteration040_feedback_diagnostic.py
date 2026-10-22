"""Read-only local wheel recurrence; the physical plant is deliberately omitted."""
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
    checkpoint = args.checkpoint.resolve()
    torch.set_num_threads(1)
    initial, _ = load_mean(checkpoint.parent / "model_initial.pt")
    candidate, info = load_mean(checkpoint)
    directory = ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust"
    report = json.loads((directory / "report.json").read_text())
    data = np.load(directory / "trajectory.npz")
    x = torch.from_numpy(data["observations"][[599, 1199, 1799]].reshape(-1, 35).copy())
    zero = x.clone()
    zero[:, 16:22] = zero[:, 26:28] = 0.
    rows = []
    with torch.inference_mode():
        assert torch.equal(initial(zero), candidate(zero))
        assert torch.equal(initial(x)[:, :4], candidate(x)[:, :4])
        for name, model in (("initial", initial), ("candidate", candidate)):
            derivatives = {}
            for axis, direction in (("encoder_sum", [1., 1.]), ("opposed_forward_odometry_proxy", [1., -1.])):
                plus, minus = x.clone(), x.clone()
                delta = x.new_tensor(direction) * .001
                plus[:, 20:22] += delta
                minus[:, 20:22] -= delta
                derivative = (model(plus) - model(minus)) / .02
                derivatives[axis] = derivative
            columns = []
            for column in (26, 27):
                plus, minus = x.clone(), x.clone()
                plus[:, column] += .0001
                minus[:, column] -= .0001
                columns.append(((model(plus) - model(minus)) / .0002)[:, 4:6])
            jacobian = torch.stack(columns, -1).numpy()
            radius = np.abs(np.linalg.eigvals(jacobian)).max(-1)
            cases = {}
            for case in ("stand_low", "stand", "stand_high"):
                ids = [i for i in range(len(x)) if report["lanes"][i % 48]["name"] == case]
                cases[case] = dict(raw_action_derivative_per_encoder_rad_s={k: v[ids].mean(0).tolist() for k, v in derivatives.items()},
                    raw_wheel_action_jacobian_wrt_previous_raw_wheel_action_mean=jacobian[ids].mean(0).tolist(),
                    local_recurrence_spectral_radius_mean=float(radius[ids].mean()),
                    local_recurrence_spectral_radius_max=float(radius[ids].max()),
                    states_with_radius_over_one=int((radius[ids] > 1.).sum()))
            rows.append(dict(model=name, cases=cases))
        weights = {name: dict(max_abs=float(head.weight.abs().max()),
            velocity_columns_max_abs=float(head.weight[:, :2].abs().max()),
            memory_columns_max_abs=float(head.weight[:, 2:].abs().max()), weights=head.weight.tolist())
            for name, head in candidate.quiet_velocity.items()}
    result = dict(schema="quiet040-local-wheel-memory-v1", checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), completed_updates=info["completed_updates"],
        observations=len(x), new_optimizer_updates=0, zero_velocity_and_previous_wheel_input_function_exact=True,
        fixed_input_leg_actions_exact=True, derivatives=rows, weights=weights,
        scope="Local fixed-IMU/joint-state raw-action recurrence only, with the physical plant omitted. Radius is a diagnostic, not a closed-loop stability verdict. Opposite wheel encoder signs represent forward rolling.")
    output = Path(__file__).parent / f"iteration040_feedback_{info['completed_updates']}_memory001.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(completed_updates=info["completed_updates"], weights=weights,
        local_radius_max={r["model"]: {k: v["local_recurrence_spectral_radius_max"] for k, v in r["cases"].items()} for r in rows})))


if __name__ == "__main__":
    main()
