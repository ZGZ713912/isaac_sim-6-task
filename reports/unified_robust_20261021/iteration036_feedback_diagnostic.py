"""Read-only local policy derivatives on recorded static states; no simulation/PPO."""
import hashlib
import argparse
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
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluation = ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust"
    data = np.load(evaluation / "trajectory.npz")
    report = json.loads((evaluation / "report.json").read_text())
    states = torch.from_numpy(data["observations"][[599, 1199, 1799]].reshape(-1, 35).copy())
    lanes = report["lanes"]
    domain_rows = []
    for lane in lanes:
        i = lane["lane"]
        xy = data["local_position"][:, i, :2]
        excursion = float(np.linalg.norm(xy - xy[:1], axis=-1).max())
        domain_rows.append(dict(lane=i, case=lane["name"], domain_enabled=lane["domain_enabled"],
            full_excursion_m=excursion, vx_mean_m_s=lane["vx_mean_m_s"],
            domain=lane["first_reset_domain"], material=lane["material_coefficients"]))
    names = ["model_initial.pt", "model_456.pt"]
    if args.final:
        names.append("model_final.pt")
    checkpoints = [ROOT / "runs/unified_robust_20261022/pilot021_velocity_condition036" / x for x in names]
    variations = {"common_wheel_speed_rad_s": (slice(20, 22), .001, .01),
                  "common_leg_angle_rad": (slice(10, 14), .0005, .0005),
                  "pitch_gyro_rad_s": (5, .0005, .001)}
    diagnostics = []
    with torch.inference_mode():
        for checkpoint in checkpoints:
            mean, _ = load_mean(checkpoint)
            action = mean(states)
            local = {}
            for name, (columns, observation_delta, physical_delta) in variations.items():
                positive, negative = states.clone(), states.clone()
                positive[:, columns] += observation_delta
                negative[:, columns] -= observation_delta
                derivative = (mean(positive) - mean(negative)) / (2 * physical_delta)
                local[name] = {case: dict(
                    mean_raw_derivative=derivative[[j for j in range(len(states))
                        if lanes[j % len(lanes)]["name"] == case]].mean(0).tolist())
                    for case in ("stand_low", "stand", "stand_high")}
            diagnostics.append(dict(checkpoint=str(checkpoint.relative_to(ROOT)),
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                outputs=action.tolist(), local_derivatives=local))
    output = dict(schema="quiet036-readonly-feedback-v1", new_optimizer_updates=0,
        states_from=str(evaluation.relative_to(ROOT)), times_s=[3., 6., 9.], state_count=len(states),
        scope="Local raw-action finite differences, not closed-loop stability or causal attribution",
        output_order=["left_leg_1", "left_leg_2", "right_leg_1", "right_leg_2", "left_wheel", "right_wheel"],
        checkpoints=diagnostics, domain_rows=domain_rows)
    if len(diagnostics) > 1:
        initial = np.asarray(diagnostics[0]["outputs"])
        state_cases = [row["case"] for row in domain_rows] * 3
        for diagnostic in diagnostics[1:]:
            difference = np.asarray(diagnostic["outputs"]) - initial
            diagnostic["action_change_on_same_states"] = {}
            for case in ("stand_low", "stand", "stand_high"):
                delta = difference[np.asarray(state_cases) == case]
                diagnostic["action_change_on_same_states"][case] = dict(
                    mean_raw_action_change=delta.mean(0).tolist(),
                    across_state_std_of_change=delta.std(0).tolist())
    filename = "iteration036_feedback_diagnostic_final.json" if args.final else "iteration036_feedback_diagnostic.json"
    path = ROOT / "reports/unified_robust_20261021" / filename
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for case in ("stand_low", "stand", "stand_high"):
        rows = [x for x in domain_rows if x["case"] == case]
        worst = max(rows, key=lambda x: x["full_excursion_m"])
        print(case, "worst_lane", worst["lane"], "excursion_cm", round(worst["full_excursion_m"] * 100, 3),
              "sensor_motor_domain", worst["domain_enabled"], "vx", round(worst["vx_mean_m_s"], 5))
        for model in diagnostics:
            print(Path(model["checkpoint"]).name, "wheel_speed_raw_derivative",
                  [round(x, 5) for x in model["local_derivatives"]["common_wheel_speed_rad_s"][case]["mean_raw_derivative"]])


if __name__ == "__main__":
    main()
