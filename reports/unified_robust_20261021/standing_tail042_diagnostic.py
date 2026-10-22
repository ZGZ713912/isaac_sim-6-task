"""Quantify whether the retained policy can observe its static creep."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
directory = ROOT / "runs/unified_robust_20261022/quiet_velocity_baseline036/robust"
report = json.loads((directory / "report.json").read_text())
data = np.load(directory / "trajectory.npz")
assert hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
rows = []
for lane in report["lanes"]:
    i = lane["lane"]
    mask = data["alive"][:, i] & (data["time"][:, i] >= 3.)
    full = data["alive"][:, i] & ~data["passive"][:, i]
    position = data["local_position"][full, i, :2]
    excursion = float(np.linalg.norm(position - position[:1], axis=-1).max())
    observed = data["observations"][mask, i]
    actual = data["motor_velocity"][mask, i]
    # .06m wheel cylinders; positive left/negative right is forward rolling.
    observed_forward = .06 * (observed[:, 20] - observed[:, 21]) * 10. / 2.
    actual_forward = .06 * (actual[:, 2] - actual[:, 5]) / 2.
    vx = data["velocity"][mask, i, 0]
    rows.append(dict(lane=i, case=lane["name"], domain_enabled=lane["domain_enabled"],
        full_window_excursion_m=excursion, body_vx_mean_m_s=float(vx.mean()),
        observed_forward_wheel_proxy_mean_m_s=float(observed_forward.mean()),
        actual_forward_wheel_proxy_mean_m_s=float(actual_forward.mean()),
        observed_minus_actual_wheel_proxy_mean_m_s=float((observed_forward - actual_forward).mean()),
        observed_proxy_minus_body_vx_mean_m_s=float((observed_forward - vx).mean()),
        material=lane["material_coefficients"], first_reset_domain=lane["first_reset_domain"]))
result = dict(schema="quiet042-static-observability-v1", optimizer_updates=0,
    trajectory_sha256=report["trajectory_sha256"], wheel_radius_m=.06,
    tail_episodes=[x for x in rows if x["full_window_excursion_m"] > .05],
    case_worst={name: max((x for x in rows if x["case"] == name), key=lambda x: x["full_window_excursion_m"])
                for name in ("stand_low", "stand", "stand_high")},
    scope="Descriptive steady3..12s rolling proxy only; no world-position observation, slip model, or unique causal conclusion")
(REPORTS / "standing_tail042_diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({name: {k: v for k, v in row.items() if k not in ("first_reset_domain", "material")}
                  for name, row in result["case_worst"].items()}))
