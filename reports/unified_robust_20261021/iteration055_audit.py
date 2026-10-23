"""Independent geometry/replay audit for an uninstalled observation component."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read, CASES
from standing_position054_probe import coordinate
from wheeled_algo.rolling_velocity import frozen_table_path, TABLE_SHA256
from wheeled_algo.standing_displacement import WheelAngleCoordinate, StandingDisplacement


def main():
    torch.set_num_threads(1)
    directory, report, _, data = read("all_prediction_static052")
    obs = data["observations"][1:].copy()
    nominal = np.array([-.42, .13742282595395358, .42, -.1374155762580851])
    q = np.zeros((*obs.shape[:2], 6))
    q[:, :, :4] = obs[:, :, 10:14] + nominal
    rng = np.random.default_rng(54054)
    table = json.loads(frozen_table_path().read_text())
    reference = []
    for case in report["lanes"]:
        i = case["lane"]
        wheels = data["motor_position"][:-1, i][:, [2, 5]].astype(np.float64)
        if case["first_reset_domain"]["sensor_delay_policy_ticks"]:
            wheels = np.concatenate((wheels[:1], wheels[:-1]))
        wheels += rng.normal(0., .001, wheels.shape) + rng.uniform(-.005, .005, (1, 2))
        q[:, i, 4:] = wheels
        reference.append(coordinate(q[:, i, :4], wheels, obs[:, i, 7:10], table))
    checkpoint = ROOT / "runs/unified_robust_20261022/quiet_all_prediction052/model_migrated.pt"
    mean, info = load_mean(checkpoint)
    probes = torch.from_numpy(obs[::20].reshape(-1, 35).copy())
    with torch.inference_mode():
        original_actions = mean(probes)
    saved_rng = torch.get_rng_state().clone()
    component = WheelAngleCoordinate()
    observer = StandingDisplacement(len(report["lanes"]), "cpu")
    assert torch.equal(saved_rng, torch.get_rng_state())
    with torch.inference_mode():
        values, valid = component(torch.from_numpy(q.reshape(-1, 6).astype(np.float32)),
                                  torch.from_numpy(obs[:, :, 7:10].reshape(-1, 3).copy()))
        assert valid.all()
        assert torch.equal(original_actions, mean(probes))
    values = values.reshape(*obs.shape[:2], 2)
    reference = np.stack(reference, axis=1)
    geometry_error = float(np.max(np.abs(values.mean(-1).numpy() - reference)))
    assert geometry_error < 2e-5
    filtered = []
    for t in range(len(obs)):
        yes = torch.ones(len(report["lanes"]), dtype=torch.bool)
        eligible = torch.from_numpy(data["time"][t+1] >= .2)
        result, _ = observer.update(values[t], yes, yes, eligible)
        filtered.append(result.mean(-1).numpy())
    filtered = np.stack(filtered)
    rows = []
    for case in report["lanes"]:
        i = case["lane"]
        selected = data["alive"][1:, i] & (data["time"][1:, i] >= .2)
        actual = data["local_position"][:-1, i, 0][selected].astype(np.float64)
        actual -= actual[0]
        error = filtered[:, i][selected] - actual
        rows.append(dict(lane=i, case=case["name"], max_abs_error_m=float(np.abs(error).max()),
                         final_error_m=float(error[-1]), mae_m=float(np.abs(error).mean())))
    summary = {case: dict(worst_error_m=max(row["max_abs_error_m"] for row in rows if row["case"] == case),
        mean_episode_mae_m=float(np.mean([row["mae_m"] for row in rows if row["case"] == case]))) for case in CASES}
    result = dict(schema="standing055-component-independent-replay-v1",
        source_trajectory_sha256=report["trajectory_sha256"], geometry_table_sha256=TABLE_SHA256,
        torch_vs_independent_numpy_max_coordinate_error_m=geometry_error,
        observations=len(q)*q.shape[1], old_actor_actions_bitwise_exact=len(probes), construction_rng_exact=True,
        filtered_coordinate_error=summary, filter_seconds=.05, policy_dt=.005,
        requested_anchor_start_seconds=.2, native_runs=0, optimizer_updates=0,
        installed_in_training=False, installed_in_rmcs=False,
        source_sha256=hashlib.sha256((ROOT / "src/wheeled_algo/standing_displacement.py").read_bytes()).hexdigest(),
        scope="Completed trajectory replay plus hypothetical wheel-angle noise. No closed-loop gain or policy receives this coordinate; it is not long-standing, slip, heading or hardware qualification.")
    (REPORTS / "iteration055_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with SummaryWriter(str(directory.parent), filename_suffix=".standing_component055") as writer:
        for case, metrics in summary.items():
            for name, value in metrics.items():
                writer.add_scalar(f"Diagnosis/standing_component055/{case}/{name}", value, info["completed_updates"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
