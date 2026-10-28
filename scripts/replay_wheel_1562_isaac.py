#!/usr/bin/env python3
"""New gearbox candidate on native V7 USD, using the actual RL wheel component.

Independent fixed-base airborne fixtures keep the closed-chain topology and
finite nonwheel carrier locks. There is no PPO, contact fitting, or hardware IO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_cases(directory):
    sources = {key: dict(np.load(directory / f"{key}.npz")) for key in ("low", "high")}
    # Dynamic holdouts precede four extra low-speed launch diagnostics. Those
    # diagnostics probe stiction behavior and are not labeled independent data.
    pairs = [("high", 2, 12, 42), ("high", 2, 13, 43), ("high", 2, 14, 44),
             ("high", 2, 27, 57), ("high", 2, 28, 58), ("high", 2, 29, 59),
             ("low", 1, 131, 267), ("high", 3, 2, None), ("high", 3, 3, None),
             ("low", 1, 8, 150), ("low", 1, 14, 156), ("low", 1, 11, 153), ("low", 1, 17, 159)]
    cases = []
    for source, run, left, right in pairs:
        data = sources[source]
        pair = []
        for side, stage in enumerate((left, right)):
            if stage is None:
                pair.append(None)
                continue
            ii = np.flatnonzero((data["state"] == 3) & (data["run_id"] == run) & (data["stage"] == stage))
            if len(ii) < 100:
                raise ValueError("Missing native replay holdout stage")
            t = (data["control_steady_ns"][ii] - data["control_steady_ns"][ii[0]]) * 1e-9
            grid = np.arange(0., min(t[-1], 20.), .001)
            # Recorded references are held between source samples. Measured RPM
            # is interpolated only for scoring, never injected into simulation.
            source_index = np.maximum(0, np.searchsorted(t, grid, side="right") - 1)
            pair.append({"stage": stage, "time_s": grid,
                "reference": data["reference_velocity"][ii[source_index], side],
                "probe": data.get("probe_torque_model", np.zeros_like(data["velocity"]))[ii[source_index], side],
                "command": data["tau_frame_api"][ii[source_index], side],
                "measured": np.interp(grid, t, data["velocity"][ii, side]),
                "initial_velocity": float(data["velocity"][ii[0], side]),
                "coast": bool(source == "high" and (run == 2 and stage in (14, 29, 44, 59) or run == 3 and stage == 3)),
                "max_log_gap_s": float(np.max(np.diff(t))), "original_duration_s": float(t[-1])})
        cases.append({"source": source, "run": run, "axes": pair})
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "model/wheel_identified_1562_20261028/model.json")
    p.add_argument("--data", type=Path, default=ROOT / "reports/wheel_real2sim_1562_20261028")
    p.add_argument("--bundle", type=Path, default=ROOT / "model/纯底盘_v7_flat_base107/urdf")
    p.add_argument("--output", type=Path, default=ROOT / "reports/wheel_real2sim_1562_20261028/native")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.gravity = (0., 0., 0.)
    args.output.mkdir(parents=True, exist_ok=False)
    model_cfg = {"enabled": True, "model_path": str(args.model), "model_sha256": digest(args.model), "apply_aggregate_delay": True}
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    spec = json.loads((args.bundle / "model_spec.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(args.bundle / name) != expected:
            raise ValueError(f"Native asset dependency changed: {name}")
    cases = prepare_cases(args.data)
    args.instances = len(cases)
    report = {"status": "prepared", "model_sha256": model_cfg["model_sha256"],
        "asset_manifest_sha256": digest(args.bundle / "manifest.json"), "bundle": str(args.bundle),
        "fixture": "fixed root, finite carrier joint locks, original closed-chain USD",
        "coordinate_mapping": "API scalar power convention mapped +1/+1 to native wheel axes; airborne test does not qualify vehicle-forward wiring",
        "state_injection": "wheel initial velocity once per stage; no measured state thereafter",
        "device": args.device, "gravity_m_s2": list(args.gravity), "ppo_updates": 0, "scores": []}
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    launcher = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher({"headless": True, "device": args.device, "enable_cameras": False})
        import torch
        from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse, WHEEL_NAMES
        from replay_wheel_response_isaac import WheelBench
        response = IdentifiedWheelResponse(len(cases), args.device, .001, model_cfg, root=ROOT)
        native = response.native_parameters()
        params = {"armature": np.array([native["armature"][n] for n in WHEEL_NAMES]),
                  "coulomb": np.array([native["dynamic_friction"][n] for n in WHEEL_NAMES]),
                  "static_coulomb": np.array([native["frictionloss"][n] for n in WHEEL_NAMES]),
                  "viscous": np.zeros(2), "effort_limit": response.native_effort_limit}
        bench = WheelBench(args, manifest, spec, params)
        report["native_readback"] = bench.readback
        for k, name in enumerate(WHEEL_NAMES):
            row = response.model["axes"][name]
            if not np.isclose(bench.readback["cad_axial_inertia_kg_m2"][k], row["cad_inertia_kg_m2"], atol=1e-8):
                raise ValueError("Wheel model CAD component does not match the native asset")
            if not np.isclose(bench.readback["total_axial_inertia_with_armature_kg_m2"][k], row["effective_total_inertia_kg_m2"], atol=1e-8):
                raise ValueError("Native total wheel inertia was double-counted or omitted")
        report["response"] = response.summary()
        # PhysX clone order is explicitly authenticated by WheelBench.
        ordered_cases = [cases[k] for k in bench.order]
        lengths = [max(len(a["time_s"]) for a in c["axes"] if a is not None) for c in ordered_cases]
        count, horizon = len(cases), max(lengths)
        references = np.zeros((horizon, count, 2)); probes = np.zeros_like(references)
        measured = np.zeros_like(references)
        initial = np.zeros((count, 2)); coast = np.ones((count, 2), dtype=bool)
        for env, case in enumerate(ordered_cases):
            for side, axis in enumerate(case["axes"]):
                if axis is None:
                    continue
                n = len(axis["time_s"])
                for array, key in ((references, "reference"), (probes, "probe"), (measured, "measured")):
                    array[:n, env, side] = axis[key]
                    array[n:, env, side] = axis[key][-1]
                initial[env, side] = axis["initial_velocity"]
                coast[env, side] = axis["coast"]
        bench.initialize(np.zeros_like(initial), initial)
        ref = torch.tensor(references, device=args.device, dtype=torch.float32)
        probe = torch.tensor(probes, device=args.device, dtype=torch.float32)
        coast_tensor = torch.tensor(coast, device=args.device)
        efforts = torch.zeros(count, 6, device=args.device)
        velocity = torch.zeros_like(efforts)
        traces = torch.zeros(horizon, count, 2, device=args.device)
        max_drift, max_motor, max_passive = 0., 0., 0.
        for step in range(horizon):
            actual = bench.robot.data.joint_vel.torch[:, bench.ids]
            traces[step] = actual
            efforts[:, (2, 5)] = torch.where(coast_tensor, 0., .5 * (ref[step] - actual) + probe[step]).clamp(-4.5, 4.5)
            velocity[:, (2, 5)] = actual
            motor = response.apply_torque(efforts, velocity)
            bench.step(motor[:, (2, 5)] + response.passive_effort)
            if (step + 1) % 1000 == 0:
                q = bench.robot.data.joint_pos.torch
                drift = (q[:, bench.carrier_ids] - bench.nominal[:, bench.carrier_ids]).abs().max().item()
                max_drift = max(max_drift, drift)
                max_motor = max(max_motor, motor[:, (2, 5)].abs().max().item())
                max_passive = max(max_passive, response.passive_effort.abs().max().item())
                if not bool(torch.isfinite(traces[step]).all()) or traces[step].abs().max().item() > 100:
                    raise RuntimeError("Native identified wheel response diverged")
            if (step + 1) % 4000 == 0:
                print(f"Native replay {step + 1}/{horizon} steps, wall {time.monotonic()-started:.1f}s", flush=True)
        predicted = traces.cpu().numpy()
        saved = {}
        for env, case in enumerate(ordered_cases):
            for side, axis in enumerate(case["axes"]):
                if axis is None:
                    continue
                n = len(axis["time_s"]); error = predicted[:n, env, side] - axis["measured"]
                label = f"{('left','right')[side]}_{case['source']}_r{case['run']}_s{axis['stage']}"
                report["scores"].append({"label": label, "side": ("left", "right")[side],
                    "coast": axis["coast"], "duration_s": float(axis["time_s"][-1]),
                    "source_original_duration_s": axis["original_duration_s"],
                    "max_log_gap_s": axis["max_log_gap_s"],
                    "speed_rmse_rad_s": float(np.sqrt(np.mean(error ** 2))),
                    "speed_mae_rad_s": float(np.mean(abs(error))),
                    "speed_abs_error_p95_rad_s": float(np.quantile(abs(error), .95))})
                for key, value in (("time_s", axis["time_s"]), ("measured", axis["measured"]), ("predicted", predicted[:n, env, side])):
                    saved[label + "_" + key] = value
        np.savez_compressed(args.output / "traces.npz", **saved)
        report.update(status="completed", native_properties_verified=True, finite=True,
            max_carrier_drift_native_units=max_drift, sampled_max_motor_torque_nm=max_motor,
            sampled_max_extra_passive_effort_nm=max_passive, physics_steps=horizon,
            simulated_wheel_seconds=float(sum(len(a["time_s"])*.001 for c in cases for a in c["axes"] if a is not None)),
            wall_s=time.monotonic()-started,
            acceptance="Native implementation/readback verified; empirical errors reported, not a hardware release or whole-robot RL qualification")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k:report[k] for k in ("status", "physics_steps", "wall_s")}), flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", wall_s=time.monotonic()-started)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
