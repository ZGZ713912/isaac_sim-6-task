#!/usr/bin/env python3
"""Zero-update native diagnostic of a constant, statically loaded leg target.

This does not train a policy or provide a standing controller. The existing
scratch runner constructs a fresh actor/critic and empty Adam, but evaluation
uses a declared constant target. Only the seven mid-height cases are tested.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("geometry", "static-load"), default="static-load")
    parser.add_argument("--reference", type=Path, default=ROOT / "reports/robust_policy_repair_20261024/scratch_nominal_static_load001.json")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output, args.reference = args.output.resolve(), args.reference.resolve()
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    wrapper_sha = sha(__file__)
    diagnostic_sources = {path: path.read_bytes() for path in (
        Path(__file__), ROOT / "scripts/record_scratch_physics.py")}
    if (reference["schema"] != "scratch-standing-nominal-static-load-diagnostic-v1"
            or not math.isclose(reference["height_m"], .305, rel_tol=0., abs_tol=1e-8) or reference["training_modified"]
            or not reference["within_leg_action_clip"] or not reference["within_motor_torque_limit"]):
        raise ValueError("Need the original nominal static-load calculation")
    for name, expected in reference["sources_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Static-load reference source changed: " + name)
    reference_cfg = next(ROOT / name for name in reference["sources_sha256"] if name.endswith("config/contract.json"))
    mechanics = json.loads(reference_cfg.read_text())
    loader = importlib.util.spec_from_file_location("scratch_static_probe_runner", ROOT / "scripts/train_scratch_standing.py")
    runner = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runner)
    original_prepare, original_nominal, original_evaluate = runner.prepare, runner.nominal_action, runner.evaluate
    probe_evaluation = False
    evaluation_completed = False

    def archive_diagnostic(phase):
        if not args.output.exists():
            return
        frozen = args.output / "diagnostic_sources"
        frozen.mkdir(exist_ok=True)
        for path, content in diagnostic_sources.items():
            target = frozen / path.name
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Diagnostic source archive changed")
            target.write_bytes(content)
        (frozen / "static_reference.json").write_bytes(reference_bytes)
        runner.write_json(args.output / "diagnostic_receipt.json", dict(
            schema="scratch-static-load-probe-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
            phase=phase, native_evaluation_completed=evaluation_completed,
            preflight_only=args.preflight_only, mode=args.mode, new_PPO=0, resume_forbidden=True,
            learned_policy=False, hardware_ready=False, reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
            wrapper_sha256=wrapper_sha, wrapper_unchanged=sha(__file__) == wrapper_sha,
            diagnostic_sources_sha256={str(path): hashlib.sha256(content).hexdigest()
                for path, content in diagnostic_sources.items()},
            diagnostic_sources_unchanged=all(path.read_bytes() == content
                for path, content in diagnostic_sources.items()),
            target_offset_control_rad=(reference["hypothetical_static_load_offset_rad"]
                if args.mode == "static-load" else [0.] * 6),
            diagnostic_source_directory=str(frozen),
            scope="Static initialization hypothesis; constant targets are not adopted by any training or deployment"))

    def prepare(options):
        if options.updates != 0 or not options.fixed_target_probe:
            raise ValueError("A diagnostic must never execute PPO")
        config, task = original_prepare(options)
        for key in ("asset_directory", "asset_manifest_sha256", "v5_control", "control_math_sha256", "contact_material"):
            if config[key] != mechanics[key]:
                raise ValueError("Static-load diagnostic mechanics differ: " + key)
        contract_path = options.output / "task_contract.json"
        contract = json.loads(contract_path.read_text())
        contract["diagnostic_constant_target"] = dict(mode=args.mode,
            reference_sha256=hashlib.sha256(reference_bytes).hexdigest(), source_sha256=wrapper_sha,
            adopted_by_training=False, learned_policy=False)
        runner.write_json(contract_path, contract)
        # Kit's app.close may exit before the outer Python finally runs.
        archive_diagnostic("prepared_before_native_app")
        return config, task

    def constant_target(env, height=.305):
        if height != .305:
            raise ValueError("Static-load diagnostic only calculated the mid-height target")
        from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
        nominal = original_nominal(env, height)
        # The unused fresh actor keeps the ordinary geometry initialization.
        # This offset is restricted to the explicitly fixed-target evaluation.
        if not probe_evaluation or args.mode == "geometry":
            return nominal
        offset = nominal.new_tensor(reference["hypothetical_static_load_offset_rad"])
        if offset.shape != (6,) or offset[[2, 5]].abs().max() != 0:
            raise ValueError("Expected four leg offsets and zero wheel correction")
        return nominal + offset[list(POLICY_FROM_CONTROL)] / .25

    def evaluate(env, actor, output, writer, completed, *, fixed_target=False):
        nonlocal probe_evaluation, evaluation_completed
        if completed != 0 or not fixed_target:
            raise ValueError("The static-load probe must have zero updates and constant actions")
        probe_evaluation = True
        try:
            result = original_evaluate(env, actor, output, writer, completed, fixed_target=True, heights=(.305,))
        finally:
            probe_evaluation = False
        result["diagnostic_target"] = ("constant_geometry_plus_offline_static_load_over_Kp"
            if args.mode == "static-load" else "constant_geometry")
        result["diagnostic_reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        from record_scratch_physics import record as record_physics
        result["physics_runtime"] = record_physics(env)
        result["scope"] = "Seven mid-height constant-target physics cases; no learned-policy success or robustness qualification"
        import numpy as np
        startup = []
        with np.load(output / "native_trace.npz") as trace:
            for lane, case in enumerate(result["cases"]):
                valid = trace["valid"][:, lane]
                dq = trace["dq"][valid, :, lane].reshape(-1, 4)
                force = trace["force"][valid, :, lane].reshape(-1, 2, 3)
                height = trace["position"][valid, lane, 2]
                row = dict(pitch_deg=case["pitch_deg"], observed_physical_ms=len(dq),
                    first100ms_complete=len(dq) >= 100, first200ms_complete=len(dq) >= 200,
                    leg_speed_first100ms_peak_rad_s=float(np.abs(dq[:100]).max()),
                    height_error_first200ms_peak_m=float(np.abs(height[:40] - .305).max()),
                    missing_support_first200ms=int((~(np.linalg.norm(force[:200], axis=-1) > 2.).all(-1)).sum()))
                startup.append(row)
                for key in ("leg_speed_first100ms_peak_rad_s", "height_error_first200ms_peak_m", "missing_support_first200ms"):
                    writer.add_scalar(f"ProbeStaticLoad/{args.mode}/pitch{case['pitch_deg']:+.2f}/{key}", row[key], completed)
        result["startup_diagnostic"] = startup
        writer.flush()
        runner.write_json(output / "result.json", result)
        evaluation_completed = True
        archive_diagnostic("evaluated_before_native_app_close")
        return result

    runner.prepare, runner.nominal_action, runner.evaluate = prepare, constant_target, evaluate
    sys.argv = [str(ROOT / "scripts/train_scratch_standing.py"), "--output", str(args.output),
                "--updates", "0", "--num-envs", "32", "--fixed-target-probe"]
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    try:
        runner.main()
    finally:
        archive_diagnostic("wrapper_finally")


if __name__ == "__main__":
    main()
