#!/usr/bin/env python3
"""Diagnostic only: compare the low-demand feedback gate with a full low-demand floor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def diagnostic_ablation(mean, variant, parent=None):
    """Keep every parameter exact; only replace the visible-input gate in memory."""
    import torch
    from wheeled_algo.separated_feedback import SeparatedFeedbackActor
    if type(mean) is not SeparatedFeedbackActor or parent is not None:
        raise ValueError("Gate isolation requires the separated mean and no parent")
    if variant == "low_gate_floor":
        original = mean.tracking_feedback_gate
        def changed_gate(obs):
            demand = torch.maximum(obs[:, :1].abs() / .75, obs[:, 2:3].abs() / 1.5)
            return torch.maximum(original(obs), (2. - demand).clamp(0., 1.))
        mean.tracking_feedback_gate = changed_gate
    elif variant != "full":
        raise ValueError("Unknown low-gate diagnostic")
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deployment-pd", action="store_true", help="Separate 1000 Hz feedback-PD replay; actor stays 50 Hz")
    parser.add_argument("--rmcs-command-reference", action="store_true",
                        help="Additional ordinary-RL replay of direct planar commands, heading hold and bounded height reference")
    parser.add_argument("--packaged-default-height", action="store_true",
                        help="Additional RMCS cohort: moving requests use the authenticated YAML default body height")
    parser.add_argument("--packaged-maximum-height", action="store_true",
                        help="Additional RMCS cohort: moving requests use the authenticated YAML maximum body height")
    parser.add_argument("--rmcs", type=Path, default=Path("/home/noir/Documents/workspace/RMCS"))
    parser.add_argument("--envelope", action="store_true", help="Additional ±3 m/s, ±4 rad/s, curves, height and stop checks")
    parser.add_argument("--reward-audit", action="store_true", help="Read-only per-case reward payments and physical phases")
    parser.add_argument("--diagnostic-ablation", choices=("full", "low_gate_floor"), default="full",
                        help="Counterfactual replay only; cannot qualify a release")
    parser.add_argument("--diagnostic-parent", type=Path)
    parser.add_argument("--diagnostic-seconds", type=float, default=4.)
    parser.add_argument("--diagnostic-training-pd", action="store_true",
                        help="Diagnostic only: keep 200 Hz PD while using the same RMCS command math")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    args = parser.parse_args()
    if args.diagnostic_training_pd and (not args.rmcs_command_reference or not args.diagnostic_ablation
                                       or args.deployment_pd):
        parser.error("Training-PD isolation requires RMCS reference, diagnostic ablation, and no deployment-PD override")
    if args.rmcs_command_reference and not (args.deployment_pd or args.diagnostic_training_pd):
        parser.error("RMCS command-reference replay requires --deployment-pd")
    if args.packaged_default_height and (not args.rmcs_command_reference or args.diagnostic_ablation):
        parser.error("Packaged default-height coverage requires ordinary RMCS reference replay without ablation")
    if args.packaged_maximum_height and (not args.rmcs_command_reference or args.diagnostic_ablation
                                        or args.packaged_default_height):
        parser.error("Packaged maximum-height coverage requires ordinary RMCS reference replay and no default-height override")
    if args.diagnostic_ablation and (not args.checkpoint or not 2. <= args.diagnostic_seconds <= 10.):
        parser.error("A diagnostic ablation requires a checkpoint and a 2–10 second horizon")
    if bool(args.diagnostic_parent) != (args.diagnostic_ablation == "parent_command"):
        parser.error("Only parent_command ablation requires --diagnostic-parent")
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle, encode, digest
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from train_chassis import preflight

    plan = read_plan(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config, seed = write_bundle(ROOT, args.output / "config", 32, evaluation=True, seed=args.seed)
    if args.envelope:
        names = [*plan["pilot_case_names"], "forward_1", "backward_1", "forward_2", "backward_2",
                 "forward_3", "backward_3", "rotate_4", "rotate_4_reverse", "curve_low", "curve_low_reverse",
                 "height_scan", "start_stop_1", "start_stop_2", "start_stop_3"]
        catalogue = json.loads((ROOT / plan["source_worker_file"]).read_text())["evaluation"]["cases"]
        settings = config["evaluation"]
        settings["cases"] = [next(c for c in catalogue if c["name"] == name) for name in names]
        settings["protocol_id"] = "flat-base-paired107-envelope-v2"
        settings["episode_seconds"] = max(c["episode_seconds"] for c in settings["cases"])
        for case in settings["cases"]:
            for key in ("height_mae_m_max", "velocity_mae_m_s_max", "yaw_mae_rad_s_max"):
                case[key] = plan["acceptance"][key]
            case["success_rate_min"] = 1.
            case["stand_drift_m_max"] = plan["acceptance"]["stand_drift_m_max"]
        design_path = args.output / "config/design.json"
        design = json.loads(design_path.read_text())
        row = design["stages"]["speed"]["roles"]["quick"]
        count = 4 * len(names)
        for mapping in (config, row["contract"]):
            mapping.update(evaluation=settings, target_num_envs=count, episode_seconds=settings["episode_seconds"],
                flat_floor_width_m=8.,
                ppo_minibatch_samples=count * plan["num_steps_per_env"] // plan["num_mini_batches"],
                scene_groups=[dict(name=n, fraction=1 / len(names), terrain=["flat"]) for n in names])
        row["invocation"]["num_envs"] = count
        design["scope"] = "flat_base_paired107_extended_command_envelope"
        design_path.write_bytes(encode(design))
        config["design_preflight"]["sha256"] = digest(design_path)
        (args.output / "config/contract.json").write_bytes(encode(config))
    reference_contract = None
    if args.rmcs_command_reference:
        from flat_rmcs_command_reference import read_reference_contract, PROTOCOL_FIXED, PROTOCOL_ENVELOPE
        reference_contract = read_reference_contract(args.rmcs)
        design_path = args.output / "config/design.json"
        design = json.loads(design_path.read_text())
        row = design["stages"]["speed"]["roles"]["quick"]
        settings = config["evaluation"]
        settings["protocol_id"] = PROTOCOL_ENVELOPE if args.envelope else PROTOCOL_FIXED
        for mapping in (config, row["contract"]):
            mapping["evaluation"] = settings
            mapping["rmcs_command_reference_replay"] = reference_contract
        design["scope"] = "flat_base_paired107_rmcs_command_reference_replay"
        design_path.write_bytes(encode(design))
        config["design_preflight"]["sha256"] = digest(design_path)
        (args.output / "config/contract.json").write_bytes(encode(config))
    if args.deployment_pd:
        # Independently frozen deployment timing source. This creates another
        # evaluation contract; it never changes a training checkpoint's clock.
        reference = ROOT / "contracts/v6_wheel_response_control_v1.json"
        expected_sha = "cc30b2b0572999224c8feb79ffa7afb3df1b8a28800bca4a153663b9ea8d4bf8"
        if digest(reference) != expected_sha:
            raise ValueError("Frozen 1000 Hz deployment control source changed")
        design_path = args.output / "config/design.json"
        design = json.loads(design_path.read_text())
        row = design["stages"]["speed"]["roles"]["quick"]
        for mapping in (config, row["contract"]):
            mapping.update(physics_dt=.001, control_math_source="contracts/v6_wheel_response_control_v1.json",
                control_math_sha256=expected_sha)
        reference_data = json.loads(reference.read_text())
        row["control"]["timing"] = reference_data["timing"]
        row["runtime"].update(reference_data["timing"])
        design["scope"] = ("flat_base_paired107_rmcs_command_reference_replay" if reference_contract is not None
                           else "flat_base_paired107_1000Hz_deployment_replay")
        design_path.write_bytes(encode(design))
        config["design_preflight"]["sha256"] = digest(design_path)
        (args.output / "config/contract.json").write_bytes(encode(config))
    default_height_contract = None
    if args.packaged_default_height:
        from flat_default_height_profile import read_default_height_contract, default_height_settings
        default_height_contract = read_default_height_contract(args.rmcs, reference_contract)
        design_path = args.output / "config/design.json"
        design = json.loads(design_path.read_text())
        row = design["stages"]["speed"]["roles"]["quick"]
        settings = default_height_settings(config["evaluation"], default_height_contract)
        for mapping in (config, row["contract"]):
            mapping.update(evaluation=settings, rmcs_default_height_replay=default_height_contract)
        design_path.write_bytes(encode(design))
        config["design_preflight"]["sha256"] = digest(design_path)
        (args.output / "config/contract.json").write_bytes(encode(config))
    maximum_height_contract = None
    if args.packaged_maximum_height:
        from flat_maximum_height_profile import read_maximum_height_contract, maximum_height_settings
        maximum_height_contract = read_maximum_height_contract(args.rmcs, reference_contract)
        design_path = args.output / "config/design.json"
        design = json.loads(design_path.read_text())
        row = design["stages"]["speed"]["roles"]["quick"]
        settings = maximum_height_settings(config["evaluation"], maximum_height_contract)
        for mapping in (config, row["contract"]):
            mapping.update(evaluation=settings, rmcs_maximum_height_replay=maximum_height_contract)
        design_path.write_bytes(encode(design))
        config["design_preflight"]["sha256"] = digest(design_path)
        (args.output / "config/contract.json").write_bytes(encode(config))
    if args.diagnostic_ablation:
        design_path = args.output / "config/design.json"
        design = json.loads(design_path.read_text())
        row = design["stages"]["speed"]["roles"]["quick"]
        settings = config["evaluation"]
        settings.update(protocol_id="flat-base-paired107-low-gate-counterfactual-v1",
                        episode_seconds=args.diagnostic_seconds, diagnostic_only=True,
                        ablation=args.diagnostic_ablation)
        for case in settings["cases"]:
            case["episode_seconds"] = args.diagnostic_seconds
        for mapping in (config, row["contract"]):
            mapping.update(evaluation=settings, episode_seconds=args.diagnostic_seconds)
        design["scope"] = "flat_base_paired107_short_counterfactual_not_release_evidence"
        design_path.write_bytes(encode(design))
        config["design_preflight"]["sha256"] = digest(design_path)
        (args.output / "config/contract.json").write_bytes(encode(config))
    _, manifest = preflight(args.output / "config/contract.json")
    control = load_chassis_control(ROOT / config["control_math_source"])
    count = config["target_num_envs"]
    static = design_preflight(config, control, ROOT, stage="speed", role="quick", num_envs=count, seed=seed)
    report = dict(schema="flat-base-behavior-evaluation-v1", status="starting", seed=seed,
        started_at=datetime.now(timezone.utc).isoformat(), new_optimizer_updates=0,
        source_onnx_sha256=plan["source_onnx_sha256"], plan_sha256=digest(ROOT / "contracts/flat_base_finetune_v1.json"),
        asset_manifest_sha256=config["asset_manifest_sha256"], model_spec_sha256=plan["model_spec_sha256"],
        contract_sha256=digest(args.output / "config/contract.json"), static_preflight=static,
        settings=config["evaluation"], policy_hz=50, physics_hz=round(1 / config["physics_dt"]),
        scope="extended_flat_command_envelope" if args.envelope else "eight_case_first_episode_diagnostic",
        hardware_deployment_ready=False)
    if reference_contract is not None:
        report["rmcs_command_reference"] = reference_contract
        report["scope"] = "ordinary_rmcs_command_reference_after_takeover_native_flat_simulation"
    if default_height_contract is not None:
        report["packaged_default_height"] = default_height_contract
        report["scope"] = "ordinary_rmcs_packaged_default_height_native_flat_simulation"
    if maximum_height_contract is not None:
        report["packaged_maximum_height"] = maximum_height_contract
        report["scope"] = "ordinary_rmcs_packaged_maximum_height_native_flat_simulation"
    sources = list((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
    sources += [Path(__file__), ROOT / "src/wheeled_algo/onnx_actor_import.py",
                ROOT / "src/wheeled_algo/command_residual.py", ROOT / "src/wheeled_algo/guarded_feedback_residual.py",
                ROOT / "src/wheeled_algo/separated_feedback.py"]
    if args.reward_audit:
        sources.append(ROOT / "scripts/flat_reward_audit.py")
    if reference_contract is not None:
        sources.append(ROOT / "scripts/flat_rmcs_command_reference.py")
    if default_height_contract is not None or maximum_height_contract is not None:
        sources.append(ROOT / "scripts/flat_default_height_profile.py")
    if maximum_height_contract is not None:
        sources.append(ROOT / "scripts/flat_maximum_height_profile.py")
    report["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for path in sources:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (args.output / "evaluation.json").write_bytes(encode(report))
    if args.preflight_only:
        report["status"] = "static_preflight_passed"
        (args.output / "evaluation.json").write_bytes(encode(report))
        print("FLAT_STATIC_PREFLIGHT", json.dumps(dict(num_envs=count, policy_hz=50,
            physics_hz=report["physics_hz"], case_count=len(config["evaluation"]["cases"]))))
        return 0
    launcher, env = None, None
    start = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_algo.onnx_actor_import import recover_actor
        from wheeled_algo.command_residual import mean_from_checkpoint
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.chassis.evaluation import grade_fixed_suite, RequestProtocolMetrics
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        actor, recovered = recover_actor(ROOT / plan["actor_bundle"] / "source.onnx")
        step = 0
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            info = state["infos"]
            if info.get("flat_base_source_onnx_sha256") != plan["source_onnx_sha256"]:
                raise ValueError("Candidate is not a descendant of the supplied flat actor")
            if info.get("asset_manifest_sha256") != config["asset_manifest_sha256"]:
                raise ValueError("Candidate mechanical asset differs")
            actor = mean_from_checkpoint(state, ROOT / plan["actor_bundle"] / "source.onnx")
            step = info["successful_updates_total"]
            report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
                          actor_adaptation=info.get("actor_adaptation"),
                          feedback_branch_actor_updates=info.get("feedback_branch_actor_updates"),
                          active_branch_actor_updates=info.get("active_branch_actor_updates"))
        if args.diagnostic_ablation:
            parent = None
            if args.diagnostic_parent:
                parent_state = torch.load(args.diagnostic_parent, map_location="cpu", weights_only=True)
                if (parent_state["infos"]["flat_base_source_onnx_sha256"] != plan["source_onnx_sha256"]
                        or parent_state["infos"]["asset_manifest_sha256"] != config["asset_manifest_sha256"]):
                    raise ValueError("Diagnostic parent source or mechanical asset differs")
                parent = mean_from_checkpoint(parent_state, ROOT / plan["actor_bundle"] / "source.onnx")
            actor = diagnostic_ablation(actor, args.diagnostic_ablation, parent)
            report["diagnostic_counterfactual"] = dict(variant=args.diagnostic_ablation,
                qualification_eligible=False, horizon_s=args.diagnostic_seconds,
                intervention="low-demand gate rows only; base, command and leg feedback retained",
                parent_checkpoint_sha256=digest(args.diagnostic_parent) if args.diagnostic_parent else None,
                checkpoint_saved=False, optimizer_updates=0)
        actor = actor.to(args.device).eval()
        env = FixedCaseEnv(config, manifest, control, ROOT, stage_name="speed", num_envs=count,
            device=args.device, seed=seed, level=1., run_role="quick")
        command_reference = None
        if reference_contract is not None:
            from flat_rmcs_command_reference import install_reference
            command_reference = install_reference(env, reference_contract,
                diagnostic_training_clock=args.diagnostic_training_pd)
            if args.diagnostic_training_pd:
                report["diagnostic_counterfactual"].update(physical_pd_hz=200,
                    rmcs_reference_pd_hz=reference_contract["physics_hz"],
                    command_math="unchanged RMCS reference", feedback_clock_isolation=True)
        report["startup"] = env.startup_report
        observations = env.reset_suite()
        settings = config["evaluation"]
        metrics = EpisodeMetrics(env.scene_groups, args.device, config["policy_dt"], settings["warmup_seconds"],
            warmup_by_group={c["name"]: c.get("warmup_seconds", settings["warmup_seconds"]) for c in settings["cases"]},
            height_range_m=[.23, .43])
        requests = RequestProtocolMetrics(env.scene_groups, args.device, config["policy_dt"])
        audit = None
        if args.reward_audit:
            from flat_reward_audit import FlatRewardAudit
            audit = FlatRewardAudit(env.scene_groups, args.device, config["policy_dt"], settings["warmup_seconds"])
        alive = torch.ones(count, dtype=torch.bool, device=args.device)
        trace_keys = ("active", "observations", "actions", "episode_ticks", "done", "commands", "velocity", "omega",
                      "height", "position", "gravity", "gap", "motor_effort", "motor_position", "motor_velocity")
        if audit is not None:
            trace_keys += ("physical_phase",)
        if command_reference is not None:
            trace_keys += ("raw_requested_commands",)
        traces = {k: [] for k in trace_keys}
        with torch.inference_mode():
            for tick in range(env.max_episode_length + 1):
                policy_input = observations["policy"]
                expected_context = policy_input.new_tensor(plan["flat_context"]).expand(count, -1)
                if not torch.equal(policy_input[:, 28:35], expected_context):
                    raise RuntimeError("Original flat context changed during replay")
                actions = actor(policy_input)
                raw_requested = command_reference.requested.clone() if command_reference is not None else None
                observations, _, done, extras = env.step(actions)
                diagnostic = extras["diagnostics"]
                metrics.observe(diagnostic, alive)
                requests.observe(diagnostic, alive)
                if audit is not None:
                    audit.observe(extras["native_reward_components"], diagnostic, alive)
                for key in trace_keys:
                    value = {"active": alive, "observations": policy_input, "actions": actions,
                             "raw_requested_commands": raw_requested}.get(key)
                    if value is None:
                        value = diagnostic[key]
                    traces[key].append(value.cpu().numpy().copy())
                alive &= ~done.bool()
                if tick % 100 == 0:
                    print("FLAT_REPLAY", json.dumps(dict(tick=tick, active=int(alive.sum()),
                        elapsed_s=round(time.monotonic() - start, 2))), flush=True)
                if not bool(alive.any()):
                    break
        measured = metrics.report()
        requests.report_into(measured)
        result = grade_fixed_suite(measured, settings, 4)
        if args.diagnostic_ablation:
            result.update(passed=False, diagnostic_only=True, qualification_eligible=False)
        result.update(unfinished_episodes=int(alive.sum()), policy_ticks=tick + 1, checkpoint_updates=step)
        np.savez_compressed(args.output / "traces.npz", **{k: np.asarray(v) for k, v in traces.items()},
            groups=np.asarray(env.scene_groups))
        # Same TensorBoard event format as the project's existing RSL logger;
        # this writes events only and starts no local TensorBoard service.
        event_dir = args.checkpoint.parent if args.checkpoint else args.output
        event_namespace = (f"FlatEvaluation/{settings['protocol_id']}/"
                           f"physics_{report['physics_hz']}Hz/seed_{seed}")
        if args.diagnostic_ablation:
            event_namespace += "/" + args.diagnostic_ablation
        if audit is not None:
            event_namespace += "/reward_audit"
            report["reward_audit"] = audit.report()
        report["tensorboard_namespace"] = event_namespace
        with SummaryWriter(str(event_dir)) as writer:
            writer.add_scalar(f"{event_namespace}/passed_cases", sum(c["passed"] for c in result["cases"].values()), step)
            for name, case in result["cases"].items():
                for key in ("survival_rate", "height_mae_m", "vx_mae_m_s", "yaw_mae_rad_s",
                            "stand_drift_max_m", "planar_speed_mean_m_s", "closure_gap_max_m"):
                    writer.add_scalar(f"{event_namespace}/{name}/{key}", case[key], step)
            if audit is not None:
                for window, cases in report["reward_audit"].items():
                    for name, values in cases.items():
                        prefix = f"{event_namespace}/{name}/RewardAudit/{window}"
                        for term, value in values["component_reward_per_sim_second"].items():
                            writer.add_scalar(f"{prefix}/{term}/reward_per_sim_second", value, step)
                        for phase, value in values["physical_phase_fraction"].items():
                            writer.add_scalar(f"{prefix}/phase_{phase}_fraction", value, step)
                        writer.add_scalar(f"{prefix}/residual_reward_per_policy_step",
                            values["residual_reward_per_policy_step"], step)
        report.update(status="evaluated", result=result, elapsed_s=time.monotonic() - start)
        print("FLAT_RESULT", json.dumps({n: {k: c[k] for k in ("passed", "survival_rate", "height_mae_m",
            "vx_mae_m_s", "yaw_mae_rad_s", "stand_drift_max_m")} for n, c in result["cases"].items()}), flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "evaluation.json").write_bytes(encode(report))
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
