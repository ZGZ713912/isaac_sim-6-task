#!/usr/bin/env python3
"""Read-only evaluation of a scratch checkpoint; never constructs Adam/PPO."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=261101)
    parser.add_argument("--duration-seconds", type=int, choices=(8, 30), default=8,
                        help="Read-only horizon;30 explicitly extends the archived8s loop and episode timeout")
    parser.add_argument("--external-force-mode", choices=("native", "every-iteration"), default="native",
                        help="Declared zero-PPO PhysX counterfactual; native keeps the checkpoint's original physics")
    parser.add_argument("--wheel-response-scale", type=float, default=1.,
                        help="Zero-PPO wheel-encoder feedback ablation;1 keeps the native policy function")
    parser.add_argument("--reset-observation-mode", choices=("native", "refresh-final-pose"), default="native",
                        help="Zero-PPO frozen-v1 reset-cache counterfactual; native keeps the archived reset")
    parser.add_argument("--velocity-iterations", type=int, choices=(1, 4), default=None,
                        help="Zero-PPO TGS counterfactual; only reduce articulation velocity iterations from32")
    parser.add_argument("--hold-initial-leg-targets", action="store_true",
                        help="Zero-PPO trained-height diagnostic: hold each lane's first four targets; wheels keep the native actor")
    parser.add_argument("--leg-hold-after-seconds", type=int, choices=(0, 3), default=0,
                        help="With the leg-hold diagnostic, retain the native prefix before capturing targets")
    parser.add_argument("--trace-feedback", action="store_true",
                        help="Record exact actor inputs, actions, gyro and wheel response without changing the fixed evaluation")
    parser.add_argument("--trace-contact-state", action="store_true",
                        help="With trace-feedback, read native full joint state and contact buffers at every existing1ms boundary")
    args = parser.parse_args()
    if args.trace_contact_state and not args.trace_feedback:
        parser.error("The native contact-state observer requires trace-feedback")
    if args.hold_initial_leg_targets and not args.trace_feedback:
        parser.error("The leg-target hold diagnostic requires trace-feedback")
    if args.leg_hold_after_seconds and not args.hold_initial_leg_targets:
        parser.error("A delayed leg-hold boundary requires the leg-target hold diagnostic")
    if args.duration_seconds != 8 and not args.trace_feedback:
        parser.error("The extended30s protocol requires trace-feedback")
    if not math.isfinite(args.wheel_response_scale) or not 0. <= args.wheel_response_scale <= 1.:
        parser.error("Wheel response scale must be finite within0..1")
    if sum((args.wheel_response_scale != 1., args.external_force_mode != "native",
            args.reset_observation_mode != "native", args.velocity_iterations is not None,
            args.hold_initial_leg_targets, args.duration_seconds != 8)) > 1:
        parser.error("Use one declared counterfactual at a time")
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.checkpoint.parent
    import torch
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    infos = state.get("infos", {})
    if (infos.get("schema") != "scratch-standing-checkpoint-v1" or infos.get("training_origin") != "scratch"
            or infos.get("initialization", {}).get("checkpoint_loaded") is not False
            or not infos.get("continuation_forbidden")):
        raise ValueError("Only authenticated new scratch policies may be evaluated here")
    archive = run / "source_snapshot"
    for name, expected in infos["source_hashes"].items():
        if digest(archive / name) != expected:
            raise ValueError("Archived scratch source changed: " + name)
    agent_path = run / "agent_config.json"
    agent = json.loads(agent_path.read_text())
    actor_name = agent["actor"]["class_name"]
    if actor_name not in ("MLPModel", "wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel"):
        raise ValueError("Unsupported scratch actor action ABI")
    agent_hash = infos.get("agent_config_sha256")
    if (agent_hash is not None and digest(agent_path) != agent_hash
            or actor_name != infos.get("actor_class", "MLPModel")
            or actor_name != "MLPModel" and agent_hash is None):
        raise ValueError("Scratch actor class/configuration is not authenticated")
    # Load the exact task/evaluation implementation used by that run. Current
    # source changes must not silently change an old checkpoint's evaluation.
    sys.path[:0] = [str(archive / "src"), str(archive / "scripts")]
    spec = importlib.util.spec_from_file_location("frozen_scratch_training", archive / "scripts/train_scratch_standing.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    training.torch = torch
    cfg = json.loads((run / "config/contract.json").read_text())
    if digest(run / "config/contract.json") != infos["mechanism_config_sha256"]:
        raise ValueError("Mechanism contract changed")
    original_design = Path(cfg["design_preflight"]["file"])
    if digest(original_design) != cfg["design_preflight"]["sha256"]:
        raise ValueError("Independent design changed")
    design = json.loads(original_design.read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    original_seed = row["runtime"]["seed"]
    row["runtime"]["seed"] = args.seed
    row["invocation"] = dict(num_envs=cfg["target_num_envs"], seed=dict(rule="fixed", value=args.seed))
    design["scope"] = "readonly_scratch_standing_evaluation"
    design["review_basis"] = dict(source_mechanism_contract=str(run / "config/contract.json"),
        source_sha256=infos["mechanism_config_sha256"], checkpoint_sha256=digest(args.checkpoint),
        only_runtime_override=dict(seed=[original_seed, args.seed]))
    if args.external_force_mode == "every-iteration":
        design["review_basis"]["only_runtime_override"]["enable_external_forces_every_iteration"] = [False, True]
    if args.wheel_response_scale != 1.:
        design["review_basis"]["only_runtime_override"]["wheel_encoder_response_scale"] = [1., args.wheel_response_scale]
    if args.reset_observation_mode != "native":
        design["review_basis"]["only_runtime_override"]["reset_observation_cache"] = ["native", "refresh-final-pose"]
    if args.hold_initial_leg_targets:
        design["review_basis"]["only_runtime_override"]["hold_initial_leg_targets"] = dict(
            height_m=.305, legs="each lane's own native targets at the declared boundary",
            hold_after_s=args.leg_hold_after_seconds, wheels="native current actor output")
    solver_counterfactual = None
    if args.velocity_iterations is not None:
        from solver_velocity_iteration_probe import configure as configure_velocity_probe
        solver_counterfactual = configure_velocity_probe(cfg, design, args.velocity_iterations)
        design["review_basis"]["only_runtime_override"]["velocity_iterations"] = [32, args.velocity_iterations]
    task = deepcopy(infos["settings"])
    duration_extension = None
    if args.duration_seconds != 8:
        from extended_scratch_standing_evaluation import configure as configure_duration
        duration_extension = configure_duration(cfg, design, task, args.duration_seconds)
        design["review_basis"]["only_runtime_override"]["evaluation_duration_s"] = [8, args.duration_seconds]
    training.write_json(args.output / "config/design.json", design)
    cfg["design_preflight"] = dict(file=str(args.output / "config/design.json"),
        sha256=digest(args.output / "config/design.json"), stage="speed")
    training.write_json(args.output / "config/contract.json", cfg)
    bank = run / "pose_bank.json"
    if not bank.exists():
        bank = Path(json.loads((run / "task_contract.json").read_text())["bank"])
    if digest(bank) != infos["bank_sha256"]:
        raise ValueError("Reset geometry changed")
    progress = dict(schema="scratch-standing-readonly-evaluation-v1", status="starting",
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        completed_updates=infos["completed_updates"], new_PPO=0, optimizer_constructed=False,
        evaluator_sha256=digest(Path(__file__)), seed=args.seed,
        frozen_source_hashes_verified=len(infos["source_hashes"]), hardware_ready=False)
    progress["external_force_mode"] = args.external_force_mode
    progress["wheel_response_scale"] = args.wheel_response_scale
    progress["reset_observation_mode"] = args.reset_observation_mode
    progress["hold_initial_leg_targets"] = args.hold_initial_leg_targets
    progress["leg_hold_after_seconds"] = args.leg_hold_after_seconds
    progress["evaluation_duration_s"] = args.duration_seconds
    if duration_extension is not None:
        progress["duration_extension"] = duration_extension
    if solver_counterfactual is not None:
        progress["solver_counterfactual"] = solver_counterfactual
    diagnostic_sources = ("evaluate_scratch_standing.py", "record_scratch_physics.py",
                          "capture_scratch_feedback.py", "analyze_scratch_feedback_trace.py",
                          "wheel_encoder_response_probe.py", "final_pose_observation_probe.py")
    if args.trace_contact_state:
        diagnostic_sources += ("capture_scratch_contact_state.py",)
    if args.velocity_iterations is not None:
        diagnostic_sources += ("solver_velocity_iteration_probe.py",)
    if args.hold_initial_leg_targets:
        diagnostic_sources += ("hold_initial_leg_target_probe.py",)
    if duration_extension is not None:
        diagnostic_sources += ("extended_scratch_standing_evaluation.py",)
    frozen_diagnostics = args.output / "diagnostic_sources"
    frozen_diagnostics.mkdir()
    progress["diagnostic_source_hashes"] = {}
    for name in diagnostic_sources:
        data = (ROOT / "scripts" / name).read_bytes()
        (frozen_diagnostics / name).write_bytes(data)
        progress["diagnostic_source_hashes"][name] = hashlib.sha256(data).hexdigest()
    if args.reset_observation_mode != "native":
        from final_pose_observation_probe import prepare as prepare_reset_probe
        progress["reset_counterfactual"] = prepare_reset_probe(archive, frozen_diagnostics)
    if duration_extension is not None:
        import inspect
        from extended_scratch_standing_evaluation import extend_function
        training.evaluate, annotation = extend_function(inspect.getsource(training.evaluate),
            args.duration_seconds, vars(training), frozen_diagnostics / "extended_evaluate.py")
        duration_extension.update(annotation)
    training.write_json(args.output / "receipt.json", progress)
    launcher = env = writer = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        from rsl_rl.utils import resolve_class
        from rsl_rl.utils.logger import Logger
        from wheeled_tasks.chassis.scratch_standing_env import ScratchStandingEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        # Restrict the intervention to this process's SimulationCfg construction.
        # The archived task, actor, actions, force reports and acceptance stay native.
        import isaaclab.sim as sim_utils
        original_simulation_cfg = sim_utils.SimulationCfg

        def simulation_cfg_with_external_forces(*positional, **keywords):
            from isaaclab_physx.physics import PhysxCfg
            value = original_simulation_cfg(*positional, **keywords)
            if value.physics is None:
                value.physics = PhysxCfg()
            if not isinstance(value.physics, PhysxCfg) or value.physics.solver_type != 1:
                raise ValueError("The external-force counterfactual requires the native TGS backend")
            value.physics.enable_external_forces_every_iteration = True
            return value

        if args.external_force_mode == "every-iteration":
            sim_utils.SimulationCfg = simulation_cfg_with_external_forces
        if args.reset_observation_mode != "native":
            from final_pose_observation_probe import activate as activate_reset_probe
            activate_reset_probe(progress["reset_counterfactual"]["modified_source"])
        try:
            env = ScratchStandingEnv(cfg, manifest, load_chassis_control(ROOT / cfg["control_math_source"]), ROOT,
                stage_name="speed", num_envs=cfg["target_num_envs"], device=args.device, seed=args.seed,
                level=1., run_role="train", bank=bank, settings=task)
        finally:
            sim_utils.SimulationCfg = original_simulation_cfg
        from record_scratch_physics import record as record_physics
        progress["physics_runtime"] = record_physics(env)
        if duration_extension is not None:
            from extended_scratch_standing_evaluation import assert_runtime as verify_duration_runtime
            duration_extension["native_runtime"] = verify_duration_runtime(env, duration_extension)
        if solver_counterfactual is not None:
            from solver_velocity_iteration_probe import assert_observed as verify_velocity_probe
            verify_velocity_probe(progress["physics_runtime"], solver_counterfactual)
            progress["solver_counterfactual_runtime_verified"] = True
        actual_flag = progress["physics_runtime"]["scene_flags"]["physxScene:enableExternalForcesEveryIteration"]
        if actual_flag is not (args.external_force_mode == "every-iteration"):
            raise ValueError("The requested external-force mode did not match the observed native scene")
        training.write_json(args.output / "receipt.json", progress)
        install(env, speed_weight=0., contact_weight=0., torque_weight=0.)
        actor_class, actor_cfg = resolve_class(agent["actor"])
        if actor_name != "MLPModel":
            import inspect
            if Path(inspect.getsourcefile(actor_class)).resolve() != archive / "src/wheeled_algo/wheel_velocity_increment.py":
                raise ValueError("Wheel increment actor was not imported from its frozen source")
        actor = actor_class(env.get_observations(), agent["obs_groups"], "actor", 6, **actor_cfg).to(args.device)
        actor.load_state_dict(state["actor_state_dict"], strict=True)
        actor.eval()
        if args.wheel_response_scale != 1.:
            from wheel_encoder_response_probe import WheelEncoderResponseProbe
            actor = WheelEncoderResponseProbe(actor, args.wheel_response_scale)
            progress["policy_counterfactual"] = dict(
                wheel_encoder_response_scale=args.wheel_response_scale,
                formula="wheel=F(neutral_encoder)+scale*(F(current_encoder)-F(neutral_encoder)); four legs=F(current)",
                neutral_features={"20": 0., "21": 0.},
                learned_policy=False, adoption_by_training=False, export_forbidden=True,
                scope="Read-only diagnostic intervention on the frozen neural function; no standing qualification")
            training.write_json(args.output / "receipt.json", progress)
        if args.hold_initial_leg_targets:
            from hold_initial_leg_target_probe import HoldInitialLegTargetProbe
            actor = HoldInitialLegTargetProbe(actor, hold_after_seconds=args.leg_hold_after_seconds).eval()
        logger = Logger(log_dir=str(args.output), cfg=agent, env_cfg=env.cfg,
            num_envs=env.num_envs, is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device=args.device)
        logger.init_logging_writer()
        writer = logger.writer
        writer.add_scalar("EvalScratchProtocol/duration_s", args.duration_seconds, infos["completed_updates"])
        writer.add_scalar("EvalScratchProtocol/startup_grace_s", 0., infos["completed_updates"])
        writer.add_scalar("EvalScratchProtocol/contact_threshold_n", 2., infos["completed_updates"])
        evaluation = training.evaluate
        if args.trace_contact_state:
            from capture_scratch_contact_state import evaluate_with_contact_state

            def evaluation(*positional, **keywords):
                return evaluate_with_contact_state(*positional, **keywords,
                    evaluate=training.evaluate, digest=digest)

        if args.trace_feedback:
            from capture_scratch_feedback import evaluate_with_feedback
            progress["evaluation"] = evaluate_with_feedback(env, actor, args.output / "evaluation", writer,
                infos["completed_updates"], evaluate=evaluation, digest=digest)
        else:
            progress["evaluation"] = evaluation(env, actor, args.output / "evaluation", writer, infos["completed_updates"])
        progress["evaluation"].update(checkpoint_loaded=True, checkpoint_loading_scope="read-only actor evaluation",
            new_PPO=0, optimizer_constructed=False)
        if duration_extension is not None:
            from extended_scratch_standing_evaluation import assert_result as verify_duration_result
            verify_duration_result(progress["evaluation"], duration_extension)
            progress["evaluation"].update(evaluation_duration_s=args.duration_seconds,
                evaluation_protocol_extension=duration_extension, behavior_qualification=False)
        if args.hold_initial_leg_targets:
            progress["policy_counterfactual"] = actor.finish(args.output / "evaluation", writer,
                infos["completed_updates"], digest)
            progress["evaluation"].update(learned_policy=False,
                diagnostic_policy_intervention=progress["policy_counterfactual"], behavior_qualification=False)
        if args.wheel_response_scale != 1.:
            progress["evaluation"].update(learned_policy=False,
                diagnostic_policy_intervention=progress["policy_counterfactual"], behavior_qualification=False)
        if args.reset_observation_mode != "native":
            progress["evaluation"].update(diagnostic_environment_intervention=progress["reset_counterfactual"],
                behavior_qualification=False)
        if solver_counterfactual is not None:
            progress["evaluation"].update(diagnostic_environment_intervention=solver_counterfactual,
                behavior_qualification=False)
        training.write_json(args.output / "evaluation/result.json", progress["evaluation"])
        progress["status"] = "completed_unqualified"
    except BaseException:
        progress.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        training.write_json(args.output / "receipt.json", progress)
        if writer is not None:
            writer.flush()
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
