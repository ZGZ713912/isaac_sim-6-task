#!/usr/bin/env python3
"""New actor, new critic and empty Adam. Checkpoint continuation is forbidden."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE_CONFIG = ROOT / "runs/flat_base_20261020/pilot034_directional200_1346/config/contract.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args):
    from wheeled_tasks.chassis.robust_getup import settings, rigid_body_domain
    from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins

    # Reuse the authenticated mechanism description only, never a model file.
    cfg = json.loads(BASE_CONFIG.read_text())
    previous_design = Path(cfg["design_preflight"]["file"])
    if digest(previous_design) != cfg["design_preflight"]["sha256"]:
        raise ValueError("Mechanism design changed")
    design = json.loads(previous_design.read_text())
    task = settings(domains_enabled=False, domain_scale=0., motion_fraction=1.,
        motion_push_fraction=0., quiet_training=True, quiet_critic_anchor=True,
        quiet_action_rate_cost=0., quiet_motor_speed_cost=0., quiet_effort_delta_cost=0.,
        quiet_spectral_cost=False, standing_support_training=False,
        episode_seconds=8., curriculum_updates=max(1, args.all_heights_after),
        origin_updates=0, native_origin_mode="isolated_overlap_v1")
    task["version"] = "scratch-standing-v1"
    if args.precise_objective:
        task["version"] = "scratch-standing-v3-coupled"
    if args.standing_failures:
        task["version"] += "-failure-resets"
    if args.wheel_velocity_increment:
        task["version"] += "-wheel-increment-v1"
    if args.static_load_initialization:
        task["version"] += "-static-load-init-v1"
    changes = dict(target_num_envs=args.num_envs, num_steps_per_env=args.rollout_steps,
        ppo_minibatch_samples=args.num_envs * args.rollout_steps // 4,
        episode_seconds=8., record_diagnostics=False, full_rl_square_ground=True,
        full_rl_isolated_origins=True, flat_floor_width_m=96., flat_half_length_m=48.,
        dynamics_randomization=rigid_body_domain(False, 0.))
    # These fields select the existing 35D flat context and privileged XY
    # channels. Replace the old transfer provenance and unrelated short groups.
    changes.update(flat_base_continuation=dict(schema="scratch-standing-context-v1",
        flat_context=[1, 0, 0, 0, 0, 0, 0], initialization="none; fresh native PPO"),
        flat_group_episode_seconds={}, curriculum="scratch_standing_foundation",
        critic_warmup_updates=0, resume_critic_warmup_updates=0, transfer_critic=False)
    material = deepcopy(cfg["contact_material"])
    for key in ("ground_static_friction_range", "ground_dynamic_friction_range"):
        material[key] = [.5, .5]
    changes["contact_material"] = material
    static_prior = None
    if args.static_load_initialization:
        from wheeled_algo.standing_load_initialization import load_prior
        reference = ROOT / "reports/robust_policy_repair_20261024/scratch_nominal_static_load001.json"
        native_pair = ROOT / "reports/robust_policy_repair_20261024/scratch_static_load_native_pair001.json"
        current = dict(cfg, **changes)
        static_prior = load_prior(reference, native_pair, mechanism=current, bank_path=args.bank, root=ROOT)
        (args.output / "static_load_reference.json").write_bytes(reference.read_bytes())
        (args.output / "static_load_native_pair.json").write_bytes(native_pair.read_bytes())
    row = design["stages"]["speed"]["roles"]["train"]
    for key, value in changes.items():
        cfg[key] = deepcopy(value)
        if key in row["contract"]:
            row["contract"][key] = deepcopy(value)
    validate_isolated_origins(cfg, args.num_envs)
    row["runtime"]["seed"] = args.seed
    row["runtime"]["contact_material"] = deepcopy(material)
    row["runtime"]["modules"]["dynamics_randomization"] = False
    row["invocation"] = dict(num_envs=args.num_envs, seed=dict(rule="fixed", value=args.seed))
    design["scope"] = "scratch_standing_mechanism_construction"
    design["review_basis"] = dict(mechanism_config=str(BASE_CONFIG), sha256=digest(BASE_CONFIG),
        checkpoint_initialization="forbidden", task_contract=str(args.output / "task_contract.json"))
    write_json(args.output / "config/design.json", design)
    cfg["design_preflight"] = dict(file=str(args.output / "config/design.json"),
        sha256=digest(args.output / "config/design.json"), stage="speed")
    write_json(args.output / "config/contract.json", cfg)
    write_json(args.output / "task_contract.json", dict(schema="scratch-standing-task-v1", settings=task,
        training_origin="seeded random actor and critic; empty Adam; no checkpoint reads",
        bank=str(args.bank), bank_sha256=digest(args.bank),
        mechanism_config_sha256=digest(args.output / "config/contract.json"),
        policy_hz=200, physics_hz=1000, action_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        leg_position_scale_rad=.25, wheel_velocity_scale_rad_s=10.,
        state_writes="reset only", initial_pitch_range_deg=[-.5, .5],
        curriculum=dict(mid_height_until_update=args.all_heights_after, then_heights_m=[.23, .305, .42]),
        current_scope="nominal standing foundation; robustness and recovery are not qualified",
        precise_standing_objective=args.precise_objective, normalized_observations=args.normalize_observations,
        desynchronized_initial_episode_lengths=args.desync_episodes,
        standing_failure_resets=args.standing_failures,
        wheel_action_parameterization=("encoder_velocity_plus_learned_increment_v1"
            if args.wheel_velocity_increment else "learned_absolute_velocity_v1"),
        static_load_initialization=static_prior, hardware_ready=False))
    return cfg, task


def sources():
    files = list((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def nominal_action(env, height=.305):
    from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
    pose = env.bank_joints[env.height_pose[height], env.ids]
    delta = pose - env.v5.nominal
    result = delta.clone().zero_()
    result[list(env.v5.LEGS)] = torch.atan2(delta[list(env.v5.LEGS)].sin(), delta[list(env.v5.LEGS)].cos()) / .25
    return result[list(POLICY_FROM_CONTROL)]


def evaluate(env, actor, output, writer, completed, *, fixed_target=False, heights=(.23, .305, .42)):
    import numpy as np
    count = 7 * len(heights)
    pitches = (0., -.25, .25, -.5, .5, -1., 1.)
    requested = [dict(height=height, pitch_deg=pitch) for height in heights for pitch in pitches]
    if env.num_envs < count:
        raise ValueError("Evaluation requires at least 21 native environments")
    cases = [dict(pose_index=env.height_pose[requested[i % count]["height"]], recovery=False,
                  command=[0., 0., requested[i % count]["height"]], pitch_deg=requested[i % count]["pitch_deg"])
             for i in range(env.num_envs)]
    env.evaluation_cases = cases
    env.cfg["record_diagnostics"] = True
    env.reset(torch.arange(env.num_envs, device=env.device))
    env.fixed_cases = cases
    initial_root = env.robot.data.root_link_pose_w.torch[:count, :3].clone()
    horizon = round(8. / env.policy_dt)
    traces = {key: [] for key in ("force", "clearance", "dq", "torque", "q", "valid", "position", "gravity", "velocity", "gap")}
    observation = env.get_observations()
    with torch.inference_mode():
        constant = torch.stack([nominal_action(env, case["command"][2]) for case in cases])
        for _ in range(horizon):
            valid = ~env.finished[:count].clone()
            action = constant if fixed_target else actor(observation)
            observation, _, _, info = env.step(action)
            contacts = env.standing_contact_monitor.last
            physical = env.physical_standing_last
            diagnostic = info["diagnostics"]
            values = dict(force=contacts["force"][:, :count], clearance=contacts["clearance"][:, :count],
                dq=physical["dq"][:, :count], torque=physical["torque"][:, :count],
                q=diagnostic["motor_position"][:count, list(env.v5.LEGS)], valid=valid,
                position=diagnostic["local_position"][:count], gravity=diagnostic["gravity"][:count],
                velocity=diagnostic["velocity"][:count], gap=diagnostic["gap"][:count])
            for key, value in values.items():
                traces[key].append(value.cpu())
    arrays = {key: torch.stack(values).numpy() for key, values in traces.items()}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "native_trace.npz", **arrays)
    rows = []
    for i, case in enumerate(requested):
        valid = arrays["valid"][:, i]
        force = arrays["force"][:, :, i][valid]
        contact = (np.linalg.norm(force, axis=-1) > 2.).all(-1)
        speed = arrays["dq"][:, :, i][valid]
        q = arrays["q"][:, i][valid]
        late = q[600:] if len(q) > 600 else q
        frequency = np.fft.rfftfreq(len(late), env.policy_dt)
        spectrum = np.fft.rfft(late - late.mean(0), axis=0)
        spectrum[(frequency < 3.) | (frequency > 95.)] = 0.
        highpass = np.fft.irfft(spectrum, n=len(late), axis=0)
        position = arrays["position"][:, i][valid]
        tilt = np.arccos(np.clip(-arrays["gravity"][:, i, 2][valid], -1., 1.))
        velocity = arrays["velocity"][:, i][valid]
        gap = arrays["gap"][:, i][valid]
        row = dict(height_m=case["height"], pitch_deg=case["pitch_deg"],
            full_horizon=len(q) == horizon, physical_ms=int(contact.size),
            missing_physical_ms=int((~contact).sum()), dual_support_fraction=float(contact.mean()),
            first20ms_continuous=bool(contact.reshape(-1)[:20].all()),
            maximum_wheel_clearance_m=float(arrays["clearance"][:, :, i][valid].max()),
            leg_speed_peak_rad_s=float(np.abs(speed).max()), leg_speed_rms_rad_s=float(np.sqrt(np.mean(speed**2))),
            leg_vibration_3_95hz_rms_mrad=float(np.sqrt(np.mean(highpass**2))*1000.),
            height_mae_m=float(np.abs(position[:, 2] - case["height"]).mean()),
            tilt_peak_deg=float(np.degrees(tilt).max()), closure_gap_peak_m=float(gap.max()),
            forward_speed_rms_m_s=float(np.sqrt(np.mean(velocity[:, 0]**2))))
        row["continuous_support_pass"] = bool(row["full_horizon"] and row["missing_physical_ms"] == 0)
        row["standing_pass"] = bool(row["continuous_support_pass"] and row["tilt_peak_deg"] < 10.
            and row["height_mae_m"] < .01 and row["forward_speed_rms_m_s"] < .03
            and row["leg_speed_peak_rad_s"] < 3. and row["closure_gap_peak_m"] < .003)
        rows.append(row)
    result = dict(schema="scratch-standing-native-evaluation-v1", completed_updates=completed,
        protocol="nominal dynamics; seven held-out pitch starts per height; no startup grace; every 1 ms",
        fixed_target_probe=fixed_target, learned_policy=not fixed_target,
        checkpoint_loaded=False, initial_root_positions=initial_root.cpu().tolist(),
        case_count=count, continuous_support_pass=sum(row["continuous_support_pass"] for row in rows),
        standing_pass=sum(row["standing_pass"] for row in rows), cases=rows,
        trace_sha256=digest(output / "native_trace.npz"), hardware_ready=False)
    write_json(output / "result.json", result)
    for height in heights:
        selected = [row for row in rows if row["height_m"] == height]
        for metric in ("dual_support_fraction", "missing_physical_ms", "leg_speed_peak_rad_s",
                       "leg_vibration_3_95hz_rms_mrad", "height_mae_m", "forward_speed_rms_m_s"):
            writer.add_scalar(f"EvalScratchStanding/{height:.3f}m/{metric}",
                              sum(row[metric] for row in selected)/len(selected), completed)
    writer.add_scalar("EvalScratchStanding/continuous_support_pass", result["continuous_support_pass"], completed)
    writer.flush()
    return result


class TrainingStopped(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json")
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=261001)
    parser.add_argument("--all-heights-after", type=int, default=512)
    parser.add_argument("--speed-weight", type=float, default=2.)
    parser.add_argument("--contact-weight", type=float, default=8.)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--precise-objective", action="store_true")
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--desync-episodes", action="store_true")
    parser.add_argument("--standing-failures", action="store_true")
    parser.add_argument("--wheel-velocity-increment", action="store_true",
                        help="Fresh actor uses raw wheel encoder velocity plus a learned correction; final action ABI stays absolute")
    parser.add_argument("--static-load-initialization", action="store_true",
                        help="Initialize fresh trainable leg bias from the authenticated nominal static load, never from a policy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-target-probe", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.updates < 0 or args.num_envs < 21 or args.num_envs > 256 or args.rollout_steps < 8:
        parser.error("Need nonnegative updates,21..256 isolated environments and rollout>=8")
    if args.fixed_target_probe and args.updates:
        parser.error("A fixed-target physics probe must have zero PPO updates")
    if args.standing_failures and not args.precise_objective:
        parser.error("Standing failure resets require the precise standing objective")
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, task = prepare(args)
    source_hashes = sources()
    for name, expected in source_hashes.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Source changed while taking the scratch snapshot")
        target = args.output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (args.output / "pose_bank.json").write_bytes(args.bank.read_bytes())
    write_json(args.output / "source_hashes.json", source_hashes)
    if args.preflight_only:
        print(json.dumps(dict(status="prepared", checkpoint_reads=0, output=str(args.output))))
        return
    global torch
    launcher = env = runner = audit = None
    start = time.monotonic()
    report = dict(schema="scratch-standing-progress-v1", status="starting",
        started_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        completed_updates=0, optimizer_attempts=0, checkpoint_reads=[],
        requested_updates=args.updates, hardware_ready=False, args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write_json(args.output / "progress.json", report)
    original_load = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        report["requested_stop_signal"] = signum

    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.chassis.scratch_standing_env import ScratchStandingEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install
        from wheeled_algo.scratch_standing import agent_config, initialize_fresh, tensor_hash
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.native_ppo_audit import NativePPOAudit

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        original_load = torch.load

        def forbidden_load(*positional, **keywords):
            report["checkpoint_reads"].append(str(positional[0] if positional else keywords.get("f")))
            raise RuntimeError("Scratch training forbids every torch.load call")

        torch.load = forbidden_load
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        env = ScratchStandingEnv(cfg, manifest, load_chassis_control(ROOT / cfg["control_math_source"]), ROOT,
            stage_name="speed", num_envs=args.num_envs, device=args.device, seed=args.seed, level=1.,
            run_role="train", bank=args.bank, settings=task, all_heights_after=args.all_heights_after)
        report["physical_reward"] = install(env, speed_weight=args.speed_weight, contact_weight=args.contact_weight)
        if args.precise_objective:
            from wheeled_tasks.chassis.precise_standing_reward import install as install_precision
            report["precise_reward"] = install_precision(env)
        if args.standing_failures:
            from wheeled_tasks.chassis.standing_failure_termination import install as install_failures
            report["standing_failure_resets"] = install_failures(env)
        runner_cfg = agent_config(args.seed, args.device, args.rollout_steps, args.learning_rate,
            normalize_observations=args.normalize_observations,
            wheel_velocity_increment=args.wheel_velocity_increment)
        write_json(args.output / "agent_config.json", runner_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(args.output), device=args.device)
        initial = nominal_action(env)
        bias_source = None
        if args.static_load_initialization:
            from wheeled_algo.standing_load_initialization import initial_action
            prior = json.loads((args.output / "task_contract.json").read_text())["static_load_initialization"]
            initial = initial_action(initial, prior)
            bias_source = prior["source"]
        report["initialization"] = initialize_fresh(runner.alg, initial, output_bias_source=bias_source)
        if args.static_load_initialization:
            report["initialization"]["static_load_initialization"] = prior
        report["initialization"]["actor_class"] = runner_cfg["actor"]["class_name"]
        report["initialization"]["wheel_velocity_increment"] = args.wheel_velocity_increment
        write_json(args.output / "initialization.json", report["initialization"])
        initial_actor = {key: value.detach().cpu().clone() for key, value in runner.alg.actor.state_dict().items()}
        initial_critic = {key: value.detach().cpu().clone() for key, value in runner.alg.critic.state_dict().items()}
        mask = RecoveryRolloutMask(runner.alg, num_envs=args.num_envs)
        native_update = runner.alg.update

        def save(path, infos=None):
            path = Path(path)
            state = runner.alg.save()
            state.update(iter=report["completed_updates"], infos=dict(schema="scratch-standing-checkpoint-v1",
                initialization=report["initialization"], completed_updates=report["completed_updates"],
                training_origin="scratch", continuation_forbidden=True,
                settings=task, source_hashes=source_hashes, bank_sha256=digest(args.bank),
                actor_class=runner_cfg["actor"]["class_name"],
                agent_config_sha256=digest(args.output / "agent_config.json"),
                mechanism_config_sha256=digest(args.output / "config/contract.json"),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                env_rng=env.generator.get_state(), domain_rng=env.domain.rng.get_state()))
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            report["last_checkpoint"] = dict(path=str(path), sha256=digest(path), updates=report["completed_updates"])

        def update():
            nonlocal audit
            report["optimizer_attempts"] += 1
            report["optimizer_update_in_flight"] = True
            write_json(args.output / "progress.json", report)
            losses = native_update()
            report["completed_updates"] += 1
            report["optimizer_update_in_flight"] = False
            for model in (runner.alg.actor, runner.alg.critic):
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise RuntimeError("Nonfinite fresh PPO parameter")
            if report["completed_updates"] == 1:
                audit.close()
                audit_result = audit.report()
                audit_result.update(actor_changed_tensors=sum(not torch.equal(initial_actor[key], value.cpu()) for key, value in runner.alg.actor.state_dict().items()),
                    critic_changed_tensors=sum(not torch.equal(initial_critic[key], value.cpu()) for key, value in runner.alg.critic.state_dict().items()),
                    actor_changed_parameter_tensors=sum(not torch.equal(initial_actor[key], value.detach().cpu()) for key, value in runner.alg.actor.named_parameters()),
                    critic_changed_parameter_tensors=sum(not torch.equal(initial_critic[key], value.detach().cpu()) for key, value in runner.alg.critic.named_parameters()),
                    actor_sha256=tensor_hash(runner.alg.actor.state_dict()), critic_sha256=tensor_hash(runner.alg.critic.state_dict()))
                if (not audit_result["all_recorded_values_finite"] or audit_result["committed_optimizer_calls"] != 12
                        or not audit_result["actor_changed_parameter_tensors"] or not audit_result["critic_changed_parameter_tensors"]):
                    raise RuntimeError("First fresh native PPO update failed its audit")
                write_json(args.output / "first_update_audit.json", audit_result)
                audit = None
            env.actor_updates = report["completed_updates"]
            report.update(status="training", elapsed_s=time.monotonic()-start,
                last_losses={key: float(value) for key, value in losses.items()}, valid_rollout=mask.metrics,
                last_physical_metrics={key: float(value) for key, value in env.physical_standing_last["metrics"].items()})
            write_json(args.output / "progress.json", report)
            if stop_requested:
                save(args.output / "model_stopped.pt")
                raise TrainingStopped()
            return losses

        runner.save, runner.alg.update = save, update
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        save(args.output / "model_initial.pt")
        if args.updates:
            audit = NativePPOAudit(runner.alg)
            runner.learn(args.updates, init_at_random_ep_len=args.desync_episodes)
        else:
            runner.logger.init_logging_writer()
        save(args.output / "model_final.pt")
        report["status"] = "evaluating"
        write_json(args.output / "progress.json", report)
        runner.alg.eval_mode()
        report["evaluation"] = evaluate(env, runner.alg.actor, args.output / "evaluation", runner.logger.writer,
                                         report["completed_updates"], fixed_target=args.fixed_target_probe)
        report.update(status="completed_unqualified", elapsed_s=time.monotonic()-start)
    except TrainingStopped:
        report.update(status="stopped", elapsed_s=time.monotonic()-start)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_s=time.monotonic()-start)
        raise
    finally:
        write_json(args.output / "progress.json", report)
        if original_load is not None:
            torch.load = original_load
        if audit is not None:
            audit.close()
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
