#!/usr/bin/env python3
"""One fresh learner: reproduce009, add basic tracking, then explicit v2 domains."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from types import SimpleNamespace

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import train_scratch_standing as base


def arguments(output, plan, *, num_envs=256, seed=None, profile="nominal", strength=0.):
    return SimpleNamespace(output=output, bank=ROOT/"reports/robust_policy_repair_20261024/pose_bank_height_matched073.json",
        updates=plan["max_updates"], num_envs=num_envs, rollout_steps=128, seed=plan["seed"] if seed is None else seed,
        all_heights_after=512, leg_target_rate_cost=None, device="cuda:0", fixed_target_probe=False,
        preflight_only=False, upright_task=None, standard_prone_task=None,
        randomization_study=(None if profile == "nominal" else ROOT/plan["randomization_contract"]),
        randomization_profile=(None if profile == "nominal" else profile), randomization_strength=strength if profile != "nominal" else 1.,
        **plan["standing_recipe"])


def configuration(output, plan, *, task="standing", profile="nominal", strength=0., num_envs=256,
                  seed=None, fully_randomized=False, evaluation=False):
    output.mkdir(parents=True, exist_ok=True)
    args = arguments(output, plan, num_envs=num_envs, seed=seed, profile=profile, strength=strength)
    cfg, settings = base.prepare(args)
    if task != "standing":
        settings["standing009_curriculum"] = deepcopy(plan)
        settings["episode_seconds"] = plan["tracking"]["episode_seconds"]
        cfg["episode_seconds"] = settings["episode_seconds"]
        cfg["record_diagnostics"] = True
    if evaluation:
        settings["standing009_curriculum"] = deepcopy(plan)
        settings["episode_seconds"] = plan["evaluation"]["seconds"]
        cfg["episode_seconds"] = settings["episode_seconds"]
        cfg["record_diagnostics"] = True
    # A held-out randomized matrix covers every case, separately from the25%
    # nominal rehearsal mask used in training. This override never shrinks ranges.
    if fully_randomized:
        if profile == "nominal":
            raise ValueError("Fully randomized holdout needs an explicit profile")
        settings["nominal_fraction"] = 0.
        cfg["dynamics_randomization"]["enabled_fraction"] = 1.
    design_path = output/"config/design.json"
    design = json.loads(design_path.read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    for key in ("episode_seconds", "dynamics_randomization", "record_diagnostics"):
        if key in row["contract"]:
            row["contract"][key] = deepcopy(cfg[key])
    design["scope"] = "009_reproduction_"+task
    base.write_json(design_path, design)
    cfg["design_preflight"]["sha256"] = base.digest(design_path)
    base.write_json(output/"config/contract.json", cfg)
    contract = json.loads((output/"task_contract.json").read_text())
    contract.update(schema="009-reproduction-task-v1", settings=settings,
        mechanism_config_sha256=base.digest(output/"config/contract.json"),
        reference_commit=plan["reference_commit"], curriculum_plan=deepcopy(plan),
        task=task, domain_profile=profile, domain_strength=strength, fully_randomized_holdout=fully_randomized,
        cold_evaluation=evaluation,
        standing_mid_height_until="actual30s behavior gate;512 minimum,1024 maximum",
        declared_mechanical_difference=plan["declared_mechanical_difference"])
    base.write_json(output/"task_contract.json", contract)
    return cfg, settings


def source_hashes():
    result = base.sources()
    for name in ("scripts/train_009_curriculum.py", "scripts/evaluate_009_curriculum.py"):
        result[name] = base.digest(ROOT/name)
    return result


def check_other_learners():
    names = {"train_009_curriculum.py", "train_standard_prone.py", "train_upright_locomotion.py",
             "train_scratch_standing.py", "train_fresh_unified.py"}
    owners = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors="replace") for x in (proc/"cmdline").read_bytes().split(b"\0") if x]
        except (OSError, ProcessLookupError):
            continue
        if argv and "python" in Path(argv[0]).name and any(Path(a).name in names for a in argv[1:3]):
            owners.append(dict(pid=int(proc.name), argv=argv[:3]))
    if owners:
        raise RuntimeError("Another wheel-leg native learner is present: "+str(owners))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT/"contracts/wheel_leg_009_reproduction_v1.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--native-probe", action="store_true")
    parser.add_argument("--signal-probe", action="store_true", help="One fresh PPO update, then self-SIGTERM at its complete boundary")
    args = parser.parse_args()
    if args.signal_probe and (args.native_probe or args.preflight_only):
        parser.error("Signal probe is a separate single-update fresh learner")
    from wheeled_tasks.chassis.standing009_curriculum import validate, domain_stage
    plan = validate(json.loads(args.plan.read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"curriculum_plan.input.json").write_bytes(args.plan.read_bytes())
    prepared_cfg, prepared_settings = configuration(output, plan)
    hashes = source_hashes()
    if args.preflight_only:
        base.write_json(output/"preflight.json", dict(source_hashes=hashes, checkpoint_reads=0,
            plan_sha256=base.digest(args.plan), max_updates=plan["max_updates"], new_PPO=0,
            phases=plan["phases"], initial_settings=prepared_settings))
        return
    check_other_learners()
    committed = base.archive_committed_sources(hashes, output)
    for name, expected in hashes.items():
        data = (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Source changed during freeze: "+name)
        target = output/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    base.write_json(output/"source_hashes.json", hashes)
    (output/"pose_bank.json").write_bytes(arguments(output, plan).bank.read_bytes())
    report = dict(schema="009-reproduction-progress-v1", status="starting", pid=os.getpid(),
        start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        started_utc=datetime.now(timezone.utc).isoformat(), requested_updates=plan["max_updates"],
        completed_updates=0, optimizer_attempts=0, checkpoint_reads=[], phase="standing", phase_updates=0,
        phases=[], contexts=[], cold_reviews=[], committed_source=committed, source_hashes=hashes,
        plan_sha256=base.digest(args.plan), hardware_ready=False, full_capability_qualified=False)
    report["diagnostic_signal_probe"] = args.signal_probe
    base.write_json(output/"progress.json", report)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        report["requested_stop_signal"] = signum
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    launcher = env = runner = audit = None
    original_load = None
    start = time.monotonic()
    context_number = 0
    active_context = output
    active_task = prepared_settings
    active_cfg = prepared_cfg
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        # AppLauncher replaces SIGTERM. Restore the update-boundary handler
        # after startup, and verify it again after each native context build.
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        import numpy as np
        import torch
        base.torch = torch
        from isaaclab.sim import SimulationContext
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.chassis.scratch_standing_env import ScratchStandingEnv
        from wheeled_tasks.chassis.standing009_env import Standing009TrackingEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install as install_physical
        from wheeled_tasks.chassis.precise_standing_reward import install as install_precision, scores
        from wheeled_tasks.chassis.standing009_curriculum import install_tracking_reward, install_tracking_failures, tracking_density
        from wheeled_tasks.chassis.standing_failure_termination import install as install_failures
        from wheeled_tasks.chassis.standing_contact_continuity import install as install_contact
        from wheeled_tasks.chassis.randomization_study import runtime_samples
        from wheeled_algo.scratch_standing import agent_config, initialize_fresh, tensor_hash
        from wheeled_algo.standing_load_initialization import initial_action
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.native_ppo_audit import NativePPOAudit
        from evaluate_009_curriculum import evaluate
        torch.set_num_threads(4)
        torch.manual_seed(plan["seed"])
        random.seed(plan["seed"])
        np.random.seed(plan["seed"])
        original_load = torch.load
        def forbidden_load(*positional, **keywords):
            report["checkpoint_reads"].append(str(positional[0] if positional else keywords.get("f")))
            raise RuntimeError("Every checkpoint/teacher/donor load is forbidden")
        torch.load = forbidden_load

        def build(task="standing", profile="nominal", strength=0., *, evaluation=False, fully_randomized=False,
                  phase_updates=0):
            nonlocal env, context_number, active_context, active_task, active_cfg
            if env is not None:
                env.close()
                if runner is not None:
                    runner.env = None
                env = None
                SimulationContext.clear_instance()
                gc.collect()
                torch.cuda.empty_cache()
            if context_number == 0:
                cfg, settings, directory = deepcopy(prepared_cfg), deepcopy(prepared_settings), output
            else:
                directory = output/"contexts"/f"{context_number:03d}_{task}_{profile}_{strength:g}_{'eval' if evaluation else 'train'}"
                directory.mkdir(parents=True, exist_ok=False)
                cfg, settings = configuration(directory, plan, task=task, profile=profile, strength=strength,
                    num_envs=64 if args.native_probe else 256,
                    seed=plan["evaluation"]["seed"] if evaluation else plan["seed"]+context_number*31,
                    fully_randomized=fully_randomized, evaluation=evaluation)
            if args.native_probe and context_number == 0:
                # The probe is a separately declared small wiring job, never PPO.
                cfg, settings = configuration(output/"probe_initial", plan, num_envs=64)
                directory = output/"probe_initial"
            cls = Standing009TrackingEnv if task != "standing" or evaluation else ScratchStandingEnv
            manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
            env = cls(cfg, manifest, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
                stage_name="speed", num_envs=64 if args.native_probe else 256, device="cuda:0",
                seed=plan["evaluation"]["seed"] if evaluation else plan["seed"]+(context_number*31 if context_number else 0),
                level=1., run_role="train", bank=output/"pose_bank.json", settings=settings, all_heights_after=10**9)
            env.actor_updates = report["completed_updates"]
            if hasattr(env, "course_updates"):
                env.course_updates = 256 if evaluation else phase_updates
                env.reset(torch.arange(env.num_envs, device=env.device))
            physical = install_physical(env, speed_weight=2., contact_weight=8., torque_weight=.05, include_motion=task != "standing")
            precision = install_precision(env) if task == "standing" else install_tracking_reward(env, plan)
            failures = install_failures(env) if task == "standing" else install_tracking_failures(env)
            contact = install_contact(env, 0.)
            report["physical_reward"], report["precise_reward"] = physical, env.precise_standing_reward
            report["standing_failure_resets"] = failures
            report["standing_contact_continuity_contract"] = contact
            active_context, active_task, active_cfg = directory, settings, cfg
            if runner is not None:
                runner.env = env
            base.write_json(directory/"native_startup.json", env.startup_report)
            base.write_json(directory/"domain_samples.json", runtime_samples(env))
            report["contexts"].append(dict(index=context_number, directory=str(directory), task=task,
                profile=profile, strength=strength, evaluation=evaluation, fully_randomized=fully_randomized,
                enabled_environments=int(env.domain.enabled.sum()), nominal_environments=int((~env.domain.enabled).sum()),
                completed_updates=report["completed_updates"], configuration_sha256=base.digest(directory/"config/contract.json")))
            context_number += 1
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            report["stop_handler_registration"] = dict(after_app_launcher=True,
                SIGTERM_is_update_boundary_handler=signal.getsignal(signal.SIGTERM) is request_stop,
                SIGINT_is_update_boundary_handler=signal.getsignal(signal.SIGINT) is request_stop)
            base.write_json(output/"progress.json", report)
            return env

        build()
        if args.native_probe:
            probes = []
            for task, profile, strength, steps in (("standing", "nominal", 0., 40),
                    ("tracking", "nominal", 0., 80), ("tracking", "requested", 1., 40)):
                if probes:
                    build(task, profile, strength, phase_updates=256)
                if task == "tracking":
                    # A labelled clock injection exercises nonzero commands in
                    # this wiring-only constant-target probe, without PPO.
                    env.episode_length_buf.fill_(round(4.5/env.policy_dt))
                prior = json.loads((active_context/"task_contract.json").read_text())["static_load_initialization"]
                action = initial_action(base.nominal_action(env), prior).expand(env.num_envs, -1).clone()
                check = dict(rates=env.dt == .001 and env.policy_dt == .005,
                    no_passive_births=not bool(env.passive.any()), no_recovery_births=not bool(env.recovery_lane.any()),
                    empty_checkpoint_reads=not report["checkpoint_reads"],
                    fixed_PD_and_spring=bool((env.domain.kp == 1.).all() and (env.domain.kd == 1.).all() and (env.domain.spring == 1.).all()))
                requested_peak = torch.zeros(3, device=env.device)
                for _ in range(steps):
                    with torch.inference_mode():
                        obs, reward, done, extra = env.step(action)
                    check["finite"] = check.get("finite", True) and bool(torch.isfinite(reward).all()) and all(bool(torch.isfinite(v).all()) for v in obs.values())
                    check["five_physical_samples"] = env.physical_standing_last["dq"].shape == (5, env.num_envs, 4)
                    state = extra["diagnostics"]
                    density = (scores(state["height"], state["gravity"], state["velocity"], state["omega"], state["commands"][:,2])
                        if task == "standing" else tracking_density(state["height"], state["gravity"], state["velocity"], state["omega"], state["commands"], plan))
                    physical = env.physical_standing_last
                    expected = density*env.policy_dt-physical["paid_speed"]-physical["paid_contact"]-physical["paid_slew"]-3.*state["terminated"]
                    if task != "standing":
                        expected -= env.tracking009_last["paid_retention"]
                    eligible = extra["policy_learning_mask"]
                    check["actual_reward_payment_matches_declared"] = check.get("actual_reward_payment_matches_declared", True) and bool(torch.allclose(reward[eligible], expected[eligible], atol=1e-6, rtol=1e-5))
                    requested_peak = torch.maximum(requested_peak, state["commands"].abs().max(0).values)
                if task == "tracking":
                    check["nonzero_motion_commands_observed"] = bool(requested_peak[0] > .25 and requested_peak[1] > .5)
                    if plan.get("repair"):
                        learning = env.course_height_learning
                        check["training_births_remain_305mm"] = all(c["birth_height_m"] == .305 for c in env.course_cases)
                        check["transition_error_exceeds_old_25mm_cutoff"] = bool(((state["height"]-state["commands"][:, 2]).abs()[learning] > .025).all())
                        check["height_trials_not_killed_for_target_error"] = not bool(state["terminated"][learning].any())
                        check["paid_height_cost_nonzero_on_requested_transition"] = bool((env.tracking009_last["paid_height"][learning] > 0.).all())
                        check["standing_rehearsal_clock_30s"] = bool((env.episode_limits[env.course_role_masks["stand"]] == 6000).all())
                        columns = [env.cfg["actor_frame_dim"]+6+env.ids[i] for i in (2, 5)]
                        error_xy = env.robot.data.root_link_pose_w.torch[:, :2]-env.origins[:, :2]-env.course_anchor
                        check["privileged_critic_anchor_matches_true_position"] = bool(torch.allclose(obs["critic"][:, columns], (error_xy*10.).clamp(-10., 10.), atol=1e-6))
                        check["original_actor35_critic81_widths"] = obs["policy"].shape[-1] == 35 and obs["critic"].shape[-1] == 81
                probes.append(dict(task=task, profile=profile, strength=strength, policy_steps=steps,
                    checks=check, all_checks_pass=all(check.values()), requested_command_abs_peak=requested_peak.cpu().tolist(),
                    enabled_environments=int(env.domain.enabled.sum()), new_PPO=0, learned_policy=False,
                    clock_injection_seconds=4.5 if task == "tracking" else None))
            # Exercise the actual cold-trace writer and case reset separately.
            # The short duration is explicitly wiring-only, never a behavior gate.
            from torch.utils.tensorboard import SummaryWriter
            build("standing", evaluation=True)
            prior = json.loads((active_context/"task_contract.json").read_text())["static_load_initialization"]
            action = initial_action(base.nominal_action(env), prior).expand(env.num_envs, -1).clone()
            short_plan = deepcopy(plan)
            short_plan["evaluation"]["seconds"] = .2
            with SummaryWriter(str(output/"cold_wiring_events")) as writer:
                cold_probe = evaluate(env, lambda obs: action, output/"cold_wiring_only_0p2s",
                    writer, 0, short_plan, "standing")
            report["cold_wiring_probe"] = dict(duration_override_seconds=.2,
                declared_behavior_duration_seconds=plan["evaluation"]["seconds"],
                case_count=cold_probe["case_count"], trace_written=True,
                model_created=False, new_PPO=0, behavior_qualification=False)
            report.update(status="native_probe_completed_unqualified", native_probes=probes,
                          all_native_checks_pass=all(p["all_checks_pass"] for p in probes), new_PPO=0)
            if not report["all_native_checks_pass"]:
                raise RuntimeError("Native009 wiring/rebuild probe failed")
            return

        runner_cfg = agent_config(plan["seed"], "cuda:0", 128, .0003, normalize_observations=True,
            wheel_velocity_increment=True, separate_leg_velocity_feedback=True)
        base.write_json(output/"agent_config.json", runner_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(output), device="cuda:0")
        prior = json.loads((output/"task_contract.json").read_text())["static_load_initialization"]
        report["initialization"] = initialize_fresh(runner.alg, initial_action(base.nominal_action(env), prior), output_bias_source=prior["source"])
        report["initialization"].update(actor_class=runner_cfg["actor"]["class_name"],
            wheel_velocity_increment=True, separate_leg_velocity_feedback=True, static_load_initialization=prior)
        base.write_json(output/"initialization.json", report["initialization"])
        initial_actor = {k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        initial_critic = {k:v.detach().cpu().clone() for k,v in runner.alg.critic.state_dict().items()}
        algorithm_identity = id(runner.alg)
        optimizer_identity = id(runner.alg.optimizer)
        actor_identity, critic_identity = id(runner.alg.actor), id(runner.alg.critic)
        mask = RecoveryRolloutMask(runner.alg, num_envs=256)
        storage_identity = id(runner.alg.storage)
        native_update = runner.alg.update

        def save(path, infos=None):
            path = Path(path)
            state = runner.alg.save()
            state.update(iter=report["completed_updates"], infos=dict(schema="009-reproduction-checkpoint-v1",
                training_origin="scratch", continuation_forbidden=True, initialization=report["initialization"],
                completed_updates=report["completed_updates"], phase=report["phase"], phase_updates=report["phase_updates"],
                configuration_directory=str(active_context), settings=active_task, source_hashes=hashes,
                committed_source=committed, plan_sha256=report["plan_sha256"],
                mechanism_config_sha256=base.digest(active_context/"config/contract.json"),
                agent_config_sha256=base.digest(output/"agent_config.json"), bank_sha256=base.digest(output/"pose_bank.json"),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                env_rng=env.generator.get_state(), domain_rng=env.domain.rng.get_state()))
            temporary = path.with_suffix(path.suffix+".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            report["last_checkpoint"] = dict(path=str(path), sha256=base.digest(path), updates=report["completed_updates"])

        def identities():
            return (id(runner.alg), id(runner.alg.optimizer), id(runner.alg.actor), id(runner.alg.critic), id(runner.alg.storage)) == (
                algorithm_identity, optimizer_identity, actor_identity, critic_identity, storage_identity)

        def update():
            nonlocal audit
            report["optimizer_attempts"] += 1
            report["optimizer_update_in_flight"] = True
            base.write_json(output/"progress.json", report)
            losses = native_update()
            report["completed_updates"] += 1
            report["phase_updates"] += 1
            report["optimizer_update_in_flight"] = False
            if not identities() or not all(bool(torch.isfinite(p).all()) for m in (runner.alg.actor, runner.alg.critic) for p in m.parameters()):
                raise RuntimeError("Learner identity or finite-parameter guard failed")
            if report["completed_updates"] == 1:
                audit.close()
                result = audit.report()
                result.update(actor_changed_parameter_tensors=sum(not torch.equal(initial_actor[k], v.detach().cpu()) for k,v in runner.alg.actor.named_parameters()),
                    critic_changed_parameter_tensors=sum(not torch.equal(initial_critic[k], v.detach().cpu()) for k,v in runner.alg.critic.named_parameters()),
                    actor_sha256=tensor_hash(runner.alg.actor.state_dict()), critic_sha256=tensor_hash(runner.alg.critic.state_dict()))
                base.write_json(output/"first_update_audit.json", result)
                if not result["all_recorded_values_finite"] or result["committed_optimizer_calls"] != 12 or not result["actor_changed_parameter_tensors"] or not result["critic_changed_parameter_tensors"]:
                    raise RuntimeError("First fresh PPO audit failed")
                audit = None
                if args.signal_probe:
                    os.kill(os.getpid(), signal.SIGTERM)
                    if not stop_requested:
                        raise RuntimeError("SIGTERM did not reach the update-boundary handler")
            env.actor_updates = report["completed_updates"]
            if hasattr(env, "course_updates"):
                env.course_updates = 256 if report["phase"] == "domains" else report["phase_updates"]
            report.update(status="training", elapsed_s=time.monotonic()-start,
                last_losses={k:float(v) for k,v in losses.items()}, valid_rollout=mask.metrics,
                last_physical_metrics={k:float(v) for k,v in env.physical_standing_last["metrics"].items()},
                live_learner_and_optimizer_identity_preserved=True)
            base.write_json(output/"progress.json", report)
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            return losses

        runner.save, runner.alg.update = save, update
        save(output/"model_initial.pt")
        audit = NativePPOAudit(runner.alg)

        def cold(task, profile="nominal", strength=0.):
            report["status"] = "cold_review"
            base.write_json(output/"progress.json", report)
            before = (tensor_hash(runner.alg.actor.state_dict()), tensor_hash(runner.alg.critic.state_dict()),
                      [int(v["step"]) for v in runner.alg.optimizer.state.values()])
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            np_rng, python_rng = np.random.get_state(), random.getstate()
            runner.alg.eval_mode()
            build(task, profile, strength, evaluation=True, fully_randomized=profile != "nominal")
            review_dir = output/"reviews"/f"u{report['completed_updates']:04d}_{task}_{profile}_{strength:g}"
            result = evaluate(env, runner.alg.actor, review_dir, runner.logger.writer,
                report["completed_updates"], plan, task, stop_requested=lambda: stop_requested)
            after = (tensor_hash(runner.alg.actor.state_dict()), tensor_hash(runner.alg.critic.state_dict()),
                     [int(v["step"]) for v in runner.alg.optimizer.state.values()])
            if before != after or not identities():
                raise RuntimeError("Cold review changed learner/normalization/Adam state")
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            np.random.set_state(np_rng)
            random.setstate(python_rng)
            result.update(live_actor_critic_Adam_unchanged=True, model_or_optimizer_loaded=False)
            result.update(actor_tensor_sha256=before[0], critic_tensor_sha256=before[1],
                          checkpoint_before_review=deepcopy(report["last_checkpoint"]),
                          committed_source=committed["commit"])
            base.write_json(review_dir/"result.json", result)
            report["cold_reviews"].append(dict(path=str(review_dir/"result.json"), completed_updates=report["completed_updates"],
                task=task, profile=profile, strength=strength, all_pass=result["all_pass"],
                pass_count=result["baseline_pass"], case_count=result["case_count"]))
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            return result["all_pass"]

        first_chunk = True
        all_phases_pass = True
        for phase in plan["phases"]:
            report.update(phase=phase["id"], phase_updates=0)
            if phase["id"] != "standing":
                profile, strength = (domain_stage(plan, 0) if phase["id"] == "domains" else ("nominal", 0.))
                build(phase["task"], profile, strength, phase_updates=256 if phase["id"] == "domains" else 0)
            phase_pass = False
            while report["phase_updates"] < phase["max_updates"]:
                if stop_requested:
                    save(output/"model_stopped.pt")
                    raise base.TrainingStopped()
                interval = 128 if phase["id"] == "domains" else plan["review_interval_updates"]
                length = min(interval, phase["max_updates"]-report["phase_updates"])
                if runner.logger.writer is not None:
                    runner.logger.writer.flush()
                    runner.logger.writer.close()
                    runner.logger.writer = None
                runner.current_learning_iteration = report["completed_updates"]
                runner.learn(length, init_at_random_ep_len=first_chunk)
                first_chunk = False
                save(output/f"model_complete_{report['completed_updates']:04d}.pt")
                if report["phase_updates"] >= phase["min_updates"]:
                    phase_pass = cold(phase["task"])
                    if phase["id"] == "domains":
                        phase_pass = cold("tracking", "requested", 1.) and phase_pass
                    if phase_pass:
                        break
                if report["phase_updates"] == phase["max_updates"]:
                    break
                if phase["id"] == "domains":
                    profile, strength = domain_stage(plan, report["phase_updates"])
                    build("tracking", profile, strength, phase_updates=256)
                elif report["phase_updates"] >= phase["min_updates"]:
                    build(phase["task"], phase_updates=report["phase_updates"])
            report["phases"].append(dict(id=phase["id"], completed_phase_updates=report["phase_updates"],
                                        completed_total_updates=report["completed_updates"], behavior_gate_pass=phase_pass))
            if not phase_pass:
                all_phases_pass = False
                report["stop_reason"] = "Phase behavior gate failed at its declared maximum; no automatic blind restart"
                break
        save(output/"model_final.pt")
        report.update(status="completed_basic_scope_unqualified" if all_phases_pass else "completed_gate_failed_unqualified",
            all_planned_basic_gates_pass=all_phases_pass, elapsed_s=time.monotonic()-start)
    except base.TrainingStopped:
        report.update(status="stopped", elapsed_s=time.monotonic()-start)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_s=time.monotonic()-start)
        raise
    finally:
        report["source_mismatches_at_exit"] = [n for n,h in hashes.items() if base.digest(ROOT/n) != h]
        base.write_json(output/"progress.json", report)
        if original_load is not None:
            import torch
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
