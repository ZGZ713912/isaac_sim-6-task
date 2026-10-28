#!/usr/bin/env python3
"""Fresh10000-update six-motor RL: standard prone, quiet standing and locomotion."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import train_scratch_standing as base


def configuration(output, plan, *, profile="nominal", strength=0., evaluation=False, num_envs=None):
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(output=output,
        bank=ROOT/"reports/robust_policy_repair_20261024/pose_bank_height_matched073.json",
        updates=plan["max_updates"], num_envs=num_envs or plan["num_envs"], rollout_steps=plan["rollout_steps"],
        seed=plan["evaluation"]["seed"] if evaluation else plan["seed"], all_heights_after=1000,
        leg_target_rate_cost=None, device="cuda:0", fixed_target_probe=False, preflight_only=False,
        upright_task=None, standard_prone_task=None, precise_objective=False,
        normalize_observations=True, desync_episodes=False, standing_failures=False,
        wheel_velocity_increment=False, separate_leg_velocity_feedback=False, static_load_initialization=False,
        contact_break_cost=None, speed_weight=2., contact_weight=8., learning_rate=plan["learning_rate"],
        randomization_study=None if profile == "nominal" else ROOT/plan["randomization_contract"],
        randomization_profile=None if profile == "nominal" else profile, randomization_strength=strength)
    cfg, settings = base.prepare(args)
    settings.update(version="prone-motion-long-v1", prone_motion_task=deepcopy(plan),
        prone_geometry=json.loads((ROOT/plan["prone_geometry_contract"]).read_text()),
        quiet_training=False, quiet_critic_anchor=True, hold_motion_during_recovery=True,
        motion_fraction=.7, motion_push_fraction=0., stable_seconds=plan["completion_hold_seconds"],
        episode_seconds=plan["training_prone_seconds"], recovery_timeout_seconds=plan["rise_deadline_seconds"],
        reward_potential_gain=0., reward_gamma=.9974905699336811, boundary_xy_m=[45., 45.])
    if evaluation and profile != "nominal":
        settings["nominal_fraction"] = 0.
        cfg["dynamics_randomization"]["enabled_fraction"] = 1.
    cfg.update(episode_seconds=plan["training_prone_seconds"], record_diagnostics=True,
               curriculum="prone_motion_long_v1", learning_rate=plan["learning_rate"], entropy_coef=.001)
    cfg["stages"][0].update(vx_max=3., yaw_max=plan["spin_target_rpm"]*math.tau/60.)
    cfg["policy_architecture"]["scope"] = "Fresh actor35, critic83, six fully trainable motor outputs"
    cfg["v5_control"]["action_clip"] = 16.
    for key in ("wheel_static_friction", "wheel_dynamic_friction"):
        cfg["contact_material"][key] = plan["friction"]["wheel"]
    if "randomization_study" in settings:
        study = settings["randomization_study"]
        study["effective_wheel_ground_friction_range"] = [
            (v+plan["friction"]["wheel"])/2. for v in study["ground_friction_range"]]
        base.write_json(output/"randomization_study.resolved.json", study)
    design_path = output/"config/design.json"
    design = json.loads(design_path.read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    for key in row["contract"]:
        if key in cfg:
            row["contract"][key] = deepcopy(cfg[key])
    row["runtime"]["contact_material"] = deepcopy(cfg["contact_material"])
    design["scope"] = "fresh_prone_motion_long"
    base.write_json(design_path, design)
    cfg["design_preflight"]["sha256"] = base.digest(design_path)
    base.write_json(output/"config/contract.json", cfg)
    base.write_json(output/"task_contract.json", dict(schema="prone-motion-native-task-v1", plan=plan,
        settings=settings, actor_dim=35, critic_dim=83, action_dim=6, policy_hz=200, physics_hz=1000,
        wheel_reduction=cfg["wheel_reduction"], wheel_action="trainable absolute output-shaft velocity",
        action_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        wheel_velocity_scale_rad_s=10., wheel_velocity_clip_rad_s=90.,
        leg_position_scale_rad=.25, leg_action_clip=16., profile=profile, strength=strength,
        full_randomized_holdout=evaluation and profile != "nominal", state_writes="reset only",
        mechanism_config_sha256=base.digest(output/"config/contract.json"), hardware_ready=False,
        training_origin=plan["initialization"], static_load_initialization=None))
    from train_chassis import preflight
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    preflight(output/"config/contract.json")
    check = design_preflight(cfg, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
        stage="speed", role="train", seed=args.seed, num_envs=args.num_envs)
    base.write_json(output/"design_preflight.json", check)
    return cfg, settings, args.bank


def learner_config(plan, device="cuda:0"):
    from wheeled_algo.scratch_standing import agent_config
    result = agent_config(plan["seed"], device, plan["rollout_steps"], plan["learning_rate"],
                          normalize_observations=True)
    result["actor"]["hidden_dims"] = result["critic"]["hidden_dims"] = [256, 128, 64]
    result["algorithm"]["entropy_coef"] = .001
    result.update(max_iterations=plan["max_updates"], save_interval=250, experiment_name="prone_motion_long")
    return result


def source_hashes(plan_path, plan):
    result = base.sources()
    names = ["scripts/train_prone_motion.py", "scripts/evaluate_prone_motion.py", "scripts/train_prone_motion.sh",
             "tests/test_prone_motion.py", "docs/PRONE_MOTION_TRAINING_20261028.md",
             "contracts/scratch_standing_base_mechanism_v2.json", "contracts/scratch_standing_base_design_v2.json",
             plan["randomization_contract"],
             plan["acceptance_contract"], plan["prone_geometry_contract"], "contracts/standing_geometry107_v1.json",
             str(plan_path.relative_to(ROOT))]
    for name in names:
        result[name] = base.digest(ROOT/name)
    return result


def check_other_learners():
    names = {"train_009_curriculum.py", "train_standard_prone.py", "train_upright_locomotion.py",
             "train_scratch_standing.py", "train_fresh_unified.py", "train_prone_motion.py"}
    owners = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors="replace") for x in (proc/"cmdline").read_bytes().split(b"\0") if x]
            if argv and "python" in Path(argv[0]).name and any(Path(a).name in names for a in argv[1:3]):
                owners.append(dict(pid=int(proc.name), argv=argv[:3]))
        except OSError:
            pass
    if owners:
        raise RuntimeError("Refusing a concurrent wheel-leg learner. Let it finish or stop it first: "+str(owners))


def stop_previous_009():
    """Only the user's explicit CLI flag permits retiring the known old run."""
    run = Path("/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/reproduce009_20261028/standing_tracking_clockfix_v3_u3072_e256")
    progress_file = run/"progress.json"
    value = json.loads(progress_file.read_text())
    pid = value["pid"]
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        if value["status"] not in ("stopped", "completed_basic_scope_unqualified", "completed_gate_failed_unqualified"):
            raise RuntimeError("Old009 disappeared without a reviewed final status; inspect "+str(progress_file))
        return
    argv = [s.decode() for s in (proc/"cmdline").read_bytes().split(b"\0") if s]
    ticks = int((proc/"stat").read_text().rsplit(")",1)[1].split()[19])
    expected_cwd = Path("/home/noir/.codex/worktrees/standard-prone-training/isaac_wheeled_rl_train")
    if (ticks != 13480232 or value["start_ticks"] != ticks or pid != 2929818
            or (proc/"cwd").resolve() != expected_cwd
            or len(argv) < 4 or Path(argv[1]).name != "train_009_curriculum.py"
            or "--output" not in argv or Path(argv[argv.index("--output")+1]).resolve() != run):
        raise RuntimeError("Old009 identity changed; refusing to signal any process")
    print(f"Identity verified. Sending one SIGTERM to old009 PID{pid}; waiting for its complete-update checkpoint.", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+600.
    while proc.exists() and time.monotonic() < deadline:
        time.sleep(.5)
    value = json.loads(progress_file.read_text())
    checkpoint = value.get("last_checkpoint", {})
    if (proc.exists() or value["status"] != "stopped" or checkpoint.get("updates") != value["completed_updates"]
            or Path(checkpoint.get("path", "")).name != "model_stopped.pt"
            or not (run/"model_stopped.pt").is_file()):
        raise RuntimeError("Old009 has not sealed and exited cleanly; no new learner will start")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    if str(pid) in gpu.stdout.split():
        raise RuntimeError("Old009 still owns a GPU context; refusing to start")
    print(f"Old009 sealed at{value['completed_updates']} updates. No weights will be reused.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT/"contracts/wheel_leg_prone_motion_v1.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--native-probe", action="store_true", help="Zero PPO; native reset/action/reward/contact wiring only")
    parser.add_argument("--stop-existing-009", action="store_true", help="Explicitly retire ONLY the authenticated old009, seal and wait; never resume it")
    args = parser.parse_args()
    if args.preflight_only and args.native_probe:
        parser.error("Choose one probe type")
    from wheeled_tasks.chassis.prone_motion import validate, domain_stage
    plan_path = args.plan.resolve()
    plan = validate(json.loads(plan_path.read_text()))
    output = args.output.resolve()
    if args.stop_existing_009:
        if args.preflight_only:
            parser.error("A CPU preflight must not signal an existing learner")
        stop_previous_009()
    if not args.preflight_only:
        check_other_learners()
    output.mkdir(parents=True, exist_ok=False)
    (output/"curriculum_plan.input.json").write_bytes(plan_path.read_bytes())
    cfg, settings, bank = configuration(output, plan)
    hashes = source_hashes(plan_path, plan)
    if args.preflight_only:
        base.write_json(output/"preflight.json", dict(source_hashes=hashes, checkpoint_reads=[], new_PPO=0,
            max_updates=plan["max_updates"], rollout_slots=plan["max_updates"]*plan["num_envs"]*plan["rollout_steps"],
            training_origin=plan["initialization"], initialized_behavior_qualified=False))
        return
    committed = base.archive_committed_sources(hashes, output)
    base.write_json(output/"source_hashes.json", hashes)
    (output/"pose_bank.json").write_bytes(bank.read_bytes())
    base.write_json(output/"input_hashes.json", dict(bank_sha256=base.digest(bank), plan_sha256=base.digest(plan_path)))
    report = dict(schema="prone-motion-progress-v1", status="starting", pid=os.getpid(),
        start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        started_utc=datetime.now(timezone.utc).isoformat(), requested_updates=plan["max_updates"],
        completed_updates=0, optimizer_attempts=0, checkpoint_reads=[], contexts=[], cold_reviews=[],
        committed_source=committed, hardware_ready=False, full_capability_qualified=False)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        report["requested_stop_signal"] = signum
    launcher = env = runner = audit = original_load = None
    active_directory = output
    start = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        import numpy as np
        import torch
        from isaaclab.sim import SimulationContext
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.randomization_study import runtime_samples
        from wheeled_algo.scratch_standing import initialize_fresh, tensor_hash
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.native_ppo_audit import NativePPOAudit
        from evaluate_prone_motion import evaluate, probe
        torch.set_num_threads(4)
        torch.manual_seed(plan["seed"])
        random.seed(plan["seed"])
        np.random.seed(plan["seed"])
        base.torch = torch
        original_load = torch.load
        def forbidden_load(*a, **kw):
            report["checkpoint_reads"].append(str(a[0] if a else kw.get("f")))
            raise RuntimeError("Checkpoint/teacher/donor loads are forbidden")
        torch.load = forbidden_load

        def build(profile="nominal", strength=0., evaluation=False):
            nonlocal env, active_directory
            changed = [name for name, expected in hashes.items() if base.digest(ROOT/name) != expected]
            if changed:
                raise RuntimeError("Frozen source or input changed; refusing a new native context: "+str(changed))
            if env is not None:
                env.close()
                if runner is not None:
                    runner.env = None
                env = None
                SimulationContext.clear_instance()
                gc.collect()
                torch.cuda.empty_cache()
            index = len(report["contexts"])
            active_directory = output/"contexts"/f"{index:03d}_{profile}_{strength:g}_{'eval' if evaluation else 'train'}"
            active_directory.mkdir(parents=True, exist_ok=False)
            cfg, settings, _ = configuration(active_directory, plan, profile=profile, strength=strength,
                evaluation=evaluation, num_envs=40 if args.native_probe else plan["num_envs"])
            manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
            env = ProneMotionEnv(cfg, manifest, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
                stage_name="speed", num_envs=cfg["target_num_envs"], device="cuda:0",
                seed=plan["evaluation"]["seed"] if evaluation else plan["seed"]+index*31,
                level=1., run_role="train", bank=output/"pose_bank.json", settings=settings)
            env.actor_updates = report["completed_updates"]
            env.reset(torch.arange(env.num_envs, device=env.device))
            if runner is not None:
                runner.env = env
            report["contexts"].append(dict(index=index, profile=profile, strength=strength,
                evaluation=evaluation, directory=str(active_directory), completed_updates=report["completed_updates"]))
            base.write_json(active_directory/"native_startup.json", env.startup_report)
            base.write_json(active_directory/"domain_samples.json", runtime_samples(env))
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            base.write_json(output/"progress.json", report)

        build()
        if args.native_probe:
            reports = []
            for profile, strength in (("nominal", 0.), ("requested", 1.)):
                if stop_requested:
                    raise base.TrainingStopped()
                if reports:
                    build(profile, strength)
                reports.append(probe(env, plan, stop_requested=lambda: stop_requested))
            report.update(status="native_probe_completed_unqualified", new_PPO=0, native_probes=reports,
                          all_native_checks_pass=all(x["all_checks_pass"] for x in reports))
            if not report["all_native_checks_pass"]:
                raise RuntimeError("Native prone-motion wiring probe failed")
            return
        runner_cfg = learner_config(plan)
        base.write_json(output/"agent_config.json", runner_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(output), device="cuda:0")
        report["initialization"] = initialize_fresh(runner.alg, base.nominal_action(env), initial_std=plan["initial_std"])
        base.write_json(output/"initialization.json", report["initialization"])
        original_actor = {k:v.detach().cpu().clone() for k,v in runner.alg.actor.named_parameters()}
        original_critic = {k:v.detach().cpu().clone() for k,v in runner.alg.critic.named_parameters()}
        mask = RecoveryRolloutMask(runner.alg, num_envs=plan["num_envs"])
        def identity():
            return tuple(id(v) for v in (runner.alg, runner.alg.actor, runner.alg.critic, runner.alg.optimizer, runner.alg.storage))
        identities = identity()
        native_update = runner.alg.update

        def save(path, infos=None):
            state = runner.alg.save()
            state.update(iter=report["completed_updates"], infos=dict(schema="prone-motion-checkpoint-v1",
                training_origin="scratch", continuation_forbidden=True, initialization=report["initialization"],
                source_revision=committed, source_hashes=hashes, configuration=str(active_directory),
                completed_updates=report["completed_updates"], bank_sha256=base.digest(output/"pose_bank.json"),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                env_rng=env.generator.get_state(), domain_rng=env.domain.rng.get_state()))
            path = Path(path)
            temporary = path.with_suffix(path.suffix+".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            report["last_checkpoint"] = dict(path=str(path), sha256=base.digest(path), updates=report["completed_updates"])

        def update():
            nonlocal audit
            report["optimizer_attempts"] += 1
            report["optimizer_update_in_flight"] = True
            base.write_json(output/"progress.json", report)
            losses = native_update()
            report["completed_updates"] += 1
            report["optimizer_update_in_flight"] = False
            if identity() != identities or not all(torch.isfinite(p).all() for m in (runner.alg.actor, runner.alg.critic) for p in m.parameters()):
                raise RuntimeError("Learner identity or finite parameter guard failed")
            if report["completed_updates"] == 1:
                audit.close()
                result = audit.report()
                result.update(actor_changed_parameter_tensors=sum(not torch.equal(original_actor[k], v.detach().cpu()) for k,v in runner.alg.actor.named_parameters()),
                    critic_changed_parameter_tensors=sum(not torch.equal(original_critic[k], v.detach().cpu()) for k,v in runner.alg.critic.named_parameters()))
                base.write_json(output/"first_update_audit.json", result)
                if not (result["all_recorded_values_finite"] and result["committed_optimizer_calls"] == 12
                        and result["actor_changed_parameter_tensors"] and result["critic_changed_parameter_tensors"]):
                    raise RuntimeError("First fresh PPO audit failed")
                audit = None
            env.actor_updates = report["completed_updates"]
            report.update(status="training", elapsed_s=time.monotonic()-start, valid_rollout=mask.metrics,
                          last_losses={k:float(v) for k,v in losses.items()})
            base.write_json(output/"progress.json", report)
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            return losses

        runner.save, runner.alg.update = save, update
        save(output/"model_initial.pt")
        audit = NativePPOAudit(runner.alg)

        def cold(profile="nominal", strength=0.):
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            report["status"] = "cold_review"
            before = (tensor_hash(runner.alg.actor.state_dict()), tensor_hash(runner.alg.critic.state_dict()),
                      [int(v["step"]) for v in runner.alg.optimizer.state.values()])
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
            runner.alg.eval_mode()
            build(profile, strength, evaluation=True)
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            directory = output/"reviews"/f"u{report['completed_updates']:05d}_{profile}"
            result = evaluate(env, runner.alg.actor, directory, runner.logger.writer,
                              report["completed_updates"], plan, stop_requested=lambda: stop_requested)
            after = (tensor_hash(runner.alg.actor.state_dict()), tensor_hash(runner.alg.critic.state_dict()),
                     [int(v["step"]) for v in runner.alg.optimizer.state.values()])
            if before != after or identity() != identities:
                raise RuntimeError("Cold evaluation changed the live learner or Adam")
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
            np.random.set_state(rng[2]); random.setstate(rng[3])
            result.update(checkpoint=deepcopy(report["last_checkpoint"]), source_revision=committed,
                          actor_sha256=before[0], critic_sha256=before[1], learner_unchanged=True,
                          profile=profile, strength=strength, fully_randomized_holdout=profile != "nominal")
            base.write_json(directory/"result.json", result)
            report["cold_reviews"].append(dict(path=str(directory/"result.json"), completed_updates=report["completed_updates"],
                passed=result["pass_count"], cases=result["case_count"], all_pass=result["all_pass"], profile=profile))

        while report["completed_updates"] < plan["max_updates"]:
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            completed = report["completed_updates"]
            profile, strength = domain_stage(plan, completed)
            current = report["contexts"][-1]
            if current["evaluation"] or (profile, strength) != (current["profile"], current["strength"]):
                build(profile, strength)
            future = [plan["max_updates"], completed+250]
            future += [v for v in plan["review_updates"] if v > completed]
            future += [x["start_update"] for x in plan["domain_sequence"] if x["start_update"] > completed]
            length = min(future)-completed
            if runner.logger.writer is not None:
                runner.logger.writer.flush(); runner.logger.writer.close(); runner.logger.writer = None
            runner.current_learning_iteration = completed
            runner.learn(length, init_at_random_ep_len=False)
            save(output/f"model_complete_{report['completed_updates']:05d}.pt")
            if report["completed_updates"] in plan["review_updates"]:
                cold()
                if report["completed_updates"] >= 8000:
                    cold("requested", 1.)
                if stop_requested:
                    save(output/"model_stopped.pt")
                    raise base.TrainingStopped()
        save(output/"model_final.pt")
        report.update(status="completed_reviewed_unqualified", elapsed_s=time.monotonic()-start)
    except base.TrainingStopped:
        report.update(status="stopped", elapsed_s=time.monotonic()-start)
        if args.native_probe:
            raise SystemExit(130)
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
            runner.logger.writer.flush(); runner.logger.writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
