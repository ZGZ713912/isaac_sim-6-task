#!/usr/bin/env python3
"""Small, auditable actor-transfer blocks; fresh critic, preserved flat mean."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FlatHeightCommandMetrics:
    """Per-frame height strata for existing behavior metrics; logging only.

    Bin edges are the midpoints of the three training heights .23/.305/.42m.
    Classify the visible end-state command, never a lane's last sampled target.
    """
    bands = ("height_low", "height_nominal", "height_high")
    tags = ("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "planar_speed_mean_m_s", "terminated_frame_fraction")
    edges_m = ((.23 + .305) / 2., (.305 + .42) / 2.)

    def __init__(self, count, device):
        import torch
        self.sums = torch.zeros(2, 3, count, 5, dtype=torch.float64, device=device)
        self.frames = torch.zeros(2, 3, count, dtype=torch.int64, device=device)

    @classmethod
    def contract(cls):
        return dict(schema="flat-height-command-metrics-v1", bands=list(cls.bands),
            height_bin_edges_m=list(cls.edges_m),
            intervals="low < lower edge; nominal >= lower and < upper edge; high >= upper edge",
            source="same post-step diagnostic command height used by existing FlatBehavior/FlatBirth metrics",
            windows="same after1s behavior and first1s birth masks as existing logger",
            count_unit="policy frames; per-frame classification across physical resets",
            step_unit="completed optimizer updates", units=dict(vx_mae_m_s="m/s", yaw_mae_rad_s="rad/s",
                height_mae_m="m", planar_speed_mean_m_s="m/s", terminated_frame_fraction="fraction"),
            scope="diagnostic only; no reward, sampling, optimizer, RNG or actor-input changes")

    def observe(self, height_command, values, behavior_valid, birth_valid):
        import torch
        height_command = height_command.detach()
        values = values.detach().double()
        low, high = self.edges_m
        height_masks = torch.stack((height_command < low,
            (height_command >= low) & (height_command < high), height_command >= high))
        windows = torch.stack((behavior_valid, birth_valid))
        selected = windows[:, None, :] & height_masks[None, :, :]
        self.frames.add_(selected)
        self.sums.add_(selected[:, :, :, None] * values[None, None, :, :])

    def log(self, writer, view, selected_lanes, step):
        for window, namespace in enumerate(("FlatBehavior", "FlatBirth")):
            for band, label in enumerate(self.bands):
                frames = self.frames[window, band, selected_lanes].sum()
                prefix = f"{namespace}/{view}/{label}"
                writer.add_scalar(prefix + "/sample_frames", int(frames), step)
                if not frames:
                    continue  # Empty strata cannot masquerade as perfect tracking.
                means = self.sums[window, band, selected_lanes].sum(0) / frames
                for tag, value in zip(self.tags, means.cpu().tolist()):
                    writer.add_scalar(prefix + "/" + tag, value, step)

    def clear(self):
        self.sums.zero_()
        self.frames.zero_()




class FlatTerminationMetrics:
    """Count native terminal and cause flags; causes may overlap on one frame."""
    reasons = ('fall','nonwheel_contact','knee','boundary','takeoff_timeout','closure_gap',
               'jump_request_timeout','jump_outcome_timeout','spring_travel')
    flags = ('terminated','done','success') + reasons
    windows = ('whole','first1s','after1s')
    bands = ('height_low','height_nominal','height_high')

    def __init__(self,count,device):
        import torch
        self.counts=torch.zeros(3,3,count,len(self.flags),dtype=torch.int64,device=device)

    @classmethod
    def contract(cls):
        return dict(schema='flat-native-termination-counts-v1',
            flags=list(cls.flags),windows=list(cls.windows),height_bands=list(cls.bands),
            unit='native flagged policy frames per completed optimizer update',
            step_unit='completed optimizer updates',
            source='unaltered pre-auto-reset diagnostics and reasons from the native environment',
            cause_semantics='Native cause flags can overlap; sum of causes is not a unique episode count. terminated is separate from done, success and boundary.',
            scope='Diagnostic counts only; no reward, termination, sampling, state, RNG or optimizer changes')

    def observe(self,data,behavior_valid,birth_valid):
        import torch
        height=data['commands'][:,2].detach()
        lo,hi=(.23+.305)/2,(.305+.42)/2
        height_masks=torch.stack((height<lo,(height>=lo)&(height<hi),height>=hi))
        windows=torch.stack((behavior_valid|birth_valid,birth_valid,behavior_valid))
        flags=torch.stack([data[n].detach().bool() if n in ('terminated','done','success')
                           else data['reasons'][n].detach().bool() for n in self.flags],-1)
        selected=windows[:,None,:]&height_masks[None,:,:]
        self.counts.add_(selected[:,:,:,None]&flags[None,None,:,:])

    def log(self,writer,view,selected_lanes,step):
        selected=self.counts[:,:,selected_lanes,:].sum(2).cpu().tolist()
        for w,window in enumerate(self.windows):
            for b,band in enumerate(self.bands):
                for i,flag in enumerate(self.flags):
                    writer.add_scalar(f'FlatTermination/{window}/{view}/{band}/{flag}_events',
                                      selected[w][b][i],step)

    def clear(self):
        self.counts.zero_()


def restored_training_frames(info, old_plan):
    """Successful inherited rollout frames; discarded trials remain separately charged."""
    value = info.get("training_transitions_total")
    if value is None:
        if info.get("population_transition"):
            raise ValueError("A changed-population parent requires its saved cumulative frame count")
        value = info["successful_updates_total"] * info["num_envs"] * old_plan["num_steps_per_env"]
    if type(value) is not int or value < 0:
        raise ValueError("Inherited successful rollout frames must be a nonnegative integer")
    return value


def validate_population_transition(info, old_plan, new_plan, new_count, revision, new_sources):
    """Authenticate a population-only migration without changing its objective or clocks."""
    changed = {key for key in set(old_plan) | set(new_plan) if old_plan.get(key) != new_plan.get(key)}
    if changed != {"allowed_num_envs", "default_num_envs"}:
        raise ValueError("Population migration can change only the two population plan fields")
    old_count = info["num_envs"]
    if (type(new_count) is not int or new_count <= 0 or new_count == old_count
            or new_plan["allowed_num_envs"] != [new_count] or new_plan["default_num_envs"] != new_count
            or old_count not in old_plan["allowed_num_envs"]):
        raise ValueError("Population migration must name the exact old and new populations")
    entry = "scripts/train_flat_base.py"
    source_changes = {name for name in set(info["source_sha256"]) | set(new_sources)
                      if info["source_sha256"].get(name) != new_sources.get(name)}
    if source_changes != {entry}:
        raise ValueError("Population migration permits only the reviewed trainer entry change")
    expected = dict(schema="flat-training-population-transition-v1", old_num_envs=old_count,
        new_num_envs=new_count, num_steps_per_env=old_plan["num_steps_per_env"],
        inherited_successful_updates=info["successful_updates_total"],
        inherited_successful_training_frames=restored_training_frames(info, old_plan),
        additional_critic_warmup_updates=0)
    if revision.get("population_transition") != expected:
        raise ValueError("Population receipt must retain the exact inherited exposure and omit new warmup")
    return expected


def probe_population_capacity(runner, env, plan, report, torch):
    """One full native rollout and all learner backward passes, with Adam writes disabled."""
    from copy import deepcopy
    import time
    def exact(left, right):
        if torch.is_tensor(left):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
        return left == right
    runner.logger.init_logging_writer()
    before = deepcopy(runner.alg.save())
    torch.cuda.reset_peak_memory_stats()
    obs = env.get_observations().to(runner.device)
    runner.alg.train_mode()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for index in range(plan["num_steps_per_env"]):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
            if (not all(torch.isfinite(value).all() for value in obs.values())
                    or not torch.isfinite(rewards).all()):
                raise RuntimeError("Capacity rollout contains a nonfinite observation or reward")
            runner.alg.process_env_step(obs, rewards, dones, extras)
            if index % 24 == 0:
                print("FLAT_CAPACITY", index, report["num_envs"], flush=True)
        runner.alg.compute_returns(obs)
    torch.cuda.synchronize()
    collection_s = time.monotonic() - started
    blocked = 0
    original_step = runner.alg.optimizer.step
    def no_optimizer_write(*args, **kwargs):
        nonlocal blocked
        blocked += 1
    runner.alg.optimizer.step = no_optimizer_write
    learning_started = time.monotonic()
    try:
        losses = runner.alg.update()
    finally:
        runner.alg.optimizer.step = original_step
    torch.cuda.synchronize()
    learning_s = time.monotonic() - learning_started
    after = runner.alg.save()
    checks = {name + "_exact": exact(before[name], after[name]) for name in
              ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
    checks["all_native_minibatch_backward_passes"] = blocked == plan["num_mini_batches"] * plan["num_learning_epochs"]
    checks["finite_native_losses"] = all(__import__("math").isfinite(value) for value in losses.values())
    report["capacity_probe"] = dict(passed=all(checks.values()), checks=checks,
        num_envs=report["num_envs"], rollout_steps=plan["num_steps_per_env"],
        native_policy_frames=report["num_envs"] * plan["num_steps_per_env"],
        optimizer_writes=0, blocked_optimizer_writes=blocked, collection_s=collection_s,
        backward_s=learning_s, policy_frames_per_collection_s=report["num_envs"] * plan["num_steps_per_env"] / collection_s,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        losses=losses, scope="Capacity only; unchanged parent weights and Adam; isolated RNG; no learning checkpoint or behavior qualification")
    writer = runner.logger.writer
    if writer:
        prefix = f"FlatCapacity/population_{report['num_envs']}"
        for name in ("collection_s", "backward_s", "policy_frames_per_collection_s", "torch_peak_allocated_bytes", "torch_peak_reserved_bytes", "optimizer_writes", "blocked_optimizer_writes"):
            writer.add_scalar(prefix + "/" + name, report["capacity_probe"][name], report["successful_updates_total"])
        writer.add_scalar(prefix + "/passed", int(all(checks.values())), report["successful_updates_total"])
    runner.logger.stop_logging_writer()
    if not all(checks.values()):
        raise RuntimeError("Capacity probe changed a model or Adam, or failed its native backward checks")


def validate_large_block_evidence(pilot, envelope, baseline, checkpoint_sha256, source_onnx_sha256,
                                  training_physics_hz=200):
    """Allow scale-up only after basic accuracy and broad skill preservation."""
    from compare_flat_base import compare
    for evidence in (pilot, envelope):
        if (evidence.get("status") != "evaluated" or evidence.get("checkpoint_sha256") != checkpoint_sha256
                or evidence.get("source_onnx_sha256") != source_onnx_sha256
                or evidence.get("physics_hz") != training_physics_hz):
            raise ValueError("Large-block evidence must evaluate this exact checkpoint at the training clock")
    basic = {"stand_230mm", "stand_305mm", "stand_430mm", "forward_05", "backward_05", "rotate_1",
             "rotate_1_reverse", "start_stop_05"}
    if (pilot.get("settings", {}).get("protocol_id") != "flat-base-paired107-fixed-v1"
            or set(pilot["result"]["cases"]) != basic or not pilot["result"]["passed"]):
        raise ValueError("The exact candidate has not passed the eight basic behavior targets")
    required = {f"{direction}_{speed}" for direction in ("forward", "backward") for speed in ("1", "2", "3")}
    required |= {"rotate_4", "rotate_4_reverse"}
    if (baseline.get("checkpoint_sha256") or baseline.get("source_onnx_sha256") != source_onnx_sha256
            or not required <= envelope["result"]["cases"].keys()):
        raise ValueError("Preservation evidence must cover the original actor's full declared speed/turn range")
    comparison = compare(baseline, envelope)
    if not comparison["no_lost_baseline_checks"]:
        raise ValueError("A broader trial lost original behavior checks")
    for name, case in envelope["result"]["cases"].items():
        vx, yaw, _ = case["command"]
        if abs(vx) >= .75 or abs(yaw) >= 1.5:
            old = baseline["result"]["cases"][name]
            if case["vx_mae_m_s"] > old["vx_mae_m_s"] + .02 or case["yaw_mae_rad_s"] > old["yaw_mae_rad_s"] + .05:
                raise ValueError(f"Broader tracking regressed in {name}; preserve skills before scale-up")
    return comparison


def main():
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle, encode, digest
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from train_chassis import preflight
    plan = read_plan(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="contracts/flat_base_finetune_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=plan["first_block_updates"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-entry-revision", type=Path,
                        help="Exact receipt for a reviewed trainer-entry fix; all environment sources stay identical")
    parser.add_argument("--resume-plan-revision", type=Path,
                        help="Exact parent/source/plan receipt for a declared mixed-curriculum migration")
    parser.add_argument("--qualified-pilot", type=Path, help="Passing same-checkpoint pilot permits a larger continuation block")
    parser.add_argument("--qualified-envelope", type=Path, help="Matched broader candidate replay for preservation")
    parser.add_argument("--baseline-envelope", type=Path, help="Original actor's identical broader replay")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    parser.add_argument("--capacity-probe", action="store_true", help="Native rollout/backward capacity only, with Adam writes disabled")
    args = parser.parse_args()
    plan = read_plan(ROOT, args.plan)
    args.num_envs = plan["default_num_envs"] if args.num_envs is None else args.num_envs
    if args.num_envs not in plan["allowed_num_envs"]:
        parser.error("Population is absent from the selected plan")
    if args.capacity_probe and (args.updates != 1 or not args.resume or args.device != "cuda:0"):
        parser.error("Capacity mode requires one isolated CUDA probe with an exact parent")
    if plan.get("training_command_reference"):
        from flat_rmcs_command_reference import read_reference_contract
        settings = plan["training_command_reference"]
        if settings["contract"] != read_reference_contract(settings["rmcs_root"]):
            raise ValueError("Training command reference differs from the inspected RMCS source")
    # Large blocks require a separately recorded successful pilot assessment.
    if not 1 <= args.updates <= plan["update_ceiling"]:
        parser.error("Invalid learning block length")
    if args.updates > plan["first_block_updates"]:
        if not all((args.resume, args.qualified_pilot, args.qualified_envelope, args.baseline_envelope)):
            parser.error("A larger block requires the candidate's passing pilot and paired full-envelope preservation evidence")
        try:
            validate_large_block_evidence(json.loads(args.qualified_pilot.read_text()),
                json.loads(args.qualified_envelope.read_text()), json.loads(args.baseline_envelope.read_text()),
                digest(args.resume), plan["source_onnx_sha256"], round(1 / plan["physics_dt"]))
        except ValueError as error:
            parser.error(str(error))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config, seed = write_bundle(ROOT, args.output / "config", args.num_envs, plan_file=args.plan)
    _, manifest = preflight(args.output / "config/contract.json")
    control = load_chassis_control(ROOT / config["control_math_source"])
    static = design_preflight(config, control, ROOT, stage="speed", role="train", num_envs=args.num_envs, seed=seed)
    report = dict(schema="flat-base-actor-transfer-learning-v1", status="starting", pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), successful_updates_total=0, actor_updates_total=0,
        attempted_updates_total=0, critic_warmup_remaining=plan["critic_warmup_updates"],
        flat_base_source_onnx_sha256=plan["source_onnx_sha256"], source_training_updates="unknown",
        initialization="exact_actor_mean_fresh_critic_std_optimizer", num_envs=args.num_envs,
        plan_file=args.plan, plan_sha256=digest(ROOT / args.plan),
        asset_manifest_sha256=config["asset_manifest_sha256"], control_math_sha256=config["control_math_sha256"],
        contract_sha256=digest(args.output / "config/contract.json"), static_preflight=static,
        requested_additional_updates=args.updates, training_release=False, hardware_deployment_ready=False)
    report["training_transitions_total"] = 0
    report["training_transition_count_scope"] = "successful inherited rollout frames; discarded optimizer consumption remains separately charged"
    report["prior_discontinued_optimizer_updates"] = plan["prior_discontinued_optimizer_updates"]
    report["consumed_optimizer_updates_total"] = plan["prior_discontinued_optimizer_updates"]
    # Freeze the exact files being executed, including uncommitted pilot code.
    sources = list((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
    sources += [Path(__file__), ROOT / "src/wheeled_algo/onnx_actor_import.py", ROOT / "src/wheeled_algo/exploration_init.py",
                ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py", ROOT / "src/wheeled_algo/flat_optimizer_rates.py"]
    if plan.get("teacher_anchor"):
        sources.append(ROOT / "src/wheeled_algo/anchored_ppo.py")
    if plan.get("critic_position_anchor"):
        sources.append(ROOT / "src/wheeled_algo/critic_input_extension.py")
    if plan.get("actor_output_layer_only"):
        sources.append(ROOT / "src/wheeled_algo/flat_actor_trainability.py")
    if plan.get("actor_adaptation"):
        sources.append(ROOT / "src/wheeled_algo/command_residual.py")
        report["actor_adaptation"] = plan["actor_adaptation"]
    if plan.get("actor_adaptation") in ("original-command-and-gated-feedback-residual-v1",
                                       "original-command-and-tracking-feedback-residual-v2",
                                       "original-command-and-separated-tracking-feedback-residual-v3",
                                       "original-command-and-continuous-low-feedback-residual-v4"):
        sources.append(ROOT / "src/wheeled_algo/guarded_feedback_residual.py")
    if plan.get("actor_adaptation") in ("original-command-and-separated-tracking-feedback-residual-v3",
                                         "original-command-and-continuous-low-feedback-residual-v4"):
        sources.append(ROOT / "src/wheeled_algo/separated_feedback.py")
    if plan.get("actor_adaptation") == "original-command-and-continuous-low-feedback-residual-v4":
        sources.append(ROOT / "src/wheeled_algo/low_demand_feedback.py")
    if plan.get("training_command_reference"):
        sources.append(ROOT / "scripts/flat_rmcs_command_reference.py")
    report["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for path in sources:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    def write_report():
        temp = args.output / "progress.tmp"
        temp.write_bytes(encode(report))
        temp.replace(args.output / "progress.json")
    write_report()
    launcher, env, runner = None, None, None
    start = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        from wheeled_tasks.chassis.env import ChassisEnv
        from wheeled_algo.onnx_actor_import import recover_actor, load_actor_mean, FORMAT
        from wheeled_algo.exploration_init import initialize_policy_std, install_axis_std_logging
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        env = ChassisEnv(config, manifest, control, ROOT, stage_name="speed", num_envs=args.num_envs,
            device=args.device, seed=seed, level=1., run_role="train")
        report["startup"] = env.startup_report
        if plan.get("training_command_reference"):
            from wheeled_tasks.chassis.flat_command_reference import install_training_reference
            report["training_command_reference"] = install_training_reference(env, plan["training_command_reference"])
        cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
        cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        cfg.seed, cfg.device, cfg.num_steps_per_env = seed, args.device, plan["num_steps_per_env"]
        cfg.save_interval = plan["checkpoint_interval"]
        for key in ("learning_rate", "num_mini_batches", "num_learning_epochs", "gamma", "lam", "entropy_coef",
                    "use_clipped_value_loss"):
            setattr(cfg.algorithm, key, plan[key])
        cfg.algorithm.schedule = "fixed"
        runner_config = cfg.to_dict()
        if plan.get("teacher_anchor"):
            runner_config["algorithm"]["class_name"] = "wheeled_algo.anchored_ppo:TeacherAnchoredPPO"
        runner_config["logger"] = "tensorboard"
        runner_config["actor"]["distribution_cfg"]["init_std"] = plan["initial_std_policy_order"][0]
        (args.output / "agent_config.json").write_bytes(encode(runner_config))
        runner = OnPolicyRunner(env, deepcopy(runner_config), log_dir=str(args.output), device=args.device)
        teacher, _ = recover_actor(ROOT / plan["actor_bundle"] / "source.onnx")
        if plan.get("teacher_anchor"):
            teacher_settings = {key: plan["teacher_anchor"][key]
                                for key in ("coefficient", "action_scales", "min_vx", "min_yaw")}
            if "min_visible_wheel_error_m_s" in plan["teacher_anchor"]:
                teacher_settings["min_visible_wheel_error_m_s"] = plan["teacher_anchor"]["min_visible_wheel_error_m_s"]
            runner.alg.configure_teacher(teacher, **teacher_settings)
            report["teacher_anchor"] = deepcopy(plan["teacher_anchor"])
            report["teacher_anchor_runtime"] = dict(
                coefficient=runner.alg.teacher_coefficient,
                action_scales=runner.alg.teacher_action_scales.detach().cpu().tolist(),
                min_vx=runner.alg.teacher_min_vx, min_yaw=runner.alg.teacher_min_yaw,
                min_visible_wheel_error_m_s=runner.alg.teacher_min_visible_wheel_error_m_s)
        feedback_kinds = ("original-command-and-gated-feedback-residual-v1",
                          "original-command-and-tracking-feedback-residual-v2",
                          "original-command-and-separated-tracking-feedback-residual-v3",
                          "original-command-and-continuous-low-feedback-residual-v4")
        separated_feedback = plan.get("actor_adaptation") in feedback_kinds[2:]
        low_demand_feedback = plan.get("actor_adaptation") == feedback_kinds[3]
        feedback_residual = plan.get("actor_adaptation") in feedback_kinds
        command_residual = plan.get("actor_adaptation") in ("original-feedback-command-residual-v1", *feedback_kinds)
        if plan.get("actor_adaptation") and not command_residual:
            raise ValueError("Unsupported actor adaptation")
        if command_residual:
            if not args.resume:
                raise ValueError("Command residual requires an explicitly authenticated parent branch")
            from wheeled_algo.command_residual import (CommandResidualActor, configure_command_residual_training,
                                                      make_command_residual_parent)
            old_actor_names = list(dict(runner.alg.actor.named_parameters()))
            runner.alg.actor.mlp = CommandResidualActor(teacher).to(args.device)
            if feedback_residual:
                from wheeled_algo.guarded_feedback_residual import (GuardedFeedbackActor, extend_feedback_parent,
                    configure_guarded_feedback_training as configure_command_residual_training)
                old_actor_names = list(dict(runner.alg.actor.named_parameters()))
                runner.alg.actor.mlp = GuardedFeedbackActor(runner.alg.actor.mlp,
                    tracking_gate=plan["actor_adaptation"] in feedback_kinds[1:]).to(args.device)
                if separated_feedback:
                    from wheeled_algo.separated_feedback import (SeparatedFeedbackActor,
                        configure_separated_feedback_training, extend_separated_feedback_parent)
                    old_actor_names = list(dict(runner.alg.actor.named_parameters()))
                    runner.alg.actor.mlp = SeparatedFeedbackActor(runner.alg.actor.mlp).to(args.device)
                    configure_command_residual_training = configure_separated_feedback_training
                    if low_demand_feedback:
                        from wheeled_algo.low_demand_feedback import (LowDemandFeedbackActor,
                            configure_low_demand_feedback_training)
                        runner.alg.actor.mlp = LowDemandFeedbackActor(runner.alg.actor.mlp).to(args.device)
                        configure_command_residual_training = configure_low_demand_feedback_training
                else:
                    from functools import partial
                    configure_command_residual_training = partial(configure_command_residual_training,
                        train_command=plan.get("feedback_training_scope") != "feedback_only")
            report["command_residual_invariant"] = runner.alg.actor.mlp.verify_original(teacher)
        # Independent rates let the fresh value network catch up without large
        # changes to the already useful actor. Both groups use native Adam/PPO.
        runner.alg.optimizer = torch.optim.Adam([
            dict(params=runner.alg.actor.parameters(), lr=plan["learning_rate"], name="actor"),
            dict(params=runner.alg.critic.parameters(), lr=plan["critic_learning_rate"], name="critic")])
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=True)
            info = checkpoint["infos"]
            for key in ("flat_base_source_onnx_sha256", "asset_manifest_sha256"):
                if info[key] != report[key]:
                    raise ValueError(f"Full-state continuation changes {key}")
            migration = info["plan_sha256"] != report["plan_sha256"]
            population_changed = info["num_envs"] != args.num_envs
            if population_changed and not migration:
                raise ValueError("A different population requires an authenticated population-only plan migration")
            if migration:
                if not args.resume_plan_revision:
                    raise ValueError("Changed training plan requires an exact migration receipt")
                revision = json.loads(args.resume_plan_revision.read_text())
                old_plan = read_plan(ROOT, revision["old_plan_file"])
                changed = sorted(key for key in set(old_plan) | set(plan) if old_plan.get(key) != plan.get(key))
                allowed = {"schema", "train_groups", "training_scope", "flat_floor_width_m", "vx_max", "yaw_max",
                           "teacher_anchor", "group_episode_seconds", "additional_critic_warmup_updates", "stationary_tracking",
                           "critic_position_anchor", "critic_dim", "actor_output_layer_only", "actor_adaptation",
                           "learning_rate", "training_command_reference", "feedback_training_scope", "command_branch_restore",
                           "physics_dt", "control_file", "control_sha256", "moving_velocity_cost_scale",
                           "allowed_num_envs", "default_num_envs"}
                if (revision.get("schema") != "flat-training-plan-revision-v1"
                        or revision["parent_checkpoint_sha256"] != digest(args.resume)
                        or revision["old_plan_sha256"] != info["plan_sha256"]
                        or digest(ROOT / revision["old_plan_file"]) != info["plan_sha256"]
                        or revision["new_plan_sha256"] != report["plan_sha256"]
                        or revision["old_source_sha256"] != info["source_sha256"]
                        or revision["new_source_sha256"] != report["source_sha256"]
                        or revision["changed_plan_fields"] != changed or set(changed) - allowed):
                    raise ValueError("Plan migration does not authenticate the exact declared change")
                report["plan_revision_sha256"] = digest(args.resume_plan_revision)
                report["changed_plan_fields"] = changed
                if population_changed:
                    report["population_transition"] = validate_population_transition(
                        info, old_plan, plan, args.num_envs, revision, report["source_sha256"])
                elif set(changed) & {"allowed_num_envs", "default_num_envs"}:
                    raise ValueError("Population plan fields require an actual population transition")
            elif info["source_sha256"] != report["source_sha256"]:
                if not args.resume_entry_revision:
                    raise ValueError("Trainer source changed; an exact entry-revision receipt is required")
                revision = json.loads(args.resume_entry_revision.read_text())
                changed = [k for k in set(info["source_sha256"]) | set(report["source_sha256"])
                           if info["source_sha256"].get(k) != report["source_sha256"].get(k)]
                entry = "scripts/train_flat_base.py"
                if (revision.get("schema") != "flat-trainer-entry-revision-v1" or changed != [entry]
                        or revision["parent_checkpoint_sha256"] != digest(args.resume)
                        or revision["old_entry_sha256"] != info["source_sha256"][entry]
                        or revision["new_entry_sha256"] != report["source_sha256"][entry]):
                    raise ValueError("Entry revision does not authenticate this exact isolated trainer change")
                report["entry_revision_sha256"] = digest(args.resume_entry_revision)
            report["training_transitions_total"] = restored_training_frames(
                info, old_plan if migration else plan)
            if info["control_math_sha256"] != report["control_math_sha256"]:
                if not migration:
                    raise ValueError("A changed control hash requires the exact clock plan migration")
                from wheeled_tasks.chassis.flat_clock_transition import validate_flat_clock_transition
                report["feedback_clock_transition"] = validate_flat_clock_transition(
                    ROOT, old_plan, plan, info["control_math_sha256"])
            elif migration and old_plan["physics_dt"] != plan["physics_dt"]:
                raise ValueError("A physics-clock change must migrate the authenticated control source")
            if migration and plan.get("command_branch_restore") and plan["command_branch_restore"] != old_plan.get("command_branch_restore"):
                from wheeled_algo.guarded_feedback_residual import restore_command_branch
                settings = plan["command_branch_restore"]
                command_path = ROOT / settings["checkpoint"]
                if (not feedback_residual or digest(args.resume) != settings["current_parent_sha256"]
                        or digest(command_path) != settings["sha256"]):
                    raise ValueError("Command branch restoration is not bound to both exact parents")
                checkpoint, proof = restore_command_branch(checkpoint,
                    torch.load(command_path, map_location=args.device, weights_only=True),
                    list(dict(runner.alg.actor.named_parameters())))
                report["command_branch_restore"] = {**proof, **settings}
                info = checkpoint["infos"]
            if plan.get("critic_position_anchor") and checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1] == 81:
                if not migration:
                    raise ValueError("Critic identity extension must be declared in a plan migration")
                from wheeled_algo.critic_input_extension import extend_flat_critic
                checkpoint = extend_flat_critic(checkpoint)
                report["critic_input_extension"] = dict(old_dim=81, new_dim=83, old_columns_and_moments="exact",
                    new_columns_and_moments="zero", actor_and_actor_Adam="exact", value_function="identity_extension")
            if separated_feedback and info.get("actor_adaptation") not in feedback_kinds[1:]:
                raise ValueError("Separated feedback must extend the tracking parent or resume its exact own state")
            if separated_feedback and info.get("actor_adaptation") == feedback_kinds[1]:
                if not migration:
                    raise ValueError("Separated feedback requires a declared identity-extension plan migration")
                checkpoint, report["separated_feedback_identity_extension"] = extend_separated_feedback_parent(
                    checkpoint, runner.alg.actor, runner.alg.critic, runner.alg.optimizer, old_actor_names)
                report["active_branch_actor_updates"] = info["active_branch_actor_updates"]
                report["feedback_branch_actor_updates"] = info["feedback_branch_actor_updates"]
                report["post_split_actor_updates"] = 0
                report["inherited_shared_feedback_updates"] = info["feedback_branch_actor_updates"]
            elif low_demand_feedback and info.get("actor_adaptation") == feedback_kinds[2]:
                if not migration:
                    raise ValueError("Low-demand gate requires a declared forward plan migration")
                report["feedback_gate_revision"] = dict(before=info["actor_adaptation"],
                    after=plan["actor_adaptation"], initial_mean_is_identity=False,
                    weights_Adam_std_critic_RNG_consumption="parent_exact",
                    gate_contract=runner.alg.actor.mlp.verify_original(teacher)["low_demand_gate"])
                for key in ("active_branch_actor_updates", "feedback_branch_actor_updates",
                            "post_split_actor_updates", "inherited_shared_feedback_updates"):
                    report[key] = info[key]
            elif (feedback_residual and info.get("actor_adaptation") in feedback_kinds
                    and info["actor_adaptation"] != plan["actor_adaptation"]):
                if (not migration or info["actor_adaptation"] != feedback_kinds[0]
                        or plan["actor_adaptation"] != feedback_kinds[1]):
                    raise ValueError("Feedback gate change requires an explicit forward plan migration")
                # Same tensors, parameter order and Adam ownership; the gate is
                # a declared function change, not a zero-branch identity extension.
                report["feedback_gate_revision"] = dict(before=info["actor_adaptation"],
                    after=plan["actor_adaptation"], initial_mean_is_identity=False,
                    weights_Adam_std_critic_RNG_consumption="parent_exact",
                    gate_contract=runner.alg.actor.mlp.verify_original(teacher)["feedback_zero_gate"])
                report["active_branch_actor_updates"] = info["active_branch_actor_updates"]
                report["feedback_branch_actor_updates"] = info["feedback_branch_actor_updates"]
            elif feedback_residual and info.get("actor_adaptation") != plan["actor_adaptation"]:
                if not migration:
                    raise ValueError("A feedback identity extension requires a declared plan migration")
                checkpoint, report["feedback_identity_extension"] = extend_feedback_parent(
                    checkpoint, runner.alg.actor, runner.alg.critic, runner.alg.optimizer, old_actor_names)
                report["active_branch_actor_updates"] = info["active_branch_actor_updates"]
                report["feedback_branch_actor_updates"] = 0
            elif command_residual and info.get("actor_adaptation") != plan["actor_adaptation"]:
                if not migration:
                    raise ValueError("A new residual branch requires a declared plan migration")
                checkpoint, report["command_residual_branch"] = make_command_residual_parent(
                    checkpoint, runner.alg.actor, runner.alg.critic, runner.alg.optimizer, old_actor_names)
                report["active_branch_actor_updates"] = 0
            elif command_residual:
                report["active_branch_actor_updates"] = info["active_branch_actor_updates"]
                if feedback_residual:
                    report["feedback_branch_actor_updates"] = info["feedback_branch_actor_updates"]
                if separated_feedback:
                    report["post_split_actor_updates"] = info["post_split_actor_updates"]
                    report["inherited_shared_feedback_updates"] = info["inherited_shared_feedback_updates"]
            runner.alg.load(checkpoint, load_cfg=None, strict=True)
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["cuda_rng_state"]])
            env.generator.set_state(checkpoint["env_generator_state"].cpu())
            def exact(a, b):
                if torch.is_tensor(a):
                    return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
                if isinstance(a, dict):
                    return a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
                if isinstance(a, (list, tuple)):
                    return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
                return a == b
            loaded = runner.alg.save()
            restored = {key: exact(checkpoint[key], loaded[key])
                        for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
            restored.update(cpu_sampling_rng=exact(checkpoint["torch_rng_state"], torch.get_rng_state()),
                cuda_sampling_rng=exact(checkpoint["cuda_rng_state"], torch.cuda.get_rng_state_all()),
                environment_rng_before_new_reset=exact(checkpoint["env_generator_state"], env.generator.get_state()))
            report["full_restore_verification"] = restored
            report["restore_verification_reference"] = "zero_extended_parent" if report.get("critic_input_extension") else "parent"
            if not all(restored.values()):
                raise RuntimeError("Continuation failed exact full-state restoration before optimization")
            for key in ("successful_updates_total", "actor_updates_total", "attempted_updates_total", "critic_warmup_remaining"):
                report[key] = info[key]
            if migration and not population_changed:
                report["critic_warmup_remaining"] += plan.get("additional_critic_warmup_updates", 0)
            report.update(initialization="full_actor_critic_std_adam_restore_with_declared_plan_migration" if migration else
                          "full_same_plan_actor_critic_std_adam_restore", parent_checkpoint=str(args.resume.resolve()),
                          parent_checkpoint_sha256=digest(args.resume),
                          resumption_semantics="exact_models_Adam_and_saved_RNG; fresh_environment_reset_from_resumed_generator")
            if report.get("critic_input_extension"):
                report.update(initialization="actor_and_Adam_exact_critic_identity_extension_with_RNG_restore",
                    resumption_semantics="actor/std/Adam exact; critic/Adam zero-extension; saved RNG; fresh physical reset")
            if command_residual:
                runner.alg.actor.mlp.verify_original(teacher)
            if report.get("command_residual_branch"):
                report.update(initialization="declared_new_branch_original_feedback_zero_residual_parent_critic_std",
                    restore_verification_reference="declared_transformed_parent_not_the_previous_actor",
                    resumption_semantics="original ONNX feedback reset; new zero residual/Adam; exact parent critic/std/Adam/RNG; all consumption retained")
            if report.get("feedback_identity_extension"):
                report.update(initialization="identity_extension_of_parent_mean_with_zero_feedback_branch",
                    restore_verification_reference="identity_extended_parent_with_retained_existing_Adam",
                    resumption_semantics="parent mean/std/critic/Adam exact; new zero feedback branch/empty Adam; saved RNG; all consumption retained")
            if report.get("command_branch_restore"):
                report.update(initialization="declared_command_branch_composition_with_current_feedback_parent",
                    restore_verification_reference="declared_command_composition_not_the_unchanged_parent_actor",
                    resumption_semantics="declared older command mean/Adam; current feedback/critic/std/other Adam/RNG/lifetime consumption; fresh physical reset")
            if report.get("separated_feedback_identity_extension"):
                report.update(initialization="exact_parent_mean_with_cloned_low_and_fast_feedback_weights_and_Adam",
                    restore_verification_reference="declared_identity_extended_parent_with_feedback_and_Adam_clones",
                    resumption_semantics="parent mean/std/critic/Adam/RNG/counters retained; new fast feedback clones parent shared feedback including Adam step; fresh physical reset")
            if report.get("feedback_gate_revision"):
                report.update(initialization="exact_parent_state_with_declared_feedback_gate_function_change",
                    resumption_semantics="parent weights/Adam/std/critic/RNG/counters exact; moving mean changes by declared gate; fresh physical reset")
            env.stage_actor_update = report["actor_updates_total"]
            env.global_actor_update = report["actor_updates_total"]
            env.training_transitions = report["training_transitions_total"]
            env.reset(torch.arange(args.num_envs, device=args.device))
        else:
            report["actor_import"] = load_actor_mean(runner.alg.actor, dict(format=FORMAT,
                actor_mean_state_dict=teacher.state_dict(), source_sha256=plan["source_onnx_sha256"]))
            report["initial_std_policy_order"] = initialize_policy_std(runner.alg, plan["initial_std_policy_order"])
        from wheeled_algo.flat_optimizer_rates import apply_flat_optimizer_rates
        parent_rates = None
        if (args.resume and migration and "learning_rate" in report["changed_plan_fields"]
                and not report.get("command_residual_branch")):
            # A new command branch already installs the explicitly declared new
            # actor group. Ordinary full-state migrations first restore old Adam.
            parent_rates = (old_plan["learning_rate"], old_plan["critic_learning_rate"])
        report["optimizer_rate_contract"] = apply_flat_optimizer_rates(runner.alg.optimizer,
            plan["learning_rate"], plan["critic_learning_rate"], parent_rates=parent_rates)
        runner.alg.learning_rate = plan["learning_rate"]
        if args.resume:
            expected_optimizer = deepcopy(checkpoint["optimizer_state_dict"])
            expected_optimizer["param_groups"][0]["lr"] = plan["learning_rate"]
            report["optimizer_after_declared_rate_exact"] = exact(
                runner.alg.optimizer.state_dict(), expected_optimizer)
            report["optimizer_rate_application_order"] = "full_parent_restore_verified_then_declared_actor_rate"
            if not report["optimizer_after_declared_rate_exact"]:
                raise RuntimeError("Rate migration changed parent Adam beyond its declared actor learning rate")
        initial_count = report["successful_updates_total"]
        checkpoint_milestones = {initial_count + 1, initial_count + report["critic_warmup_remaining"],
                                 initial_count + report["critic_warmup_remaining"] + 1}
        report["consumed_optimizer_updates_total"] = plan["prior_discontinued_optimizer_updates"] + initial_count
        if plan["prior_discontinued_optimizer_updates"] + initial_count + args.updates > plan["update_ceiling"]:
            raise ValueError("Actor-transfer learning consumption exceeds its explicit ceiling")
        runner.current_learning_iteration = initial_count
        install_axis_std_logging(runner.logger)
        frozen_actor = {k: v.detach().clone() for k, v in runner.alg.actor.state_dict().items()}
        output_layer_only = bool(plan.get("actor_output_layer_only"))
        if output_layer_only:
            from wheeled_algo.flat_actor_trainability import configure_flat_actor_training, OUTPUT_LAYER
            report["actor_trainability"] = configure_flat_actor_training(runner.alg.actor,
                enabled=report["critic_warmup_remaining"] == 0, output_layer_only=True)
            report["frozen_actor_state_names"] = sorted(frozen_actor.keys() - OUTPUT_LAYER)
        if command_residual:
            report["actor_trainability"] = configure_command_residual_training(runner.alg.actor,
                enabled=report["critic_warmup_remaining"] == 0)
            report["frozen_actor_state_names"] = report["actor_trainability"]["frozen_state_names"]
        original_update = runner.alg.update
        original_step = env.step
        behavior_sums = torch.zeros(args.num_envs, 5, dtype=torch.float64, device=args.device)
        behavior_frames = torch.zeros(args.num_envs, dtype=torch.int64, device=args.device)
        birth_sums = torch.zeros_like(behavior_sums)
        birth_frames = torch.zeros_like(behavior_frames)
        height_metrics = FlatHeightCommandMetrics(args.num_envs, args.device)
        report["height_command_metrics_contract"] = height_metrics.contract()
        termination_metrics = FlatTerminationMetrics(args.num_envs, args.device)
        report["termination_metrics_contract"] = termination_metrics.contract()

        def step(actions):
            result = original_step(actions)
            data = result[3]["diagnostics"]
            valid = data["episode_ticks"] > round(plan["acceptance"]["warmup_seconds"] / env.policy_dt)
            values = torch.stack(((data["velocity"][:, 0] - data["commands"][:, 0]).abs(),
                (data["omega"][:, 2] - data["commands"][:, 1]).abs(),
                (data["height"] - data["commands"][:, 2]).abs(), data["planar_speed_world"],
                data["terminated"].float()), -1)
            behavior_sums.add_(values.double() * valid[:, None])
            behavior_frames.add_(valid)
            birth_valid = (data["episode_ticks"] > 0) & (data["episode_ticks"] <= round(1. / env.policy_dt))
            birth_sums.add_(values.double() * birth_valid[:, None])
            birth_frames.add_(birth_valid)
            height_metrics.observe(data["commands"][:, 2], values, valid, birth_valid)
            termination_metrics.observe(data, valid, birth_valid)
            return result
        env.step = step

        def save_checkpoint(path):
            state = runner.alg.save()
            state.update(iter=report["successful_updates_total"], infos=deepcopy(report),
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all(),
                env_generator_state=env.generator.get_state())
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            return path

        def update():
            enabled = report["critic_warmup_remaining"] == 0
            if command_residual:
                report["actor_trainability"] = configure_command_residual_training(runner.alg.actor, enabled)
            elif output_layer_only:
                report["actor_trainability"] = configure_flat_actor_training(runner.alg.actor,
                    enabled=enabled, output_layer_only=True)
            else:
                runner.alg.actor.requires_grad_(enabled)
            report["attempted_updates_total"] += 1
            report["optimizer_phase"] = "actor_and_critic" if enabled else "critic_warmup"
            write_report()
            result = original_update()
            # Charge a returned optimizer update before any invariant check.
            report["successful_updates_total"] += 1
            report["training_transitions_total"] += args.num_envs * plan["num_steps_per_env"]
            report["consumed_optimizer_updates_total"] = plan["prior_discontinued_optimizer_updates"] + report["successful_updates_total"]
            report["actor_updates_total"] += int(enabled)
            if command_residual and plan.get("feedback_training_scope") != "feedback_only":
                report["active_branch_actor_updates"] += int(enabled)
            if feedback_residual:
                report["feedback_branch_actor_updates"] += int(enabled)
            if separated_feedback:
                report["post_split_actor_updates"] += int(enabled)
            report["critic_warmup_remaining"] = max(0, report["critic_warmup_remaining"] - int(not enabled))
            report["last_losses"] = result
            if not all(torch.isfinite(v).all().item() for model in (runner.alg.actor, runner.alg.critic)
                       for v in model.state_dict().values()):
                raise RuntimeError("Nonfinite learned parameters; checkpoint cannot be finalized")
            if not enabled and any(not torch.equal(v, frozen_actor[k]) for k, v in runner.alg.actor.state_dict().items()):
                raise RuntimeError("Frozen actor or exploration std changed during critic warmup")
            if (output_layer_only or command_residual) and any(not torch.equal(runner.alg.actor.state_dict()[k], frozen_actor[k])
                                         for k in report["frozen_actor_state_names"]):
                raise RuntimeError("Frozen actor features/buffers or exploration std changed during scoped continuation")
            report["actor_mean_max_change_from_block_start"] = max(float((v - frozen_actor[k]).abs().max())
                for k, v in runner.alg.actor.state_dict().items() if k.startswith("mlp."))
            report["elapsed_s"] = time.monotonic() - start
            report["status"] = "training"
            write_report()
            env.stage_actor_update = report["actor_updates_total"]
            env.global_actor_update = report["actor_updates_total"]
            env.training_transitions = report["training_transitions_total"]
            writer = runner.logger.writer
            if writer:
                writer.add_scalar("FlatTransfer/training_transitions_total", report["training_transitions_total"], report["successful_updates_total"])
                writer.add_scalar("FlatTransfer/actor_updates_total", report["actor_updates_total"], report["successful_updates_total"])
                if command_residual:
                    writer.add_scalar("FlatTransfer/active_branch_actor_updates", report["active_branch_actor_updates"], report["successful_updates_total"])
                if feedback_residual:
                    writer.add_scalar("FlatTransfer/feedback_branch_actor_updates", report["feedback_branch_actor_updates"], report["successful_updates_total"])
                if separated_feedback:
                    writer.add_scalar("FlatTransfer/post_split_actor_updates", report["post_split_actor_updates"], report["successful_updates_total"])
                writer.add_scalar("FlatTransfer/actor_mean_max_change_from_block_start", report["actor_mean_max_change_from_block_start"], report["successful_updates_total"])
                if hasattr(env, "diagnostic_reward_stats"):
                    for name, value in env.diagnostic_reward_metrics().items():
                        writer.add_scalar(name, float(value), report["successful_updates_total"])
                for name in dict.fromkeys(env.scene_groups):
                    selected = torch.tensor([g == name for g in env.scene_groups], device=args.device)
                    views = [(name, selected)]
                    if hasattr(env, "rmcs_direct_lanes"):
                        views += [(name + "/rmcs_direct", selected & env.rmcs_direct_lanes),
                                  (name + "/legacy_ramp", selected & ~env.rmcs_direct_lanes)]
                    for view, mask in views:
                        if not mask.any():
                            continue
                        for namespace, sums, counts in (("FlatBehavior", behavior_sums, behavior_frames),
                                                         ("FlatBirth", birth_sums, birth_frames)):
                            frames = counts[mask].sum()
                            writer.add_scalar(f"{namespace}/{view}/sample_frames", int(frames), report["successful_updates_total"])
                            if not frames:
                                continue  # An empty window is not perfect tracking.
                            means = sums[mask].sum(0) / frames
                            for tag, value in zip(("vx_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "planar_speed_mean_m_s", "terminated_frame_fraction"), means.cpu().tolist()):
                                writer.add_scalar(f"{namespace}/{view}/{tag}", value, report["successful_updates_total"])
                        height_metrics.log(writer, view, mask, report["successful_updates_total"])
                        termination_metrics.log(writer, view, mask, report["successful_updates_total"])
            height_metrics.clear()
            termination_metrics.clear()
            behavior_sums.zero_()
            behavior_frames.zero_()
            birth_sums.zero_()
            birth_frames.zero_()
            write_report()
            if report["successful_updates_total"] in checkpoint_milestones:
                save_checkpoint(args.output / f"model_{report['successful_updates_total']:06d}.pt")
            return result

        def save(path, infos=None):
            return save_checkpoint(args.output / f"model_{report['successful_updates_total']:06d}.pt")
        if args.capacity_probe:
            # Call the original learner, with no reporting/update wrapper installed.
            save_checkpoint(args.output / "initial_state.pt")
            probe_population_capacity(runner, env, plan, report, torch)
            report.update(status="capacity_complete", elapsed_s=time.monotonic() - start,
                requested_additional_updates=0, new_optimizer_updates=0)
            print("FLAT_CAPACITY_COMPLETE", json.dumps(report["capacity_probe"]), flush=True)
        else:
            runner.alg.update = update
            runner.save = save
            save_checkpoint(args.output / "initial_state.pt")
            runner.learn(args.updates, init_at_random_ep_len=False)
            report.update(status="block_complete", elapsed_s=time.monotonic() - start)
            path = save_checkpoint(args.output / f"model_{report['successful_updates_total']:06d}.pt")
            report.update(final_checkpoint=str(path), final_checkpoint_sha256=digest(path))
            print("FLAT_TRAIN_COMPLETE", json.dumps({k: report[k] for k in ("successful_updates_total", "actor_updates_total", "elapsed_s", "final_checkpoint")}), flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    return 0 if report["status"] in ("block_complete", "capacity_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
