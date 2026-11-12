#!/usr/bin/env python3
"""Read-only native wiring probe for the balanced WheelBipe jump curriculum.

Loads an actor for inference only. No optimizer or PPO runner is constructed.
Native behavior and full acceptance qualification are separate from wiring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=119)
    p.add_argument("--course-update", type=int, default=20000)
    p.add_argument("--motion-update-offset", type=int, default=0)
    p.add_argument("--randomization-profile", default="nominal")
    p.add_argument("--randomization-strength", type=float, default=0.)
    p.add_argument("--training-profile", choices=("foundation-jump-v1", "foundation-jump-v2","foundation-jump-v3","foundation-jump-v4","foundation-jump-v5","foundation-jump-v6","foundation-jump-v7","foundation-jump-v8"), default="foundation-jump-v1")
    p.add_argument("--leg-kp",type=float,default=None)
    p.add_argument("--leg-kd",type=float,default=None)
    p.add_argument("--leg-target-overtravel-rad",type=float,default=None)
    p.add_argument("--leg-target-overtravel-scope",choices=("all","jump-request"),default=None)
    p.add_argument("--tuck-spring-compensation",choices=("off","model"),default=None)
    p.add_argument("--tuck-spring-gain",type=float,default=None)
    p.add_argument("--tuck-spring-cap-nm",type=float,default=None)
    p.add_argument("--course-updates",type=int,default=None)
    p.add_argument("--spec-override",type=json.loads,default=None,
        help="Read-only scenario: JSON deep-merged into the mixed course (e.g. fixed platform height/speed)")
    a = p.parse_args()
    if a.steps < 300 or a.output.exists():
        p.error("Use a new output directory and at least six simulated seconds")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"src"))
    from wheeled_tasks.wheelbipe_port.model import expose_source
    expose_source()
    a.output.mkdir(parents=True)
    pinned = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    receipt = dict(status="preparing", ppo_updates=0, optimizer_created=False,
        checkpoint_sha256=pinned, hardware_io=False, behavior_qualified=False,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    def save():
        (a.output/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    save()
    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False)).app
    env = writer = None
    started = time.monotonic()
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.wheelbipe_port.config import WheelbipePortFlatCfg
        from wheeled_tasks.wheelbipe_port.mixed_environment import WheelbipeMixedEnv
        from wheeled_tasks.wheelbipe_port.mixed_curriculum import quota_counts, rollout_metrics, validate
        from wheeled_tasks.wheelbipe_port.training_profile import configure_profile, apply_phase
        torch.set_num_threads(4)
        cfg = WheelbipePortFlatCfg()
        cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
        cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .005, 4, 4
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = 64
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 32
        cfg.robot_cfg.actuators["legs_act"].armature = .015795
        cfg.port_leg_kp, cfg.port_leg_kd = 60., 2.
        cfg.port_randomization_profile = a.randomization_profile
        cfg.port_randomization_strength = a.randomization_strength
        cfg.port_record_peaks = True
        configure_profile(cfg, a.training_profile, course_updates=a.course_updates)
        from wheeled_tasks.wheelbipe_port.training_profile import apply_control_overrides
        apply_control_overrides(cfg,kp=a.leg_kp,kd=a.leg_kd,
            overtravel=a.leg_target_overtravel_rad,
            overtravel_scope=a.leg_target_overtravel_scope,tuck_mode=a.tuck_spring_compensation,
            tuck_gain=a.tuck_spring_gain,tuck_cap_nm=a.tuck_spring_cap_nm)
        if a.spec_override:
            def merge(target, patch):
                for key, value in patch.items():
                    if isinstance(value, dict) and isinstance(target.get(key), dict):
                        merge(target[key], value)
                    else:
                        target[key] = value
            merge(cfg.port_mixed_spec, a.spec_override)
            validate(cfg.port_mixed_spec)
            receipt["scenario_override"] = a.spec_override
        if cfg.port_real2sim is not None:
            limit = cfg.port_real2sim["leg_friction"]["native_effort_limit_nm"]
            cfg.robot_cfg.actuators["legs_act"].effort_limit = limit
            cfg.robot_cfg.actuators["legs_act"].effort_limit_sim = limit
        cfg.port_start_completed_updates = a.course_update
        cfg.port_motion_update_offset = a.motion_update_offset
        apply_phase(cfg, a.course_update+a.motion_update_offset)
        # Controlled standing births isolate command/reward wiring from the
        # original independent multi-pose recovery curriculum.
        cfg.use_leg_random_start = False
        receipt["resolved_port_control"] = dict(kp=cfg.port_leg_kp,kd=cfg.port_leg_kd,
            virtual_pair_overtravel_rad=cfg.port_leg_target_overtravel_rad,
            virtual_pair_overtravel_scope=cfg.port_leg_target_overtravel_scope,
            tuck_spring_compensation=dict(cfg.port_tuck_spring_compensation))
        env = WheelbipeMixedEnv(cfg)
        state = torch.load(a.checkpoint, map_location=env.device, weights_only=False)["model_state_dict"]
        layers = []
        for i, (ni, no) in enumerate(((35, 256), (256, 128), (128, 64), (64, 6))):
            layer = torch.nn.Linear(ni, no, device=env.device)
            layer.weight.data.copy_(state[f"actor.{i*2}.weight"])
            layer.bias.data.copy_(state[f"actor.{i*2}.bias"])
            layers.append(layer)
            if i < 3:
                layers.append(torch.nn.ELU())
        actor = torch.nn.Sequential(*layers).eval()
        writer = SummaryWriter(str(a.output/"events"))
        expected = quota_counts(a.num_envs, cfg.port_mixed_spec)
        counts = {name: int(env.task_schedule.mask(name).sum()) for name in env.task_schedule.tasks}
        if counts != expected:
            raise ValueError("Native task allocation differs from its contract")
        receipt.update(status="sampling", allocated_envs=counts, expected_envs=expected,
            policy_hz=1/env.step_dt, physics_hz=1/env.physics_dt,
            domain=env.domain_receipt, source_profile_sha256=cfg.port_profile_sha256)
        save()
        sums = {name: dict(samples=0, reward=0., reward_square=0., resets=0) for name in env.task_schedule.tasks}
        metric_sums = {name: dict(vx_error_m_s=0., yaw_error_rad_s=0., height_error_m=0., tilt_deg=0.)
                       for name in env.task_schedule.tasks}
        event_counts = {name: 0 for name in ("liftoff_event", "landing_event", "success_event", "timeout_event")}
        events_by_task = {task: dict(event_counts) for task in ("jump", "airborne")}
        outcomes = {task: dict(completed=0, flown=0, landed=0, succeeded=0, terminated=0, timeout=0)
                    for task in ("jump", "airborne")}
        old_rewards = env._get_rewards
        def observe_payment():
            reward = old_rewards()
            for name in event_counts:
                event_counts[name] += int(getattr(env.jump_observer, name).sum())
                for task in events_by_task:
                    events_by_task[task][name] += int((getattr(env.jump_observer, name)
                        & env.task_schedule.mask(task)).sum())
            return reward
        env._get_rewards = observe_payment
        platform_counts = {}
        if env.platform_observer is not None:
            from wheeled_tasks.wheelbipe_port.mixed_curriculum import OVER
            observer, schedule = env.platform_observer, env.task_schedule
            names = (*observer.EVENTS, "top_success_event")
            platform_counts = {name: 0 for name in names}
            payment_with_jumps = env._get_rewards
            def observe_platform():
                reward = payment_with_jumps()
                over = schedule.mask("platform") & (schedule.mode == OVER)
                for name in observer.EVENTS:
                    platform_counts[name] += int(getattr(observer, name).sum())
                platform_counts["top_success_event"] += int((env.jump_observer.success_event & over).sum())
                return reward
            env._get_rewards = observe_platform
        old_dones = env._get_dones
        def observe_outcomes():
            terminate, timeout = old_dones()
            for task, row in outcomes.items():
                mask = env.task_schedule.mask(task) & (terminate | timeout)
                row["completed"] += int(mask.sum())
                for key in ("flown", "landed", "succeeded"):
                    row[key] += int((mask & getattr(env.jump_observer, key)).sum())
                row["terminated"] += int((mask & terminate).sum())
                row["timeout"] += int((mask & timeout).sum())
            return terminate, timeout
        env._get_dones = observe_outcomes
        with torch.inference_mode():
            obs, _ = env.reset()
            receipt["observation_shapes"] = {k: list(v.shape) for k, v in obs.items()}
            if obs["policy"].shape != (a.num_envs, 35) or obs["critic"].shape != (a.num_envs, 78):
                raise ValueError("Actor/critic ABI changed")
            rollout_rewards = []
            for step in range(a.steps):
                action = actor(obs["policy"])
                obs, reward, terminate, timeout, extras = env.step(action)
                tensors = [*obs.values(), reward, env.robot.data.root_state_w, env.robot.data.joint_pos,
                           env.robot.data.joint_vel, env.motor_effort]
                if not all(torch.isfinite(t).all() for t in tensors):
                    raise ValueError("Nonfinite native state or payment")
                done = terminate | timeout
                measured = dict(vx_error_m_s=(env.robot.data.root_lin_vel_b[:, 0]-env.command[:, 0]).abs(),
                    yaw_error_rad_s=(env.robot.data.root_ang_vel_b[:, 2]-env.command[:, 2]).abs(),
                    height_error_m=(env.robot.data.root_pos_w[:, 2]-env.terrain.env_origins[:, 2]-env.height_cmd).abs(),
                    tilt_deg=torch.acos((-env.robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.))*180./torch.pi)
                for name in env.task_schedule.tasks:
                    mask = env.task_schedule.mask(name)
                    selected = reward[mask]
                    sums[name]["samples"] += int(mask.sum())
                    sums[name]["reward"] += float(selected.sum())
                    sums[name]["reward_square"] += float(selected.square().sum())
                    sums[name]["resets"] += int(done[mask].sum())
                    for key, value in measured.items():
                        metric_sums[name][key] += float(value[mask].sum())
                for tag, value in extras.get("log", {}).items():
                    if tag.startswith(("Mixed/", "Port/", "JumpSequence/")):
                        writer.add_scalar(tag, float(value), step+1)
                rollout_rewards.append(reward.detach().clone())
                if (step+1) % 24 == 0:
                    sample = torch.stack(rollout_rewards).unsqueeze(-1)
                    # Exercise logging shapes without inventing measured PPO
                    # advantages: only samples/rewards are emitted here.
                    for tag, value in rollout_metrics(env.task_schedule.role, sample, sample, sample,
                                                      env.task_schedule.tasks).items():
                        if tag.endswith("/samples") or tag.endswith("/reward_rms"):
                            writer.add_scalar("Probe/"+tag, value, (step+1)//24)
                    rollout_rewards.clear()
                if (step+1) % 100 == 0:
                    print(json.dumps(dict(step=step+1, events=event_counts)), flush=True)
        for name, row in sums.items():
            row["reward_mean_per_step"] = row.pop("reward")/row["samples"]
            row["reward_rms_per_step"] = (row.pop("reward_square")/row["samples"])**.5
            row["actual_sample_fraction"] = row["samples"]/(a.num_envs*a.steps)
            row["tracking_mean_absolute"] = {key: value/row["samples"]
                                             for key, value in metric_sums[name].items()}
        if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != pinned:
            raise ValueError("Read-only actor checkpoint changed during probe")
        receipt.update(status="complete", elapsed_s=time.monotonic()-started,
            simulated_seconds_per_lane=a.steps*env.step_dt, tasks=sums, physical_events=event_counts,
            physical_events_by_task=events_by_task, episode_outcomes=outcomes,
            platform_events=platform_counts,
            max_constraint_gap_policy_sample_m=float(env.port_constraint_gap_peak),
            finite_native_states=True, interpretation="Read-only physical outcomes for the pinned actor; full capability and hardware qualification remain separate")
        if env.jump_observer.sequence is not None:
            receipt["jump_sequence_metrics"] = {tag: float(value)
                for tag, value in env.jump_observer.sequence.metrics().items()}
        save()
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc(), elapsed_s=time.monotonic()-started)
        save()
        raise
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
