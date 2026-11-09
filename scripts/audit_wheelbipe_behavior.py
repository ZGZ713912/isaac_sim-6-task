#!/usr/bin/env python3
"""Read-only command, action, physical effort and reward-payment comparison.

Every trace uses TensorBoard. JSON contains bounded window summaries, never a
second training-curve database. No optimizer, donor policy, or hardware IO.
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
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--vehicle", choices=("port", "reference"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--asset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--training-profile",choices=("legacy","foundation-v1","foundation-v2","foundation-jump-v1","foundation-jump-v2","foundation-jump-v3","foundation-jump-v4"),default="legacy")
    p.add_argument("--control", choices=("actor", "wheel-pulse"), default="actor")
    p.add_argument("--gas-feedforward", choices=("source", "disabled"), default="source",
        help="Isolated port-only control intervention; keeps physical gas springs unchanged")
    p.add_argument("--cases", nargs="+", default=["stand_low", "stand_mid", "stand_high", "forward", "reverse"])
    p.add_argument("--leg-kp",type=float,default=None)
    p.add_argument("--leg-kd",type=float,default=None)
    p.add_argument("--leg-target-overtravel-rad",type=float,default=None)
    p.add_argument("--leg-target-overtravel-scope",choices=("all","jump-request"),default=None)
    p.add_argument("--tuck-spring-compensation",choices=("off","model"),default=None)
    p.add_argument("--tuck-spring-gain",type=float,default=None)
    p.add_argument("--tuck-spring-cap-nm",type=float,default=None)
    p.add_argument("--course-updates",type=int,default=None)
    a = p.parse_args()
    if a.output.exists() or a.steps < 20 or a.num_envs < 1:
        p.error("Require a new output directory and a positive sampling budget")
    if a.control == "actor" and a.checkpoint is None:
        p.error("Actor inference requires --checkpoint")
    if a.vehicle == "reference" and a.asset is None:
        p.error("Reference evaluation requires its original --asset")
    a.output.mkdir(parents=True)
    root = a.source_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from wheeled_tasks.wheelbipe_port.model import expose_source
    expose_source()
    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False)).app
    env = writer = None
    started = time.monotonic()
    receipt = dict(status="preparing", ppo_updates=0, hardware_io=False, behavior_qualified=False,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        gap_scope="policy observation boundaries; no hard distance limit",
        reference_scope="August policy/asset under vendored V14 code; not exact historical replay")
    def save():
        target = a.output / "receipt.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from isaaclab.utils.math import quat_apply
        from pxr import UsdPhysics
        from wheeled_tasks.wheelbipe_port.environment import WheelbipePortEnv, OfflineTerrainImporter
        from wheeled_tasks.wheelbipe_port.config import WheelbipePortFlatCfg
        from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import WheelbipeV14Env
        from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import WheelbipeV14FlatEnvCfg
        torch.set_num_threads(4)
        class ReferenceEnv(WheelbipeV14Env):
            def _on_command_updated(self):
                super()._on_command_updated()
                if hasattr(self, "eval_command"):
                    vx, wz, h = self.eval_command
                    self.command[:, 0], self.command[:, 1], self.command[:, 2] = vx, 0., wz
                    self.height_cmd.fill_(h)
        port = a.vehicle == "port"
        cfg = (WheelbipePortFlatCfg if port else WheelbipeV14FlatEnvCfg)()
        cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
        cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .005, 4, 4
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = 64
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 32
        cfg.sim.physx.enable_external_forces_every_iteration = False
        cfg.use_leg_random_start = False
        cfg.commands.special_modes = {}
        cfg.events.reset_base.params["pose_range"] = {}
        cfg.events.reset_base.params["velocity_range"] = {}
        if port:
            from wheeled_tasks.wheelbipe_port.training_profile import configure_profile
            cfg.port_leg_kp, cfg.port_leg_kd = 60., 2.
            configure_profile(cfg, a.training_profile, course_updates=a.course_updates)
            from wheeled_tasks.wheelbipe_port.training_profile import apply_control_overrides
            apply_control_overrides(cfg,kp=a.leg_kp,kd=a.leg_kd,
                overtravel=a.leg_target_overtravel_rad,
                overtravel_scope=a.leg_target_overtravel_scope,tuck_mode=a.tuck_spring_compensation,
                tuck_gain=a.tuck_spring_gain,tuck_cap_nm=a.tuck_spring_cap_nm)
            cfg.use_leg_random_start = False
            cfg.robot_cfg.actuators["legs_act"].armature = .015795
            cfg.port_randomization_profile, cfg.port_randomization_strength = "nominal", 0.
            cfg.port_record_peaks = True
        else:
            cfg.robot_cfg.spawn.usd_path = str(a.asset.resolve())
            cfg.terrain.class_type = OfflineTerrainImporter
            cfg.use_self_obs_noise, cfg.use_self_act_noise = False, False
            cfg.use_obs_delay, cfg.use_act_delay = False, False
            cfg.spring_settings["random_force"] = None
            cfg.play_wheel_material_debug = cfg.play_wheel_motor_z_axis_align_debug = False
            for name, value in vars(cfg.events).items():
                if hasattr(value, "mode") and name != "reset_base":
                    setattr(cfg.events, name, None)
            receipt["asset_sha256"] = hashlib.sha256(a.asset.read_bytes()).hexdigest()
        if port:
            receipt["resolved_port_control"] = dict(kp=cfg.port_leg_kp,kd=cfg.port_leg_kd,
                virtual_pair_overtravel_rad=cfg.port_leg_target_overtravel_rad,
                virtual_pair_overtravel_scope=cfg.port_leg_target_overtravel_scope,
                tuck_spring_compensation=dict(cfg.port_tuck_spring_compensation))
        if port and a.training_profile.startswith("foundation-jump-"):
            from wheeled_tasks.wheelbipe_port.mixed_environment import WheelbipeMixedEnv
            env = WheelbipeMixedEnv(cfg)
        else:
            env = (WheelbipePortEnv if port else ReferenceEnv)(cfg)
        if a.gas_feedforward == "disabled":
            if not port:
                raise ValueError("The reference has no spring-cancellation feedforward to disable")
            motor_efforts = env.control.motor_efforts
            env.control.gas_compensation_effort = torch.zeros(a.num_envs, 4, device=env.device)
            def without_spring_cancellation(*args, **kwargs):
                kwargs["apply_gas_compensation"] = False
                return motor_efforts(*args, **kwargs)
            env.control.motor_efforts = without_spring_cancellation
        actor = None
        if a.checkpoint is not None:
            state = torch.load(a.checkpoint, map_location=env.device, weights_only=False)["model_state_dict"]
            if any("obs_normalizer" in k for k in state):
                raise ValueError("This audit expects the original non-normalized actor ABI")
            layers = []
            for i, (ni, no) in enumerate(((35, 256), (256, 128), (128, 64), (64, 6))):
                layer = torch.nn.Linear(ni, no, device=env.device)
                layer.weight.data.copy_(state[f"actor.{i*2}.weight"])
                layer.bias.data.copy_(state[f"actor.{i*2}.bias"])
                layers.append(layer)
                if i < 3:
                    layers.append(torch.nn.ELU())
            actor = torch.nn.Sequential(*layers).eval()
            receipt["checkpoint_sha256"] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        writer = SummaryWriter(str(a.output / "events"))
        receipt.update(status="sampling", actor_joint_order=[env.robot.joint_names[i] for i in env._actuate_idx],
            native_joint_order=env.robot.cad_joint_names if port else env.robot.joint_names,
            actor_action_order=[env.robot.joint_names[i] for i in env._legs_act_idx + env._wheel_idx],
            default_motor_position_rad=env.robot.data.default_joint_pos[0, env._actuate_idx].cpu().tolist(),
            leg_action_scale=env.leg_action_scale, wheel_action_scale=env.wheel_action_scale,
            reward_weights=dict(cfg.rewards), cases=[])
        # Independently measure excluded anchors on the original USD too.
        anchors = []
        if not port:
            for prim in env.sim.stage.Traverse():
                if not str(prim.GetPath()).startswith("/World/envs/env_0/") or not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                if not joint.GetExcludeFromArticulationAttr().Get():
                    continue
                names = [str(v).split("/")[-1] for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()) for v in rel.GetTargets()]
                if len(names) == 2 and all(n in env.robot.body_names for n in names):
                    anchors.append((env.robot.body_names.index(names[0]), env.robot.body_names.index(names[1]),
                        list(joint.GetLocalPos0Attr().Get()), list(joint.GetLocalPos1Attr().Get())))
            if not anchors:
                raise ValueError("No measurable reference anchors")
            id0, id1 = [r[0] for r in anchors], [r[1] for r in anchors]
            p0, p1 = [torch.tensor([r[i] for r in anchors], device=env.device) for i in (2, 3)]
        commands = {"stand_low": (0., 0., .23), "stand_mid": (0., 0., .305),
            "stand_high": (0., 0., .42), "forward": (.5, 0., .305), "reverse": (-.5, 0., .305),
            "spin_positive": (0., 100.*2.*torch.pi/60., .305),
            "spin_negative": (0., -100.*2.*torch.pi/60., .305)}
        if a.control == "wheel-pulse":
            commands = {"wheel_same_positive": (0., 0., .305), "wheel_same_negative": (0., 0., .305),
                "wheel_opposite": (0., 0., .305)}
            a.cases = list(commands)
        masses = env.robot.root_physx_view.get_masses().to(env.device)
        receipt["total_mass_kg"] = float(masses[0].sum())
        for name in a.cases:
            command = commands[name]
            env.eval_command = command
            with torch.inference_mode():
                obs, _ = env.reset()
            origin = env.robot.data.root_pos_w.clone()
            rows, rewards, resets, samples = {}, {}, 0, []
            def record(key, value):
                rows.setdefault(key, []).append(value.detach().float().clone())
            for step in range(a.steps):
                with torch.inference_mode():
                    if a.control == "actor":
                        action = actor(obs["policy"])
                    else:
                        action = torch.zeros(a.num_envs, 6, device=env.device)
                        length = torch.full((a.num_envs, 2), .255 if port else .25, device=env.device)
                        if port:
                            target = env.reset_lookup.sample(length, torch.zeros_like(length))[:, env._legs_act_idx]
                        else:
                            target = env._inverse_kinematics(length, torch.zeros_like(length)).transpose(-2, -1).reshape(a.num_envs, -1)[:, :4]
                        action[:, :4] = (target - env.robot.data.default_joint_pos[:, env._legs_act_idx])/env.leg_action_scale
                        pulse = {"wheel_same_positive": (.5, .5), "wheel_same_negative": (-.5, -.5), "wheel_opposite": (.5, -.5)}[name]
                        if step >= 10:
                            action[:, 4:] = torch.tensor(pulse, device=env.device)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated | truncated
                    resets += int(done.sum())
                    data = env.robot.data
                    if not torch.isfinite(torch.cat((data.root_state_w, data.joint_pos, data.joint_vel, reward[:, None]), -1)).all():
                        raise ValueError(f"Nonfinite state/reward in {name}:{step}")
                    record("vx_m_s", data.root_lin_vel_b[:, 0])
                    record("vy_m_s", data.root_lin_vel_b[:, 1])
                    record("yaw_rad_s", data.root_ang_vel_b[:, 2])
                    record("height_m", data.root_pos_w[:, 2])
                    record("height_error_m", data.root_pos_w[:, 2]-command[2])
                    record("tilt_deg", torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1., 1.))*180./torch.pi)
                    record("action", action)
                    record("leg_q_rad", data.joint_pos[:, env._legs_act_idx])
                    record("leg_dq_rad_s", data.joint_vel[:, env._legs_act_idx])
                    record("wheel_dq_rad_s", data.joint_vel[:, env._wheel_idx])
                    record("wheel_target_rad_s", env.wheel_actions)
                    record("physical_motor_torque_Nm", data.applied_torque[:, env._actuate_idx])
                    record("total_reward", reward)
                    wheel_pos = data.body_pos_w[:, env._wheel_link_idx]
                    com_pos = data.body_com_pos_w
                    com = (com_pos*masses[..., None]).sum(1)/masses.sum(1, keepdim=True)
                    record("com_minus_axle_world_x_m", com[:, 0]-wheel_pos[:, :, 0].mean(1))
                    record("wheel_fork_world_x_m", wheel_pos[:, 0, 0]-wheel_pos[:, 1, 0])
                    if port:
                        record("electrical_motor_torque_Nm", env.motor_effort)
                        record("spring_FF_Nm", env.control.gas_compensation_effort)
                        raw = env.leg_actions[:, env.leg_target_order]
                        paired = env.paired_targets(raw, data.joint_pos[:, [env.motor_ids[i] for i in (0, 1, 3, 4)]])
                        record("discarded_target_rad", raw-paired)
                        record("spring_force_N", env.spring_force)
                        record("gap_m", env.constraint_gaps().amax(-1))
                        record("leg_boundary_relative_rad", raw.reshape(-1, 2, 2).diff(dim=-1).squeeze(-1)*raw.new_tensor([1., -1.]))
                    else:
                        pos, quat = data.body_pos_w, data.body_quat_w
                        gap = (pos[:, id0]+quat_apply(quat[:, id0], p0.expand(a.num_envs, -1, -1))
                            - pos[:, id1]-quat_apply(quat[:, id1], p1.expand(a.num_envs, -1, -1))).norm(dim=-1)
                        record("gap_m", gap.amax(-1))
                    for key, value in env._last_reward_terms.items():
                        if cfg.rewards.get(key, 0.) != 0.:
                            rewards.setdefault(key, []).append(value.detach().float().clone())
                    for key in ("vx_m_s", "height_m", "tilt_deg", "com_minus_axle_world_x_m", "gap_m", "total_reward"):
                        writer.add_scalar(f"{name}/{key}", float(rows[key][-1].mean()), step+1)
                    if step in (0, a.steps//2, a.steps-1):
                        samples.append(dict(step=step+1, observation=obs["policy"][0].cpu().tolist(),
                            action=action[0].cpu().tolist(), command=env.command[0].cpu().tolist(),
                            leg_target=env.leg_actions[0].cpu().tolist(), wheel_target=env.wheel_actions[0].cpu().tolist()))
            tensors = {key: torch.stack(value) for key, value in rows.items()}
            def summary(value):
                # First dimension is policy time, second is independent lane.
                flat = value.flatten(0, 1)
                return dict(mean=flat.mean(0).cpu().tolist(), abs_mean=flat.abs().mean(0).cpu().tolist(),
                    minimum=flat.amin(0).cpu().tolist(), maximum=flat.amax(0).cpu().tolist())
            windows = {}
            for window, start in (("whole", 0), ("tail_half", a.steps//2)):
                windows[window] = {key: summary(value[start:]) for key, value in tensors.items()}
                windows[window]["reward_rate_per_s"] = {key: float(torch.stack(value)[start:].mean()/env.step_dt)
                    for key, value in rewards.items()}
            displacement = (env.robot.data.root_pos_w[:, :2]-origin[:, :2]).norm(dim=-1)
            row = dict(case=name, command=list(command), envs=a.num_envs, duration_s=a.steps*env.step_dt,
                resets=resets, window_definition="tail_half excludes first half, still contains resets if any",
                world_xy_final_displacement_m=summary(displacement.unsqueeze(0)), windows=windows,
                representative_lane0_samples=samples)
            receipt["cases"].append(row)
            writer.flush()
            save()
            print(json.dumps(dict(case=name, resets=resets, vx=windows["tail_half"]["vx_m_s"],
                height=windows["tail_half"]["height_m"])), flush=True)
        receipt.update(status="complete", elapsed_s=time.monotonic()-started)
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
