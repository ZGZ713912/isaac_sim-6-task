#!/usr/bin/env python3
"""Matched zero-PPO solver comparisons with an explicit 200 Hz controller."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--physics-hz', type=int, choices=(200, 1000), default=200)
    p.add_argument('--solver', type=int, nargs=2, default=(64, 32))
    p.add_argument('--clocked-pd', action='store_true')
    p.add_argument('--num-envs', type=int, default=32)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--seed', type=int, default=119)
    p.add_argument('--randomization-strength', type=float, default=1.)
    a = p.parse_args()
    if (a.output.exists() or a.num_envs % 8 or a.steps < 300
            or a.physics_hz != 200 and not a.clocked_pd):
        p.error('Use a new output, a multiple of eight environments, >=300 steps and explicit fine-step PD')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from wheeled_tasks.wheelbipe_port.model import expose_source
    expose_source()
    a.output.mkdir(parents=True)
    receipt = dict(status='preparing', ppo_updates=0, optimizer_created=False,
        hardware_io=False, behavior_qualified=False,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
    def save():
        temporary = a.output/'receipt.tmp'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        temporary.replace(a.output/'receipt.json')
    save()
    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True, device='cuda:0', enable_cameras=False)).app
    env = writer = None
    started = time.monotonic()
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_algo.durable_checkpoint import fingerprint
        from wheeled_tasks.wheelbipe_port.config import WheelbipePortFlatCfg
        from wheeled_tasks.wheelbipe_port.environment import WheelbipePortEnv
        from wheeled_tasks.wheelbipe_port.training_profile import configure_profile
        torch.set_num_threads(4)
        hz = a.physics_hz
        stride = hz//200
        names = ('stand_mid', 'spin_low_positive', 'spin_mid_positive', 'spin_high_positive',
                 'spin_mid_negative', 'drop', 'spin_drop', 'passive_drop')
        spin = 100.*2.*torch.pi/60.
        commands = ((0.,0.,.305), (0.,spin,.23), (0.,spin,.305), (0.,spin,.42),
                    (0.,-spin,.305), (0.,0.,.305), (0.,spin,.305), (0.,0.,.305))

        class ProbeEnv(WheelbipePortEnv):
            probe_ready = False
            physics_samples = control_samples = 0

            def _on_command_updated(self):
                super()._on_command_updated()
                if self.probe_ready:
                    command = self.probe_commands[self.probe_role]
                    self.command[:,0], self.command[:,1], self.command[:,2] = command[:,0], 0., command[:,1]
                    self.height_cmd.copy_(command[:,2])

            def _custom_reset_random(self, ids):
                super()._custom_reset_random(ids)
                if not self.probe_ready:
                    return
                q = self.robot.data.default_joint_pos[ids].clone()
                state = self.robot.data.default_root_state[ids].clone()
                state[:,:3] += self.terrain.env_origins[ids]
                minimum = self.collision_support.minimum_z(q, state[:,3:7])
                state[:,2] = self.terrain.env_origins[ids,2]-minimum+.002
                state[:,2] += .18*(self.probe_role[ids] >= 5)
                state[:,7:] = 0.
                self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), None, ids)
                self.robot.write_root_state_to_sim(state, ids)
                self.predefined_reset_ground_zero_torque_until_time[ids] = 0.

            def _probe_measure(self, common=False):
                gap = self.constraint_gaps().amax(-1)
                velocity = self.robot.data.joint_vel
                nonwheel = [i for i in self._port_rotary_ids if i not in self._wheel_idx]
                speed = velocity[:,nonwheel].abs().amax(-1)
                slider = velocity[:,self.spring_ids].abs().amax(-1)
                wheel_ids = self._find_contact_sensor_indices(['left_wheel_link','right_wheel_link'])
                force = self.contact_sensor.data.net_forces_w[:,wheel_ids].norm(dim=-1).sum(-1)
                self.probe_gap_peak = torch.maximum(self.probe_gap_peak, gap)
                self.probe_nonwheel_speed_peak = torch.maximum(self.probe_nonwheel_speed_peak, speed)
                self.probe_slider_peak = torch.maximum(self.probe_slider_peak, slider)
                self.probe_wheel_force_peak = torch.maximum(self.probe_wheel_force_peak, force)
                if float(gap.max()) > self.probe_peak.get('gap_m', -1.):
                    lane = int(gap.argmax())
                    self.probe_peak = dict(gap_m=float(gap[lane]), lane=lane,
                        case=names[int(self.probe_role[lane])], physics_sample=self.physics_samples,
                        joint_velocities={n:float(v) for n,v in zip(self.robot.cad_joint_names,velocity[lane])})
                if common:
                    self.probe_common_gap_peak = torch.maximum(self.probe_common_gap_peak, gap)
                    self.probe_gap_bad_count += gap > .003
                    self.probe_common_count += 1

            def _apply_action(self):
                if self.probe_ready:
                    self._probe_measure(common=self.physics_samples % stride == 0)
                if not a.clocked_pd:
                    super()._apply_action()
                    self.control_samples += 1
                else:
                    self.port_physics_calls += 1
                    q = self.robot.data.joint_pos[:,self.motor_ids]
                    dq = self.robot.data.joint_vel[:,self.motor_ids]
                    if self.physics_samples % stride == 0 or not hasattr(self,'held_motor_command'):
                        feedback = q.clone()
                        feedback[:,(0,1,3,4)] += self.pd_encoder_bias
                        legs = self.paired_targets(self.leg_actions[:,self.leg_target_order],feedback[:,(0,1,3,4)])
                        wheels = self.wheel_actions[:,self.wheel_target_order].clamp(-90.,90.)
                        motor = self.control.motor_efforts(feedback,dq,legs,wheels,
                            apply_gas_compensation=getattr(self.cfg,'port_gas_feedforward',True))*self.motor_strength
                        motor[:,:2].clamp_(-40.,40.)
                        motor[:,3:5].clamp_(-40.,40.)
                        motor[:,(2,5)] = motor[:,(2,5)].clamp(-4.5,4.5)
                        limits = self.control.gas_compensation.cfg['relative_soft_limits']
                        coefficients = q.new_tensor([[-1.,1.],[1.,-1.]])
                        relative = (q[:,(0,1,3,4)].reshape(-1,2,2)*coefficients).sum(-1)
                        effort = motor[:,(0,1,3,4)].reshape(-1,2,2)
                        outward = (effort*coefficients).sum(-1)
                        blocked = ((relative>=limits['maximum']-limits['margin'])&(outward>0.))|((relative<=limits['minimum']+limits['margin'])&(outward<0.))
                        motor[:,(0,1,3,4)] = (effort-coefficients*torch.where(blocked,outward/2.,0.).unsqueeze(-1)).flatten(-2)
                        self.held_motor_command = motor
                        self.control_samples += 1
                    enabled = self.finish_init & (self.episode_length_buf*self.step_dt >= self.predefined_reset_ground_zero_torque_until_time)
                    motor = self.held_motor_command.clone()
                    motor[~enabled] = 0.
                    motor = self.wheel_response.apply_torque(motor,dq)
                    motor[~enabled] = 0.
                    self.motor_effort.copy_(motor)
                    physical = motor.clone()
                    physical[:,(0,1,3,4)] += self.leg_friction.effort(dq[:,(0,1,3,4)])
                    physical[:,(2,5)] += self.wheel_response.passive_effort
                    self.applied_physical_effort.copy_(physical)
                    self.robot.set_joint_effort_target(physical,joint_ids=self.motor_ids)
                    self._apply_spring()
                    self._update_obs()
                if self.probe_ready:
                    passive = self.probe_role == 7
                    self.motor_effort[passive] = 0.
                    self.applied_physical_effort[passive] = 0.
                    self.robot.set_joint_effort_target(self.applied_physical_effort,joint_ids=self.motor_ids)
                self.physics_samples += 1

            def _get_dones(self):
                if self.probe_ready:
                    self._probe_measure(common=False)
                return super()._get_dones()

        cfg = WheelbipePortFlatCfg()
        cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
        cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1./hz, hz//50, hz//50
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = a.solver[0]
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = a.solver[1]
        cfg.robot_cfg.actuators['legs_act'].armature = .015795
        cfg.port_leg_kp, cfg.port_leg_kd = 60., 2.
        cfg.port_record_peaks = True
        cfg.port_randomization_profile = 'requested'
        cfg.port_randomization_strength = a.randomization_strength
        configure_profile(cfg,'foundation-v2')
        cfg.use_leg_random_start = False
        cfg.events.reset_base.params['pose_range'] = {}
        cfg.events.reset_base.params['velocity_range'] = {}
        cfg.commands.special_modes = {}
        env = ProbeEnv(cfg)
        env.eval_command = (0.,0.,.305)
        env.probe_role = torch.arange(a.num_envs,device=env.device) % len(names)
        env.probe_commands = torch.tensor(commands,device=env.device)
        for attr in ('probe_gap_peak','probe_common_gap_peak','probe_nonwheel_speed_peak','probe_slider_peak','probe_wheel_force_peak','probe_gap_bad_count'):
            setattr(env,attr,torch.zeros(a.num_envs,device=env.device))
        env.probe_common_count = 0
        env.probe_peak = {}
        env.probe_ready = True
        state = torch.load(a.checkpoint,map_location=env.device,weights_only=False)['model_state_dict']
        layers = []
        for i,(ni,no) in enumerate(((35,256),(256,128),(128,64),(64,6))):
            layer = torch.nn.Linear(ni,no,device=env.device)
            layer.weight.data.copy_(state[f'actor.{i*2}.weight'])
            layer.bias.data.copy_(state[f'actor.{i*2}.bias'])
            layers.append(layer)
            if i < 3:
                layers.append(torch.nn.ELU())
        actor = torch.nn.Sequential(*layers).eval()
        with torch.inference_mode():
            obs,_ = env.reset()
        initial = dict(root=env.robot.data.root_state_w,joints=env.robot.data.joint_pos,
            joint_velocity=env.robot.data.joint_vel,actor_observation=obs['policy'],
            mass=env.robot.root_physx_view.get_masses(),inertia=env.robot.root_physx_view.get_inertias(),
            materials=env.robot.root_physx_view.get_material_properties(),coms=env.robot.root_physx_view.get_coms(),
            armature=env.robot.root_physx_view.get_dof_armatures())
        initial_hashes = {key:fingerprint(value)for key,value in initial.items()}
        # The pinned 5.1 tensor view does not expose native iteration counters.
        # Read applied stage attributes on all roots and scene clamps instead.
        from pxr import UsdPhysics, PhysxSchema
        roots = [prim for prim in env.sim.stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != a.num_envs:
            raise ValueError('Applied articulation root count differs')
        solver_readback = {}
        for label,method,expected in [('position','GetSolverPositionIterationCountAttr',a.solver[0]),
                                      ('velocity','GetSolverVelocityIterationCountAttr',a.solver[1])]:
            value = [getattr(PhysxSchema.PhysxArticulationAPI(prim),method)().Get() for prim in roots]
            if any(v != expected for v in value):
                raise ValueError('Applied stage solver iteration override differs')
            solver_readback[label] = dict(minimum=min(value),maximum=max(value),
                scope='USD applied root attributes; internal solver counts unavailable')
        env.physics_samples = env.control_samples = 0
        env.port_physics_calls = 0
        writer = SummaryWriter(str(a.output/'events'))
        sums = {name:dict(resets=0,terminated=0,timeout=0,vx_error=0.,yaw_error=0.,height_error=0.,tilt=0.)for name in names}
        trajectory_hash = hashlib.sha256()
        receipt.update(status='sampling',versions={n:importlib.metadata.version(n)for n in('isaacsim','isaaclab','rsl-rl-lib','torch')},
            policy_hz=50,pd_hz=200,physics_hz=hz,solver_readback=solver_readback,initial_hashes=initial_hashes,
            passive_model_updates_hz=hz,external_forces_every_iteration=False,
            interpretation='Controlled cold starts; not exact replay of training event or physical qualification')
        save()
        sample_start = time.monotonic()
        with torch.inference_mode():
            for step in range(a.steps):
                actions = actor(obs['policy'])
                obs,reward,terminate,timeout,extras = env.step(actions)
                all_state = torch.cat((env.robot.data.root_state_w,env.robot.data.joint_pos,env.robot.data.joint_vel),-1)
                if not torch.isfinite(all_state).all() or not torch.isfinite(reward).all():
                    raise ValueError('Nonfinite native dynamics')
                trajectory_hash.update(all_state.cpu().numpy().tobytes())
                for i,name in enumerate(names):
                    mask = env.probe_role == i
                    row = sums[name]
                    row['resets'] += int((terminate[mask] | timeout[mask]).sum())
                    row['terminated'] += int(terminate[mask].sum())
                    row['timeout'] += int(timeout[mask].sum())
                    if step >= a.steps//2:
                        target = env.probe_commands[env.probe_role]
                        row['vx_error'] += float((env.robot.data.root_lin_vel_b[mask,0]-target[mask,0]).abs().sum())
                        row['yaw_error'] += float((env.robot.data.root_ang_vel_b[mask,2]-target[mask,1]).abs().sum())
                        row['height_error'] += float((env.robot.data.root_pos_w[mask,2]-env.terrain.env_origins[mask,2]-target[mask,2]).abs().sum())
                        row['tilt'] += float(torch.acos((-env.robot.data.projected_gravity_b[mask,2]).clamp(-1.,1.)).sum()*180./torch.pi)
                    if (step+1)%10 == 0:
                        writer.add_scalar('Solver/'+name+'/gap_max_m',float(env.probe_gap_peak[mask].max()),step+1)
                        writer.add_scalar('Solver/'+name+'/nonwheel_speed_peak_rad_s',float(env.probe_nonwheel_speed_peak[mask].max()),step+1)
                if (step+1)%100 == 0:
                    print(json.dumps(dict(step=step+1,gap_peak_m=float(env.probe_gap_peak.max()))),flush=True)
        elapsed = time.monotonic()-sample_start
        for i,name in enumerate(names):
            mask = env.probe_role == i
            row = sums[name]
            for key in ('vx_error','yaw_error','height_error','tilt'):
                row[key] /= (a.num_envs//len(names))*(a.steps-a.steps//2)
            row.update(gap_peak_full_physics_m=float(env.probe_gap_peak[mask].max()),
                gap_peak_common_200hz_m=float(env.probe_common_gap_peak[mask].max()),
                gap_over_3mm_common_fraction=float(env.probe_gap_bad_count[mask].sum())/(int(mask.sum())*env.probe_common_count),
                nonwheel_speed_peak_rad_s=float(env.probe_nonwheel_speed_peak[mask].max()),
                slider_speed_peak_m_s=float(env.probe_slider_peak[mask].max()),
                wheel_force_peak_n=float(env.probe_wheel_force_peak[mask].max()))
        if env.control_samples != a.steps*4 or env.physics_samples != a.steps*(hz//50):
            raise ValueError('Measured controller/physics clock count differs')
        if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != receipt['checkpoint_sha256']:
            raise ValueError('Read-only policy checkpoint changed')
        receipt.update(status='complete',cases=sums,peak=env.probe_peak,physics_calls=env.physics_samples,
            pd_calls=env.control_samples,trajectory_sha256=trajectory_hash.hexdigest(),
            simulation_seconds=a.steps/50.,sampling_wall_s=elapsed,total_wall_s=time.monotonic()-started,
            wall_s_per_simulated_second=elapsed/(a.steps/50.),finite_native_states=True)
        save()
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        save()
        raise
    finally:
        if writer:
            writer.flush()
            writer.close()
        if env:
            env.close()
        app.close(skip_cleanup=True)


if __name__ == '__main__':
    main()
