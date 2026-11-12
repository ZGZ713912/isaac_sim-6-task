#!/usr/bin/env python3
"""Offline time-driven actuator sweep for V8; no optimizer or hardware.

Candidates run through the normal six-axis action/PD/friction/spring path.
The best observed trajectory is a feasible reference, never a proven maximum.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')


def candidates(crouch_lengths, push_holds):
    if (not crouch_lengths or not push_holds
            or any(not .16 <= length <= .30 for length in crouch_lengths)
            or any(not .04 <= hold <= .4 for hold in push_holds)):
        raise ValueError('Use crouch lengths in [0.16,0.30] m and push holds in [0.04,0.4] s')
    return [dict(crouch_length_m=length, push_hold_s=hold)
            for length, hold in itertools.product(crouch_lengths, push_holds)]


def sweep_course(spec, count):
    from copy import deepcopy
    from wheeled_tasks.wheelbipe_port.mixed_curriculum import quota_counts
    spec = deepcopy(spec)
    spec['task_fractions'] = {name:1./count for name in spec['tasks']}
    spec['task_fractions']['jump'] = 1.-(len(spec['tasks'])-1)/count
    spec['jump']['request_delay_s'] = [.6, .6]
    quota_counts(count, spec)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--steps', type=int, default=320)
    p.add_argument('--seed', type=int, default=211)
    p.add_argument('--crouch-lengths', type=float, nargs='+', default=[.16, .20, .24])
    p.add_argument('--push-holds', type=float, nargs='+', default=[.08, .12, .18])
    p.add_argument('--tuck-seconds', type=float, default=.16)
    p.add_argument('--randomization-profile', choices=('nominal', 'requested'), default='nominal')
    p.add_argument('--randomization-strength', type=float, default=0.)
    p.add_argument('--checkpoint', type=Path,
        help='Optional read-only 35D actor: balance before the request and control wheels during the sweep; never initializes training')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    grid = candidates(a.crouch_lengths, a.push_holds)
    if a.num_envs < max(20, len(grid)+12) or a.steps < 301 or not .04 <= a.tuck_seconds <= .5:
        p.error('Need enough lanes for every candidate, at least 301 steps and tuck-seconds in [0.04,0.5]')
    if not 0. <= a.randomization_strength <= 1.:
        p.error('randomization-strength must be in [0,1]')
    if a.checkpoint is not None and not a.checkpoint.is_file():
        p.error('The read-only baseline checkpoint must exist')
    if a.plan_only:
        print(json.dumps(dict(candidates=grid, ppo_updates=0, optimizer_created=False), indent=2))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='preparing', ppo_updates=0, actor_created=False, optimizer_created=False,
        hardware_io=False, behavior_qualified=False, candidates=grid,
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        interpretation='Time-driven target intervention measures this model only; no teacher/donor, maximum-height proof or hardware qualification')
    def save():
        (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    save()
    from wheeled_tasks.wheelbipe_port.model import expose_source
    expose_source()
    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True, device='cuda:0', enable_cameras=False)).app
    env = writer = None
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.wheelbipe_port.config import WheelbipePortFlatCfg
        from wheeled_tasks.wheelbipe_port.mixed_environment import WheelbipeMixedEnv
        from wheeled_tasks.wheelbipe_port.training_profile import configure_profile, apply_phase
        torch.set_num_threads(4)
        cfg = WheelbipePortFlatCfg()
        cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
        cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .005, 4, 4
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = 64
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 32
        cfg.port_randomization_profile = a.randomization_profile
        cfg.port_randomization_strength = a.randomization_strength
        configure_profile(cfg, 'foundation-jump-v8')
        cfg.port_start_completed_updates = 0
        apply_phase(cfg, 0)
        cfg.use_leg_random_start = False
        cfg.events.reset_base.params['pose_range'] = {}
        cfg.events.reset_base.params['velocity_range'] = {}
        # Keep every task present, with most lanes reserved for the sweep.
        base_profile_hash = cfg.port_profile_sha256
        spec = sweep_course(cfg.port_mixed_spec, a.num_envs)
        cfg.port_mixed_spec = spec
        cfg.port_profile_sha256 = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        limit = cfg.port_real2sim['leg_friction']['native_effort_limit_nm']
        cfg.robot_cfg.actuators['legs_act'].effort_limit = limit
        cfg.robot_cfg.actuators['legs_act'].effort_limit_sim = limit
        env = WheelbipeMixedEnv(cfg)
        actor = None
        checkpoint_digest = None
        if a.checkpoint is not None:
            checkpoint_digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
            state = torch.load(a.checkpoint, map_location=env.device, weights_only=False)['model_state_dict']
            layers = []
            for i, (ni, no) in enumerate(((35, 256), (256, 128), (128, 64), (64, 6))):
                layer = torch.nn.Linear(ni, no, device=env.device)
                layer.weight.data.copy_(state[f'actor.{i*2}.weight'])
                layer.bias.data.copy_(state[f'actor.{i*2}.bias'])
                layers.append(layer)
                if i < 3:
                    layers.append(torch.nn.ELU())
            actor = torch.nn.Sequential(*layers).eval()
        writer = SummaryWriter(str(a.output/'events'))
        lane = env.task_schedule.mask('jump')
        indices = lane.nonzero().flatten()
        assignment = torch.zeros(a.num_envs, dtype=torch.long, device=env.device)
        assignment[indices] = torch.arange(len(indices), device=env.device) % len(grid)
        low = torch.tensor([x['crouch_length_m'] for x in grid], device=env.device)[assignment]
        hold = torch.tensor([x['push_hold_s'] for x in grid], device=env.device)[assignment]
        rows = []
        original_dones = env._get_dones
        def observe_outcomes():
            terminated, timeout = original_dones()
            done = (terminated | timeout) & lane
            seq, observer = env.jump_observer.sequence, env.jump_observer
            for i in done.nonzero().flatten().tolist():
                row = dict(lane=i, candidate=int(assignment[i]), terminated=bool(terminated[i]),
                    flown=bool(observer.flown[i]), succeeded=bool(observer.succeeded[i]))
                row.update({key:float(seq.values[key][i]) for key in ('release_vz', 'flight_com_rise',
                    'max_tuck', 'max_wheel_clearance', 'push_work', 'push_impulse', 'push_duration',
                    'push_force_peak_bw', 'landing_force_peak_bw', 'reextension', 'buffer_compression')})
                rows.append(row)
            return terminated, timeout
        env._get_dones = observe_outcomes
        obs, _ = env.reset()
        if obs['policy'].shape != (a.num_envs, 35) or obs['critic'].shape != (a.num_envs, 78):
            raise ValueError('Jump diagnostic changed the observation ABI')
        receipt.update(status='sampling', domain=env.domain_receipt,
            actor_created=actor is not None, checkpoint_sha256=checkpoint_digest,
            observation_shapes={k:list(v.shape) for k,v in obs.items()},
            policy_dt_s=env.step_dt, physics_dt_s=env.physics_dt,
            profile_sha256=base_profile_hash, effective_course_sha256=cfg.port_profile_sha256, effective_course=spec,
            implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'scripts/probe_wheelbipe_jump.py', 'src/wheeled_tasks/wheelbipe_port/jump_objectives.py',
                'src/wheeled_tasks/wheelbipe_port/mixed_curriculum.py',
                'src/wheeled_tasks/wheelbipe_port/mixed_environment.py')})
        save()
        for step in range(a.steps):
            age = env.episode_length_buf.float()*env.step_dt-env.task_schedule.delay
            prepare = (age/cfg.port_mixed_spec['jump']['crouch_seconds']).clamp(0., 1.)
            prepare = 3*prepare.square()-2*prepare.pow(3)
            length = .25+(low-.25)*prepare
            push_start = cfg.port_mixed_spec['jump']['crouch_seconds']
            length = torch.where(age >= push_start, .36, length)
            length = torch.where(age >= push_start+hold, .16, length)
            length = torch.where(age >= push_start+hold+a.tuck_seconds, .32, length)
            length = torch.where(age >= push_start+hold+a.tuck_seconds+.5, .25, length)
            q = env.reset_lookup.sample(length[:, None].expand(-1, 2), torch.zeros(a.num_envs, 2, device=env.device))
            with torch.inference_mode():
                actions = actor(obs['policy']) if actor is not None else torch.zeros(a.num_envs, 6, device=env.device)
                selected = lane & (age >= 0.)
                actions[selected, :4] = ((q-env.robot.data.default_joint_pos)[:, env._legs_act_idx]/cfg.leg_action_scale)[selected]
                obs, reward, _, _, extras = env.step(actions)
            if not (torch.isfinite(reward).all() and torch.isfinite(env.robot.data.root_state_w).all()
                    and torch.isfinite(env.robot.data.joint_vel).all()):
                raise ValueError('Nonfinite native sweep state')
            if env.motor_effort[:, (0, 1, 3, 4)].abs().max() > 40.0001:
                raise ValueError('Electrical leg torque exceeded 40 Nm')
            for tag, value in extras.get('log', {}).items():
                if tag.startswith(('JumpSequence/', 'Mixed/', 'Port/')):
                    writer.add_scalar(tag, float(value), step+1)
        eligible = [r for r in rows if r['flown'] and not r['terminated']]
        receipt.update(status='complete', episode_rows=rows,
            best_observed=(max(eligible, key=lambda row:row['release_vz']) if eligible else None),
            jump_sequence_metrics={k:float(v) for k,v in env.jump_observer.sequence.metrics().items()})
        if not rows:
            raise ValueError('No completed sweep episodes; increase --steps')
        if a.checkpoint is not None and hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != checkpoint_digest:
            raise ValueError('Read-only baseline checkpoint changed')
        save()
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
