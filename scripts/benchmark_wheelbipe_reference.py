#!/usr/bin/env python3
"""Time the user's original Wheelbipe V14 environment without a learned policy.

Run with the reference's Isaac Sim 5.1 Python. The source checkout is read-only;
only explicit batch size, clock and solver overrides differ from its flat task.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import types


def save(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--physics-hz', type=int, default=200)
    parser.add_argument('--policy-hz', type=int, default=50)
    parser.add_argument('--position-iterations', type=int, default=12)
    parser.add_argument('--velocity-iterations', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=261028)
    parser.add_argument('--ground-usd', type=Path,
        help='Offline copy of the SDK default_environment.usd; retain its original physics')
    args = parser.parse_args()
    if (min(args.num_envs, args.physics_hz, args.policy_hz, args.steps) < 1
            or args.physics_hz % args.policy_hz or args.warmup < 0):
        parser.error('Require positive sizes and an integer physics/policy ratio')
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='starting', source=str(source), arguments=vars(args) | {
        'ground_usd': str(args.ground_usd) if args.ground_usd else None,
        'source': str(source), 'output': str(output)}, new_PPO=0, optimizer_attempts=0,
        checkpoint_reads=[], behavior_qualification=False)
    files = sorted((source/'src').rglob('*.py'))
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['source_hashes'] = hashes
    save(output/'result.json', result)
    env = launcher = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device='cuda:0', enable_cameras=False))
        import torch
        from torch.utils.tensorboard import SummaryWriter
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        sys.path[:0] = [str(source), str(source/'src')]
        # Avoid importing unrelated robot tasks from agent_tasks/__init__.py.
        # All Wheelbipe modules and their parent packages still run unchanged.
        package = types.ModuleType('agent_tasks')
        package.__path__ = [str(source/'src/agent_tasks')]
        sys.modules['agent_tasks'] = package
        from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import WheelbipeV14FlatEnvCfg
        from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import WheelbipeV14Env
        if args.ground_usd is not None:
            import isaaclab.sim as sim_utils
            original_ground_cfg = sim_utils.GroundPlaneCfg
            ground = args.ground_usd.resolve()
            result['offline_ground'] = dict(file=str(ground), sha256=hashlib.sha256(ground.read_bytes()).hexdigest(),
                original_url=original_ground_cfg().usd_path)
            def cached_ground_cfg(*a, **kw):
                return original_ground_cfg(*a, **kw, usd_path=str(ground))
            sim_utils.GroundPlaneCfg = cached_ground_cfg
        cfg = WheelbipeV14FlatEnvCfg()
        cfg.seed = args.seed
        cfg.scene.num_envs = args.num_envs
        cfg.sim.dt = 1./args.physics_hz
        cfg.decimation = args.physics_hz//args.policy_hz
        cfg.sim.render_interval = cfg.decimation
        props = cfg.robot_cfg.spawn.articulation_props
        props.solver_position_iteration_count = args.position_iterations
        props.solver_velocity_iteration_count = args.velocity_iterations
        env = WheelbipeV14Env(cfg)
        if args.ground_usd is not None:
            sim_utils.GroundPlaneCfg = original_ground_cfg
        result['runtime'] = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
            policy_dt=env.step_dt, physics_dt=cfg.sim.dt, decimation=cfg.decimation,
            num_envs=env.num_envs, joint_count=env.robot.num_joints,
            body_count=env.robot.num_bodies, env_spacing=cfg.scene.env_spacing,
            replicate_physics=cfg.scene.replicate_physics,
            position_iterations=args.position_iterations, velocity_iterations=args.velocity_iterations)
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                env.step(actions)
            counters = {}
            def wrap(obj, name, label):
                original = getattr(obj, name)
                def measured(*a, **kw):
                    started = time.perf_counter()
                    try:
                        return original(*a, **kw)
                    finally:
                        row = counters.setdefault(label, [0, 0.])
                        row[0] += 1
                        row[1] += time.perf_counter()-started
                setattr(obj, name, measured)
            for obj, name, label in ((env.sim, 'step', 'simulation_step'),
                    (env, '_apply_action', 'control'), (env, '_get_observations', 'observations'),
                    (env, '_get_rewards', 'rewards'), (env, '_reset_idx', 'reset'),
                    (env.robot, 'write_data_to_sim', 'write_data')):
                wrap(obj, name, label)
            times, terminated_count = [], 0
            for _ in range(args.steps):
                torch.cuda.synchronize()
                started = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(actions)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
                terminated_count += int((terminated | truncated).sum())
            finite = bool(torch.isfinite(reward).all()) and all(
                bool(torch.isfinite(value).all()) for value in obs.values() if isinstance(value, torch.Tensor))
        mean = sum(times)/len(times)
        result.update(status='completed' if finite else 'failed_nonfinite', finite=finite,
            step_mean_s=mean, step_times_s=times, policy_samples_per_s=env.num_envs/mean,
            collection_per_64_policy_steps_s=mean*64,
            physics_calls_per_64_policy_steps=64*cfg.decimation,
            termination_events=terminated_count,
            instrumented_calls={k:dict(count=v[0], total_s=v[1]) for k, v in counters.items()})
        writer = SummaryWriter(str(output/'events'))
        for i, seconds in enumerate(times):
            writer.add_scalar('PerformanceProbe/policy_step_s', seconds, i)
        writer.add_scalar('PerformanceProbe/policy_step_mean_s', mean, len(times))
        writer.add_scalar('PerformanceProbe/policy_samples_per_s', env.num_envs/mean, len(times))
        writer.close()
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        result['source_mismatches_at_exit'] = [name for name, expected in hashes.items()
            if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected]
        save(output/'result.json', result)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == '__main__':
    main()
