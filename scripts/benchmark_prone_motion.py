"""Bounded local performance/trajectory probe. Never creates or reads a policy."""
import argparse
from collections.abc import Mapping
import cProfile
import datetime
import hashlib
import inspect
import json
import os
import pickle
from pathlib import Path
import pstats
import random
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--plan', type=Path, help='Explicit versioned plan; default is current v4')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--steps', type=int, default=96)
parser.add_argument('--warmup', type=int, default=16)
parser.add_argument('--profile-steps', type=int, default=24)
parser.add_argument('--torch-profile-steps', type=int, default=0,
    help='Optional CPU/CUDA operator trace after the unprofiled timing window')
parser.add_argument('--trace-steps', type=int, default=32)
parser.add_argument('--profile', default='nominal')
parser.add_argument('--strength', type=float, default=0.)
parser.add_argument('--actor-updates', type=int, default=0)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--physics-hz', type=int, choices=(200,1000))
parser.add_argument('--policy-hz', type=int, choices=(50,100))
parser.add_argument('--rollout-steps', type=int)
parser.add_argument('--hidden-dims', type=int,nargs='+')
parser.add_argument('--ppo-epochs', type=int)
parser.add_argument('--ppo-mini-batches', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--environment-layout', choices=('grid', 'overlap'))
parser.add_argument('--position-iterations', type=int)
parser.add_argument('--velocity-iterations', type=int)
parser.add_argument('--global-initial',action='store_true',help='Use the actual mastery course initial all-standing reset')
parser.add_argument('--backend',choices=('eager','cuda_graph'),default='cuda_graph')
parser.add_argument('--sampling-backend',choices=('packed','reference'),default='packed')
parser.add_argument('--instrument', action='store_true')
parser.add_argument('--kit-threads', type=int, default=0)
args = parser.parse_args()
if args.steps < 1 or min(args.warmup,args.profile_steps,args.torch_profile_steps,args.trace_steps,args.kit_threads) < 0:
    parser.error('Timed steps must be positive; all other counts must be nonnegative')
root, out = args.root.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path[:0] = [str(root/'scripts'), str(root/'src')]
import train_prone_motion as entry
entry.check_other_learners()
from wheeled_tasks.chassis.prone_motion import validate
plan_path = (args.plan or root/'contracts/wheel_leg_prone_motion_v4.json').resolve()
plan = validate(json.loads(plan_path.read_text()))
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, OPTION_KEYS
layout_override = {}
if 'environment_layout' in OPTION_KEYS:
    layout_override['environment_layout'] = args.environment_layout
elif args.environment_layout is not None:
    parser.error('The selected source revision does not support layout overrides')
plan = resolve_plan(plan, num_envs=args.num_envs, position_iterations=args.position_iterations,
                    velocity_iterations=args.velocity_iterations, physics_hz=args.physics_hz, seed=args.seed,
                    policy_hz=args.policy_hz,num_steps_per_env=args.rollout_steps,hidden_dims=args.hidden_dims,
                    ppo_epochs=args.ppo_epochs,ppo_mini_batches=args.ppo_mini_batches,
                    **layout_override)
hashes = entry.source_hashes(plan_path, plan)
result = dict(schema='local-prone-motion-performance-v1', status='starting', pid=os.getpid(),
    start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root=str(root),
    new_PPO=0, optimizer_attempts=0, actor_or_critic_created=False, checkpoint_reads=[],
    source_hashes=hashes, backend=args.backend,agent_configuration=entry.learner_config(plan),
    args=vars(args)|{'root':str(root),'output':str(out),'plan':str(plan_path)},
    behavior_qualification=False, hardware_ready=False)
entry.base.write_json(out/'result.json',result)
launcher = env = None
try:
    from isaaclab.app import AppLauncher
    kit_args = '--/exts/omni.kit.telemetry/skipDeferredStartup=true'
    if args.kit_threads:
        kit_args += (f' --/plugins/carb.tasking.plugin/threadCount={args.kit_threads}'
            f' --/plugins/omni.tbb.globalcontrol/maxThreadCount={args.kit_threads}'
            f' --/persistent/physics/numThreads={args.kit_threads}')
    launcher = AppLauncher(dict(headless=True,device='cuda:0',enable_cameras=False,kit_args=kit_args))
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    entry.base.torch = torch
    torch.set_num_threads(4)
    seed = plan['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    def forbidden(*a, **kw):
        result['checkpoint_reads'].append(str(a[0] if a else kw.get('f')))
        raise RuntimeError('No checkpoint reads allowed in the performance probe')
    torch.load = forbidden
    backend_kw = ({'performance_backend':args.backend, 'sampling_backend':args.sampling_backend}
        if 'performance_backend' in inspect.signature(entry.configuration).parameters else {})
    cfg, settings, bank = entry.configuration(out/'config_context',plan,profile=args.profile,
        strength=args.strength,num_envs=args.num_envs,seed=seed,**backend_kw)
    manifest = json.loads((root/cfg['asset_directory']/'manifest.json').read_text())
    course_kw = {}
    if args.global_initial:
        from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
        from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
        mastery = RecoveryMastery(plan['multi_pose_recovery']['mastery_gate'])
        course_kw = dict(recovery_mastery=mastery,global_curriculum=GlobalCurriculum(plan,mastery))
    env = ProneMotionEnv(cfg,manifest,load_chassis_control(root/cfg['control_math_source']),root,
        stage_name='speed',num_envs=args.num_envs,device='cuda:0',seed=seed,level=1.,run_role='train',bank=bank,settings=settings,**course_kw)
    from pxr import PhysxSchema
    solver_readback = []
    for prim in env.sim.stage.Traverse():
        if str(prim.GetPath()).startswith('/World/envs/env_') and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            api = PhysxSchema.PhysxArticulationAPI(prim)
            solver_readback.append([api.GetSolverPositionIterationCountAttr().Get(),
                                    api.GetSolverVelocityIterationCountAttr().Get()])
    expected_solver = [cfg['position_iterations'],cfg['velocity_iterations']]
    result['solver_usd_readback'] = dict(articulation_count=len(solver_readback),
        unique_iteration_counts=sorted({tuple(v) for v in solver_readback}),
        all_match_requested=len(solver_readback)==env.num_envs and all(v==expected_solver for v in solver_readback))
    if not result['solver_usd_readback']['all_match_requested']:
        raise RuntimeError('Authored articulation solver iterations do not match the requested run')
    if args.backend != 'eager' and not hasattr(env,'performance_backend'):
        from wheeled_tasks.chassis.prone_motion_performance import install
        install(env, args.backend)
    env.actor_updates = args.actor_updates
    entry.base.write_json(out/'native_startup.json',env.startup_report)
    result['runtime'] = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
        policy_dt=env.policy_dt, physics_dt=env.dt, decimation=env.decimation, num_envs=env.num_envs,
        position_iterations=cfg['position_iterations'],velocity_iterations=cfg['velocity_iterations'],
        record_diagnostics=cfg['record_diagnostics'])
    import carb.settings
    result['runtime']['kit_threads'] = {k:carb.settings.get_settings().get(k) for k in (
        '/plugins/carb.tasking.plugin/threadCount','/plugins/omni.tbb.globalcontrol/maxThreadCount',
        '/persistent/physics/numThreads')}
    with torch.inference_mode():
        env.reset(torch.arange(env.num_envs, device=env.device))
        action = entry.base.nominal_action(env).expand(env.num_envs,-1).clone()
        # Fixed gentle six-axis excitation; no teacher or learned policy.
        action += .015*torch.sin(torch.arange(env.num_envs*6,device=env.device).reshape(-1,6))
        initial = {'action':action.cpu().numpy(),
                   'root':env.robot.data.root_link_pose_w.torch.cpu().numpy().copy(),
                   'joint':env.robot.data.joint_pos.torch.cpu().numpy().copy()}
        trace = {}
        def rng_snapshot():
            states = dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().numpy().tobytes(),
                cuda=[state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()],
                environment=env.generator.get_state().cpu().numpy().tobytes(),
                domain=env.domain.rng.get_state().cpu().numpy().tobytes())
            if hasattr(env, 'recovery_sampler'):
                states['recovery_sampler'] = env.recovery_sampler.rng.getstate()
            return {name:hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
                    for name,value in states.items()}
        result['rng_before_trace'] = rng_snapshot()
        def capture(prefix, value):
            if isinstance(value,torch.Tensor):
                trace.setdefault(prefix,[]).append(value.detach().cpu().numpy().copy())
            elif isinstance(value,Mapping):
                for k,v in value.items(): capture(prefix+'/'+str(k),v)
            elif isinstance(value,(float,int,bool)):
                trace.setdefault(prefix,[]).append(np.asarray(value))
        for i in range(args.trace_steps):
            obs,reward,done,info = env.step(action)
            capture('obs',obs);capture('reward',reward);capture('done',done)
            capture('diag',info.get('diagnostics',{}))
            capture('log',info.get('log',{}))
            capture('physical',getattr(env,'prone_motion_physical',{}))
            capture('root',env.robot.data.root_link_pose_w.torch)
            capture('joint',env.robot.data.joint_pos.torch)
            capture('command',env.commands)
        if args.trace_steps and not {'obs/policy', 'obs/critic'}.issubset(trace):
            raise RuntimeError('Native probe did not capture both policy and critic observations')
        result['observation_trace_keys'] = sorted(k for k in trace if k.startswith('obs/'))
        result['rng_after_trace'] = rng_snapshot()
        result['trace_termination_events'] = int(np.stack(trace['done']).sum()) if args.trace_steps else 0
        np.savez_compressed(out/'trace.npz',**{'initial/'+k:v for k,v in initial.items()},
                            **{k:np.stack(v) for k,v in trace.items()})
        for _ in range(args.warmup): env.step(action)
        counters = {}
        if args.instrument:
            def wrap(obj,name,label):
                original = getattr(obj,name)
                def measured(*a,**kw):
                    t = time.perf_counter()
                    try: return original(*a,**kw)
                    finally:
                        row = counters.setdefault(label,[0,0.])
                        row[0] += 1; row[1] += time.perf_counter()-t
                setattr(obj,name,measured)
            for obj,name,label in ((env.sim,'step','simulation_step'),
                    (env.robot,'write_data_to_sim','write_data'),
                    (env.robot,'set_joint_effort_target_index','set_effort'),
                    (env.v5,'motor_efforts','motor_control'),
                    (env.v5,'spring_efforts','physical_spring'),
                    (env.identified_wheels,'apply_torque','wheel_response'),
                    (env,'task_metrics','task_metrics'),(env,'potential','potential'),
                    (env,'route_state','route_state'),(env,'get_observations','observations'),
                    (env,'task_transition','task_transition'),(env,'reset','reset'),
                    (env,'closure_gap','closure_gap'),(env.robot,'update','robot_update')):
                wrap(obj,name,label)
        times = []
        closure_peak_m = 0.
        physics_closure_peak_m = 0.
        termination_events = 0
        termination_reasons = dict.fromkeys(('mechanical','closure_gap','impact_bad','boundary_bad'),0)
        for _ in range(args.steps):
            torch.cuda.synchronize(); t = time.perf_counter()
            obs,reward,done,info = env.step(action)
            torch.cuda.synchronize(); times.append(time.perf_counter()-t)
            closure_peak_m = max(closure_peak_m, float(env.closure_gap().max()))
            physics_closure_peak_m = max(physics_closure_peak_m,float(info['log']['UnifiedGetup/closure_gap_max_m']))
            termination_events += int(done.sum())
            for key in termination_reasons:
                termination_reasons[key] += int(info['diagnostics'][key].sum())
        result['instrumented_calls'] = {k:dict(count=v[0],total_s=v[1]) for k,v in counters.items()}
        prof = cProfile.Profile()
        prof.enable()
        for _ in range(args.profile_steps): env.step(action)
        torch.cuda.synchronize(); prof.disable()
        prof.dump_stats(str(out/'profile.pstats'))
        with (out/'profile.txt').open('w') as stream:
            pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(65)
        if args.torch_profile_steps:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as operator_profile:
                for _ in range(args.torch_profile_steps):
                    with torch.profiler.record_function('sampling/policy_step'):
                        env.step(action)
            operator_profile.export_chrome_trace(str(out/'operators.json'))
            with (out/'operators.txt').open('w') as stream:
                stream.write(operator_profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=45))
            result['operator_profile'] = {
                item.key:dict(calls=item.count, self_cpu_us=item.self_cpu_time_total,
                    cpu_us=item.cpu_time_total, cuda_us=item.device_time_total)
                for item in operator_profile.key_averages()
                if item.key in ('aten::nonzero','aten::item','aten::_local_scalar_dense',
                    'cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync')}
        result['rng_at_end'] = rng_snapshot()
        finite = all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(reward).all())
    result.update(status='completed' if finite else 'failed_nonfinite', finite=finite,
        collection_per_rollout_s=float(np.mean(times)*plan['rollout_steps']),
        samples_per_rollout=env.num_envs*plan['rollout_steps'],
        simulated_seconds_per_env_per_rollout=plan['rollout_steps']*env.policy_dt,
        physics_calls_per_rollout=plan['rollout_steps']*env.decimation,
        step_mean_s=float(np.mean(times)),step_median_s=float(np.median(times)),step_p90_s=float(np.quantile(times,.9)),
        measured_policy_steps=len(times), measured_physics_steps=len(times)*env.decimation,
        post_step_closure_peak_m=closure_peak_m, termination_events=termination_events,
        physics_closure_peak_m=physics_closure_peak_m,termination_reason_events=termination_reasons,
        step_times_s=times, trace_sha256=entry.base.digest(out/'trace.npz'),
        final_control_timing=env.startup_report['control_timing'])
    writer = SummaryWriter(str(out/'events'))
    for i,v in enumerate(times): writer.add_scalar('PerformanceProbe/policy_step_s',v,i)
    writer.add_scalar('PerformanceProbe/policy_step_mean_s',result['step_mean_s'],len(times))
    writer.add_scalar('PerformanceProbe/policy_samples_per_s',env.num_envs/result['step_mean_s'],len(times))
    for name, row in result.get('operator_profile', {}).items():
        tag = name.replace('::','_')
        writer.add_scalar('PerformanceProbe/'+tag+'_calls_per_policy_step',
            row['calls']/args.torch_profile_steps,len(times))
    writer.close()
    print('PERFORMANCE_RESULT',json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','step_times_s')}),flush=True)
except BaseException:
    result.update(status='failed',error=traceback.format_exc())
    raise
finally:
    result['source_mismatches_at_exit'] = [n for n,h in hashes.items() if entry.base.digest(root/n)!=h]
    entry.base.write_json(out/'result.json',result)
    if env is not None: env.close()
    if launcher is not None: launcher.app.close()
