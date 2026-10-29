import collections,datetime,gc,hashlib,json,os,pathlib,random,resource,shutil,sys,time,traceback
root=pathlib.Path('/root/gpufree-data/workspace/prone_motion_long')
sys.path.insert(0,str(root/'scripts'))
import train_prone_motion as entry
from wheeled_tasks.chassis.prone_motion import validate
out=root/'runs/server_compatibility_20261029/full256_smoke001'
out.mkdir(parents=True,exist_ok=False)
plan_path=root/'contracts/wheel_leg_prone_motion_v1.json'
plan=validate(json.loads(plan_path.read_text()))
hashes=entry.source_hashes(plan_path,plan)
source=entry.base.archive_committed_sources(hashes,out)
result=dict(schema='server-full-population-smoke-v1',status='starting',source=source,pid=os.getpid(),start_ticks=int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),num_envs=plan['num_envs'],policy_steps_per_domain=64,new_PPO=0,optimizer_attempts=0,actor_or_critic_created=False,checkpoint_reads=[],contexts=[],learned_behavior=False,hardware_ready=False,full_capability_qualified=False)
launcher=env=None
start=time.monotonic()
try:
 from isaaclab.app import AppLauncher
 launcher=AppLauncher(dict(headless=True,device='cuda:0',enable_cameras=False,kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
 import numpy as np
 import torch
 from isaaclab.sim import SimulationContext
 from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
 from wheeled_tasks.chassis.control_contract import load_chassis_control
 from train_scratch_standing import nominal_action
 entry.base.torch=torch
 torch.set_num_threads(4)
 def forbidden(*args,**kwargs):
  result['checkpoint_reads'].append(str(args[0] if args else kwargs.get('f')))
  raise RuntimeError('Checkpoint reads forbidden in zero-PPO software smoke')
 torch.load=forbidden
 for index,(profile,strength) in enumerate([('nominal',0.),('requested',1.)]):
  seed=plan['seed']+index*31
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  cfg,settings,bank=entry.configuration(out/f'{index:03d}_{profile}',plan,profile=profile,strength=strength,num_envs=plan['num_envs'],seed=seed)
  manifest=json.loads((root/cfg['asset_directory']/'manifest.json').read_text())
  env=ProneMotionEnv(cfg,manifest,load_chassis_control(root/cfg['control_math_source']),root,stage_name='speed',num_envs=plan['num_envs'],device='cuda:0',seed=seed,level=1.,run_role='train',bank=bank,settings=settings)
  torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   env.reset(torch.arange(env.num_envs,device=env.device))
   action=nominal_action(env).expand(env.num_envs,-1).clone()
   finite=True
   step_started=time.monotonic()
   for _ in range(64):
    obs,reward,done,extra=env.step(action)
    finite=finite and bool(torch.isfinite(reward).all()) and all(bool(torch.isfinite(v).all()) for v in obs.values())
   step_elapsed=time.monotonic()-step_started
  counts=collections.Counter(str(p.GetPath()).split('/')[3] for p in env.sim.stage.Traverse() if str(p.GetPath()).endswith('/Robot'))
  checks=dict(finite=finite,actor35_critic83=obs['policy'].shape==(256,35) and obs['critic'].shape==(256,83),one_robot_per_environment=len(counts)==256 and all(v==1 for v in counts.values()),loop_count=env.startup_report['loop_constraints']==6*256,native_wheels=env.startup_report['training_identified_wheels']['native_properties_verified'])
  entry.base.write_json(out/f'{index:03d}_{profile}'/'native_startup.json',env.startup_report)
  result['contexts'].append(dict(profile=profile,strength=strength,checks=checks,all_checks_pass=all(checks.values()),physical_steps=64*env.decimation,stepping_elapsed_s=step_elapsed,mean_policy_step_s=step_elapsed/64.,torch_cuda_peak_allocated_b=torch.cuda.max_memory_allocated(),torch_cuda_peak_reserved_b=torch.cuda.max_memory_reserved(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  entry.base.write_json(out/'result.json',result)
  print('FULL256',profile,result['contexts'][-1],flush=True)
  env.close();env=None;SimulationContext.clear_instance();gc.collect();torch.cuda.empty_cache()
 result['all_checks_pass']=all(c['all_checks_pass'] for c in result['contexts'])
 result['status']='completed_software_only' if result['all_checks_pass'] else 'failed_checks'
except BaseException:
 result.update(status='failed',error=traceback.format_exc())
 raise
finally:
 result.update(elapsed_s=time.monotonic()-start,source_mismatches_at_exit=[n for n,h in hashes.items() if entry.base.digest(root/n)!=h],script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 entry.base.write_json(out/'result.json',result)
 if env is not None:env.close()
 if launcher is not None:launcher.app.close()
