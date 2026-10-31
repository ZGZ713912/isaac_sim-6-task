#!/usr/bin/env python3
"""Zero-PPO native course wiring and real cold-review lifecycle probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--profiles',nargs='+',choices=('nominal','requested'),default=['nominal','requested'])
    p.add_argument('--num-envs',type=int,default=32)
    args=p.parse_args()
    if not 20 <= args.num_envs <= 80:p.error('Use20..80 environments')
    args.output.mkdir(parents=True,exist_ok=False)
    from train_prone_motion import configuration
    from train_scratch_standing import write_json,digest
    plan=json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())
    report=dict(schema='global-curriculum-native-probe-v1',status='starting',device=args.device,
        new_PPO=0,optimizer_step_calls=0,checkpoint_reads=[],profiles=[],learned_behavior=False,
        full_capability_qualified=False,hardware_ready=False,phase_state_injection='synthetic_wiring_only',
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),
            ROOT/'contracts/wheel_leg_prone_motion_v4.json',ROOT/'scripts/train_prone_motion.py',
            ROOT/'scripts/evaluate_global_curriculum.py',ROOT/'scripts/evaluate_recovery_mastery.py',
            ROOT/'scripts/evaluate_prone_motion.py',ROOT/'src/wheeled_tasks/chassis/global_curriculum.py',
            ROOT/'src/wheeled_algo/global_curriculum.py',ROOT/'src/wheeled_tasks/chassis/prone_motion_env.py']})
    launcher=env=None
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(dict(headless=True,device=args.device,enable_cameras=False))
        import torch
        from isaaclab.sim import SimulationContext
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum,SKILLS
        from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_algo.global_curriculum import GlobalCurriculumHook
        from evaluate_prone_motion import probe
        import train_scratch_standing as base
        base.torch=torch;torch.set_num_threads(4)
        def forbidden(*a,**kw):
            report['checkpoint_reads'].append(str(a[0] if a else kw.get('f')))
            raise RuntimeError('Native course probe cannot read checkpoint weights')
        torch.load=forbidden
        for profile in args.profiles:
            directory=args.output/profile
            cfg,settings,bank=configuration(directory,plan,profile=profile,strength=float(profile!='nominal'),
                num_envs=args.num_envs,device=args.device,performance_backend='cuda_graph' if args.device.startswith('cuda') else 'eager')
            manifest=json.loads((ROOT/cfg['asset_directory']/'manifest.json').read_text())
            env=ProneMotionEnv(cfg,manifest,load_chassis_control(ROOT/cfg['control_math_source']),ROOT,
                stage_name='speed',num_envs=args.num_envs,device=args.device,seed=plan['seed'],level=1.,
                run_role='train',bank=bank,settings=settings)
            write_json(directory/'startup.json',env.startup_report)
            checks={};wiring=probe(env,plan)
            write_json(directory/'training_native_gate.json',wiring)
            checks['training_native_gate']=wiring['all_checks_pass']
            nominal=env.bank_joints[env.nominal_pose_index]
            checks['literal_urdf_nominal_birth']=bool(torch.allclose(env.robot.data.joint_pos.torch,
                nominal[None].expand(env.num_envs,-1),atol=1e-6,rtol=0.))

            class ConstantPolicy(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.anchor=torch.nn.Parameter(torch.zeros((),device=env.device))
                    self.register_buffer('action',base.nominal_action(env).expand(env.num_envs,-1).clone())
                def forward(self,observations):return self.action
            actor=ConstantPolicy();critic=torch.nn.Linear(1,1).to(env.device)
            optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
            def no_step(*a,**kw):
                report['optimizer_step_calls']+=1;raise RuntimeError('Cold native probe attempted optimization')
            optimizer.step=no_step
            fresh=[]
            def act(obs):
                current=env.get_observations()
                fresh.append(all(torch.equal(obs[k],current[k]) for k in current.keys()))
                return obs
            runner=SimpleNamespace(env=env,device=args.device,logger=SimpleNamespace(writer=None,
                cur_reward_sum=torch.ones(env.num_envs,device=env.device),cur_episode_length=torch.ones(env.num_envs,device=env.device)),
                alg=SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer,storage=SimpleNamespace(step=0),act=act))
            hook=GlobalCurriculumHook(runner,env.global_curriculum,completed=lambda:250,output=directory/'cold_hook')
            stale=env.get_observations().clone()
            with torch.inference_mode():runner.alg.act(stale)
            checks['cold_hook_refreshes_observations']=all(fresh)
            checks['cold_hook_clears_partial_episode_totals']=bool((runner.logger.cur_reward_sum==0).all() and (runner.logger.cur_episode_length==0).all())
            checks['cold_hook_no_optimizer_updates']=not optimizer.state and report['optimizer_step_calls']==0
            checks['cold_hook_restores_actor_training_mode']=actor.training
            hook.close()
            phases=[]
            states=[('nominal',{}),('foundation',dict(standing=1)),
                ('low_motion',dict(standing=1,straight=2,static_height=2)),
                ('high_motion',dict(standing=1,straight=3,static_height=2,turn=2,spin=2,braking=1)),
                ('all_nominal',{s:len(plan['global_curriculum']['skills'][s]['levels']) for s in SKILLS})]
            for name,levels in states:
                recovery=RecoveryMastery(plan['multi_pose_recovery']['mastery_gate'])
                c=GlobalCurriculum(plan,recovery);c.levels.update(levels)
                if name in ('low_motion','high_motion','all_nominal'):recovery.unlocked_at=250
                env.global_curriculum=c;env.recovery_sampler.mastery=recovery
                env.actor_updates=9999
                with torch.inference_mode():
                    env.reset(torch.arange(env.num_envs,device=env.device))
                    at6=env.task_commands(torch.full((env.num_envs,),6.,device=env.device))
                    at9=env.task_commands(torch.full((env.num_envs,),9.,device=env.device))
                    combined=env.course_moving_height
                    ok=dict(finite_commands=bool(torch.isfinite(at9).all()),
                        no_time_based_randomization=c.domain_context()==('nominal',0.),
                        combined_height_delayed=bool((at6[combined,2]==.305).all()),
                        combined_height_at_motion_plateau=bool(torch.allclose(at9[combined,2],env.course_targets[combined,2])))
                    if name=='nominal':ok['all_standing']=all(r=='stand' for r in env.course_roles) and not bool(env.recovery_lane.any())
                    if name=='foundation':
                        info=[env.poses[int(i)].get('recovery',{}) for i in env.pose_index[env.recovery_lane]]
                        ok['belly_only']=bool(info) and all(r.get('family')=='belly_down' for r in info)
                    for _ in range(16):
                        obs,reward,done,extra=env.step(actor.action)
                        ok['finite_native_step']=ok.get('finite_native_step',True) and bool(torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
                    phases.append(dict(phase=name,checks=ok,all_checks_pass=all(ok.values()),
                        course_skills=sorted({r['curriculum_skill'] for r in env.course_cases}),
                        combined_height_lanes=int(combined.sum()),synthetic_state_only=True))
            checks['all_synthetic_phase_wiring']=all(p['all_checks_pass'] for p in phases)
            result=dict(profile=profile,checks=checks,all_checks_pass=all(checks.values()),phases=phases,
                gpu_dynamics_enabled=env.startup_report['gpu_dynamics_enabled'],optimizer_updates=0)
            report['profiles'].append(result);write_json(directory/'phase_wiring.json',result)
            env.close();env=None;SimulationContext.clear_instance()
        report.update(status='completed',passed=all(p['all_checks_pass'] for p in report['profiles']))
        if not report['passed']:raise RuntimeError('Global native wiring checks failed')
    except BaseException:
        import traceback
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        write_json(args.output/'report.json',report)
        if env is not None:env.close()
        if launcher is not None:launcher.app.close()


if __name__=='__main__':main()
