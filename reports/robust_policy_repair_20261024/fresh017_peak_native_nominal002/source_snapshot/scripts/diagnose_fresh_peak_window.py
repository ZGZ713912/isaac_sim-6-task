#!/usr/bin/env python3
"""Short frozen-source native peak attribution; never a full capability score.

Run only after the exact learner exits. Uses the full fixed case catalogue but
observes the first two seconds to separate passive placement from active RL.
No PPO, controller substitution, mid-episode reset or reward modification.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
ROOT=Path(__file__).resolve().parents[1]
from evaluate_fresh_full_suite import authenticate,digest,prepare,verify_actor_chain,write_json


def process_live(progress):
    try:
        ticks=int(Path(f"/proc/{progress['pid']}/stat").read_text().rsplit(') ',1)[1].split()[19])
        return ticks==progress['start_ticks']
    except FileNotFoundError:
        return False


def passive_clock_candidate(task, enabled):
    """Prepare an explicitly declared startup comparison, never train with it."""
    if not enabled:
        return task,None,None
    source=ROOT/'reports/robust_policy_repair_20261024/short_passive_activation_candidate001.py'
    module_spec=importlib.util.spec_from_file_location('declared_short_passive_candidate',source)
    candidate=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(candidate)
    result,declaration=candidate.prepare(task)
    declaration.update(source_file=str(source),source_sha256=digest(source),
        evaluation_only=True,original_native_guards_and_enable_predicate=True)
    return result,declaration,source


def engineering_summaries(arrays):
    """Read native1ms guard maxima reduced over each owning5ms interval."""
    import numpy as np
    valid = arrays['valid']
    shape = valid.shape
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError('Need the original Boolean native validity mask')
    names = ('gap', 'impact', 'impulse', 'terminated', 'mechanical', 'closure_gap', 'impact_bad')
    if any(arrays[key].shape != shape for key in names):
        raise ValueError('Native engineering samples must share their owning interval')
    phases = dict(passive_placement=arrays['passive'],
        active_recovery=~arrays['passive'] & arrays['recovery'],
        ordinary_active_control=~arrays['passive'] & ~arrays['recovery'])
    result = {}
    for name, phase in phases.items():
        selected = valid & phase
        row = dict(valid_policy_intervals=int(selected.sum()))
        for key, unit in (('gap', 'm'), ('impact', 'n'), ('impulse', 'ns')):
            values = arrays[key][selected]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite original native guard observation')
            row[key+'_peak_'+unit] = float(values.max()) if len(values) else None
        for key in ('terminated', 'mechanical', 'closure_gap', 'impact_bad'):
            if arrays[key].dtype != np.bool_:
                raise ValueError('Need original native Boolean termination flags')
            row[key+'_case_count'] = int((selected & arrays[key]).any(0).sum())
        result[name] = row
    return result


def summarize(arrays,cases):
    import numpy as np
    dq,tau,valid=(arrays[name] for name in ('dq','torque','valid'))
    if dq.shape!=tau.shape or dq.shape[1:]!=(5,len(cases),4) or valid.shape!=(len(dq),len(cases)):
        raise ValueError('Need every original physical sample and its pre-interval phase')
    if not np.isfinite(dq).all() or not np.isfinite(tau).all():
        raise ValueError('Nonfinite native physical witness')
    phases={'passive_placement':arrays['passive'],
        'active_recovery':~arrays['passive']&arrays['recovery'],
        'ordinary_active_control':~arrays['passive']&~arrays['recovery']}
    groups=[]
    roles=sorted({case['capability'] for case in cases})
    for role in ['all']+roles:
        lanes=np.array([role=='all' or case['capability']==role for case in cases])
        for phase,mask in phases.items():
            eligible=valid&mask&lanes[None]
            active=np.broadcast_to(eligible[:,None,:,None],dq.shape)
            magnitude=np.where(active,np.abs(dq),-np.inf)
            row=dict(role=role,phase=phase,physical_leg_motor_samples=int(active.sum()))
            if active.any():
                frame,substep,lane,leg=np.unravel_index(np.argmax(magnitude),dq.shape)
                row.update(peak_leg_speed_rad_s=float(magnitude[frame,substep,lane,leg]),
                    samples_at_or_above_15_rad_s=int((magnitude>=15.).sum()),
                    samples_at_or_above_30_rad_s=int((magnitude>=30.).sum()),
                    samples_at_or_above_50_rad_s=int((magnitude>=50.).sum()),
                    witness=dict(policy_interval=int(frame),physical_substep_one_based=int(substep+1),
                        physical_time_s=float(frame*.005+(substep+1)*.001),lane=int(lane),leg_policy_index=int(leg),
                        case=cases[lane],signed_leg_speed_rad_s=float(dq[frame,substep,lane,leg]),
                        actual_leg_torque_nm_same_1ms=float(tau[frame,substep,lane,leg]),
                        all_leg_speed_rad_s_same_1ms=dq[frame,substep,lane].tolist(),
                        all_leg_torque_nm_same_1ms=tau[frame,substep,lane].tolist(),
                        policy_end_gravity=arrays['gravity'][frame,lane].tolist(),
                        policy_end_height_m=float(arrays['height'][frame,lane])))
            groups.append(row)
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spec',type=Path,default=ROOT/'contracts/wheel_leg_full_suite_v1.json')
    parser.add_argument('--randomization-study',type=Path,default=ROOT/'contracts/wheel_leg_randomization_study_v2.json')
    parser.add_argument('--randomization-profile',choices=['nominal','requested'],required=True)
    parser.add_argument('--seed',type=int,default=271301)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--short-passive-activation',action='store_true',
        help='Explicit unadopted5ms startup comparison; original native first interval remains passive')
    args=parser.parse_args()
    args.checkpoint=args.checkpoint.resolve();args.output=args.output.resolve()
    progress=json.loads((args.checkpoint.parent/'progress.json').read_text())
    if not args.preflight_only:
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if process_live(progress) or str(progress['pid']) in gpu.splitlines():
            raise ValueError('Wait for the original learner and its GPU identity to exit')
    args.output.mkdir(parents=True,exist_ok=False)
    original_study=args.randomization_study.resolve()
    study=json.loads(original_study.read_text());study['nominal_fraction']=0.
    derived=args.output/'all_randomized_study.json';write_json(derived,study)
    args.randomization_study=derived
    bundle=authenticate(args.checkpoint)
    prepared=prepare(args,bundle)
    cfg,task,spec,requirements,cases,domain,overrides=prepared
    task,clock_intervention,clock_source=passive_clock_candidate(task,args.short_passive_activation)
    if clock_intervention is not None:
        write_json(args.output/'passive_clock_intervention.json',clock_intervention)
        write_json(args.output/'task_contract.json',task)
    sources={str(Path(__file__).relative_to(ROOT)):digest(__file__)}
    for name in ('scripts/evaluate_fresh_full_suite.py','scripts/wheel_leg_full_suite.py',
            'scripts/scratch_domain_evaluation.py','scripts/record_scratch_physics.py'):
        sources[name]=digest(ROOT/name)
    if clock_source is not None:
        sources[str(clock_source.relative_to(ROOT))]=digest(clock_source)
    for name,expected in sources.items():
        frozen=args.output/'source_snapshot'/name
        frozen.parent.mkdir(parents=True,exist_ok=True)
        frozen.write_bytes((ROOT/name).read_bytes())
        if digest(frozen)!=expected:
            raise ValueError('Diagnostic source changed during its snapshot')
    receipt=dict(schema='fresh-native-short-peak-attribution-v1',status='prepared',
        observed_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),
        completed_updates=bundle['infos']['completed_updates'],original_training_pid=progress['pid'],
        source_sha256=sources,frozen_training_source_sha256=bundle['infos']['source_hashes'],
        seed=args.seed,profile=args.randomization_profile,case_count=len(cases),
        full_suite_spec_sha256=digest(args.spec),native_policy_steps=0,diagnostic_seconds=2.,
        study_parent_sha256=digest(original_study),derived_study_sha256=digest(derived),
        study_only_changed_field='nominal_fraction:0.25 to0; all requested cases actually randomized',
        full_suite_qualification=False,new_PPO=0,optimizer_constructed=False,
        exact_training_event_replay=False,
        hardware_ready=False,short_window_not_behavior_pass=True,
        passive_clock_intervention=clock_intervention)
    write_json(args.output/'receipt.json',receipt)
    if args.preflight_only:
        return
    import numpy as np
    import torch
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(dict(headless=True,device=args.device,enable_cameras=False,
        kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
    env=logger=None
    try:
        from rsl_rl.utils import resolve_class
        from rsl_rl.utils.logger import Logger
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.root_kinematics import update_robot_data
        from wheeled_tasks.chassis.grounded_standing_reset import install as ground
        from wheeled_tasks.chassis.physical_standing_reward import install as physical
        from wheeled_tasks.chassis.randomization_study import runtime_samples
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from record_scratch_physics import record
        if 'include_motion' not in inspect.signature(ground).parameters:
            raise ValueError('This diagnostic requires the authenticated all-upright reset helper')

        class PeakEnv(RobustGetupEnv):
            def __init__(self,*positional,**keywords):
                self.peak_ready=False
                super().__init__(*positional,**keywords)
                self.peak_ready=True
                self.reset(torch.arange(self.num_envs,device=self.device))
                ground(self,include_motion=True)

            def reset(self,ids):
                if not self.peak_ready:
                    return super().reset(ids)
                self.fixed_cases=cases
                super().reset(ids)
                root=self.robot.data.root_link_pose_w.torch[ids].clone()
                angle=root.new_tensor([cases[i]['pitch_deg'] for i in ids.tolist()])*torch.pi/180.
                x,y,z,w=root[:,3:].unbind(-1);s,c=(angle/2).sin(),(angle/2).cos()
                root[:,3:]=torch.stack((x*c-z*s,y*c+w*s,z*c+x*s,w*c-y*s),-1)
                self.robot.write_root_link_pose_to_sim_index(root_pose=root,env_ids=ids)
                update_robot_data(self,self.dt)
                self.previous_potential[ids]=self.potential()[ids]

        torch.set_num_threads(4);torch.manual_seed(args.seed)
        manifest_file=ROOT/cfg['asset_directory']/'manifest.json'
        if digest(manifest_file)!=cfg['asset_manifest_sha256']:
            raise ValueError('Mechanical manifest changed')
        env=PeakEnv(cfg,json.loads(manifest_file.read_text()),load_chassis_control(ROOT/cfg['control_math_source']),ROOT,
            stage_name='speed',num_envs=len(cases),device=args.device,seed=args.seed,level=1.,run_role='train',
            bank=bundle['bank_file'],settings=task)
        physical(env,speed_weight=0.,contact_weight=0.,torque_weight=0.)
        physics=record(env)
        if physics['physical_dt_s']!=.001 or physics['policy_dt_s']!=.005 or physics['articulation_iterations']!={
                'physxArticulation:solverPositionIterationCount':64,'physxArticulation:solverVelocityIterationCount':32}:
            raise ValueError('Native physical settings changed')
        samples=runtime_samples(env)
        if any(value is not (args.randomization_profile=='requested') for value in samples['samples']['enabled']):
            raise ValueError('Actual native domain masks differ from the diagnostic profile')
        receipt.update(status='native_evaluating',physics_runtime=physics,actual_domain_samples=samples,
            actual_initial_grounding=deepcopy(env.grounded_standing_reset),
            initial_root_local_xyzw=torch.cat((env.robot.data.root_link_pose_w.torch[:,:3]-env.origins,
                env.robot.data.root_link_pose_w.torch[:,3:]),dim=-1).cpu().tolist())
        write_json(args.output/'receipt.json',receipt)
        cls,actor_cfg=resolve_class(bundle['agent']['actor'])
        verify_actor_chain(cls,bundle['archive'],bundle['actor_source_chain'])
        observation=env.get_observations()
        actor=cls(observation,bundle['agent']['obs_groups'],'actor',6,**actor_cfg).to(args.device).eval()
        actor.load_state_dict(bundle['state']['actor_state_dict'],strict=True)
        before={key:value.detach().cpu().clone() for key,value in actor.state_dict().items()}
        scalar_diagnostics=('gap','impact','impulse','terminated','mechanical','closure_gap','impact_bad')
        trace={name:[] for name in ('valid','passive','recovery','dq','torque','gravity','height',
            'velocity','omega','local_position',*scalar_diagnostics)}
        with torch.inference_mode():
            for step in range(400):
                for name,value in dict(valid=~env.finished,passive=env.passive,recovery=env.recovery_active).items():
                    trace[name].append(value.detach().cpu().clone())
                observation,_,_,extras=env.step(actor(observation))
                state=extras['diagnostics'];physical_state=env.physical_standing_last
                for name,value in dict(dq=physical_state['dq'],torque=physical_state['torque'],
                        gravity=state['gravity'],height=state['height']).items():
                    trace[name].append(value.detach().cpu().clone())
                for name in ('velocity','omega','local_position',*scalar_diagnostics):
                    trace[name].append(state[name].detach().cpu().clone())
                if (step+1)%100==0:
                    receipt['native_policy_steps']=step+1;write_json(args.output/'receipt.json',receipt)
        if any(not torch.equal(before[key],value.detach().cpu()) for key,value in actor.state_dict().items()):
            raise ValueError('Read-only inference changed actor or normalization buffers')
        arrays={name:torch.stack(values).numpy() for name,values in trace.items()}
        np.savez_compressed(args.output/'native_peak_trace.npz',**arrays)
        groups=summarize(arrays,cases)
        engineering=engineering_summaries(arrays)
        logger=Logger(log_dir=str(args.output),cfg=bundle['agent'],env_cfg=env.cfg,num_envs=env.num_envs,
            is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device=args.device)
        logger.init_logging_writer();expected={}
        for group in groups:
            for key,value in group.items():
                if isinstance(value,(int,float)):
                    tag=f"PeakWindow/{group['role']}/{group['phase']}/{key}"
                    expected[tag]=value;logger.writer.add_scalar(tag,value,receipt['completed_updates'])
        for phase,values in engineering.items():
            for key,value in values.items():
                if value is not None:
                    tag=f'PeakWindow/engineering/{phase}/{key}'
                    expected[tag]=value;logger.writer.add_scalar(tag,value,receipt['completed_updates'])
        logger.writer.flush();logger.writer.close();logger=None
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(args.output),size_guidance={'scalars':0}).Reload()
        for tag,value in expected.items():
            rows=events.Scalars(tag)
            if len(rows)!=1 or rows[0].step!=receipt['completed_updates'] or not np.isclose(rows[0].value,value,rtol=2e-6,atol=2e-8):
                raise ValueError('Peak phase RSL readback failed: '+tag)
        if digest(args.checkpoint)!=receipt['checkpoint_sha256'] or any(digest(ROOT/n)!=v for n,v in sources.items()):
            raise ValueError('A bound source or checkpoint changed')
        result=dict(schema='fresh-native-short-peak-attribution-result-v1',groups=groups,
            native_engineering=engineering,
            checkpoint_sha256=receipt['checkpoint_sha256'],native_policy_steps=400,physical_steps=2000,
            trace_sha256=digest(args.output/'native_peak_trace.npz'),RSL_values_read_back=len(expected),
            whole_trajectory_qualified=False,new_PPO=0,optimizer_constructed=False,hardware_ready=False,
            interpretation='Only the first2s of independent fixed cases, not an exact replay of the training peak or its RNG/lane history. Phase is captured before the5ms interval; actual leg dq and applied motor torque are paired at every1ms boundary. Passive spring/contact motion is not an actor torque command. No recovery/standing/full13 qualification follows from this diagnostic.')
        write_json(args.output/'result.json',result)
        receipt.update(status='completed_diagnostic_unqualified',native_policy_steps=400,
            result_sha256=digest(args.output/'result.json'),trace_sha256=result['trace_sha256'],actor_state_unchanged=True)
        write_json(args.output/'receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],global_phase_groups=[r for r in groups if r['role']=='all'])))
    except BaseException:
        receipt.update(status='failed',error=traceback.format_exc());write_json(args.output/'receipt.json',receipt)
        raise
    finally:
        if logger is not None:logger.writer.close()
        if env is not None:env.close()
        launcher.app.close()


if __name__=='__main__':main()
