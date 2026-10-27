#!/usr/bin/env python3
"""Zero-PPO native reset-isolation probe, with no actor or optimizer construction."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--expect',choices=['legacy-leak','isolated'],required=True)
    parser.add_argument('--seed',type=int,default=271504)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    import train_scratch_standing as training
    recipe=SimpleNamespace(output=output,bank=args.bank.resolve(),all_heights_after=512,
        precise_objective=True,standing_reward_shape='gaussian',standing_failures=True,
        wheel_velocity_increment=True,separate_leg_velocity_feedback=True,static_load_initialization=False,
        contact_break_cost=0.,leg_target_rate_cost=None,num_envs=24,rollout_steps=128,seed=args.seed,
        randomization_study=ROOT/'contracts/wheel_leg_randomization_study_v2.json',
        randomization_profile='requested',randomization_strength=1.,normalize_observations=True,
        desync_episodes=False)
    cfg,task=training.prepare(recipe)
    sources=training.sources()
    sources['scripts/probe_partial_reset_sensor_clock.py']=sha(__file__)
    for name,digest in sources.items():
        source=ROOT/name;destination=output/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('Source changed while freezing '+name)
        destination.write_bytes(data)
    training.write_json(output/'source_hashes.json',sources)
    receipt=dict(schema='native-partial-reset-sensor-clock-probe-v1',status='starting',
        observed_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),seed=args.seed,
        expected=args.expect,new_PPO=0,actor_constructed=False,optimizer_constructed=False,
        checkpoint_reads=[],physics_steps=0,source_sha256=sources,hardware_ready=False,
        behavior_qualification=False)
    launcher=env=writer=None
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(dict(headless=True,device='cuda:0',enable_cameras=False,
            kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
        import torch
        from wheeled_tasks.chassis.scratch_standing_env import ScratchStandingEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.randomization_study import runtime_samples
        from record_scratch_physics import record
        from rsl_rl.utils.logger import Logger
        torch.set_num_threads(4);torch.manual_seed(args.seed)
        manifest=json.loads((ROOT/cfg['asset_directory']/'manifest.json').read_text())
        env=ScratchStandingEnv(cfg,manifest,load_chassis_control(ROOT/cfg['control_math_source']),ROOT,
            stage_name='speed',num_envs=24,device='cuda:0',seed=args.seed,level=1.,run_role='train',
            bank=recipe.bank,settings=task)
        action=training.nominal_action(env).repeat(env.num_envs,1)
        for _ in range(8):
            env.step(action)
        receipt['physics_steps']=8*env.decimation
        env.get_observations()
        tick=env.tick;cursor=env.domain.sensor_cursor
        lane=env.num_envs-1
        untouched=torch.arange(lane,device=env.device)
        before=env.domain.observed[untouched].clone()
        history=env.domain.sensor_history[:,untouched].clone()
        position=env.robot.data.root_link_pose_w.torch[untouched].clone()
        joint=env.robot.data.joint_pos.torch[untouched].clone()
        # This is the real ScratchStandingEnv + grounded final-pose reset chain,
        # followed by the same observation refresh as a training failure.
        env.reset(torch.tensor([lane],device=env.device))
        env.history.last_tick=None
        env.get_observations()
        after=env.domain.observed[untouched].clone()
        changed=(after!=before).any(-1)
        same_root=torch.equal(position,env.robot.data.root_link_pose_w.torch[untouched])
        same_joint=torch.equal(joint,env.robot.data.joint_pos.torch[untouched])
        checks=dict(same_physical_tick=env.tick==tick,unreset_root_state_unchanged=same_root,
            unreset_joint_state_unchanged=same_joint,
            source_frozen=all(sha(ROOT/name)==digest for name,digest in sources.items()))
        if args.expect=='legacy-leak':
            checks['expected_observation_leak_reproduced']=bool(changed.any())
        else:
            checks.update(untouched_observation_unchanged=not bool(changed.any()),
                untouched_history_unchanged=torch.equal(history,env.domain.sensor_history[:,untouched]),
                global_sensor_cursor_unchanged=cursor==env.domain.sensor_cursor)
        receipt.update(status='completed_unqualified',tick=tick,reset_lane=lane,
            untouched_lane_ids=untouched.cpu().tolist(),untouched_changed=changed.cpu().tolist(),
            maximum_untouched_sensor_delta=float((after-before).abs().max()),
            sensor_cursor_before=cursor,sensor_cursor_after=env.domain.sensor_cursor,
            configured_sensor_delay_ticks=env.domain.sensor_delay.cpu().tolist(),
            untouched_before=before.cpu().tolist(),untouched_after=after.cpu().tolist(),
            actual_domains=runtime_samples(env),physics_runtime=record(env),checks=checks,
            all_checks_pass=all(checks.values()),
            scope='Native same-tick neighbor reset only; no learned policy or robustness qualification.')
        # Keep diagnostic measurements in the existing RSL event format.
        from wheeled_algo.scratch_standing import agent_config
        logger=Logger(log_dir=str(output),cfg=agent_config(args.seed,'cuda:0',128),env_cfg=env.cfg,
            num_envs=env.num_envs,is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cuda:0')
        logger.init_logging_writer();writer=logger.writer
        for tag,value in dict(untouched_changed_fraction=float(changed.float().mean()),
                maximum_untouched_sensor_delta=receipt['maximum_untouched_sensor_delta'],
                physical_time_advanced_s=(env.tick-tick)*env.policy_dt).items():
            writer.add_scalar('ProbeSensorReset/'+tag,value,0)
        writer.flush()
        if not receipt['all_checks_pass']:
            raise AssertionError('Native reset isolation failed: '+str(checks))
        print(json.dumps({k:receipt[k] for k in ['status','checks','maximum_untouched_sensor_delta',
            'sensor_cursor_before','sensor_cursor_after','physics_steps']}))
    except BaseException:
        receipt.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        training.write_json(output/'receipt.json',receipt)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__=='__main__':
    main()
