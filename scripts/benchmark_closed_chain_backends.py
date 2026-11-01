"""Bounded mechanics comparison, not an RL sampler or behavior qualification.

PhysX exports the actual standing reset. Both engines then use identical
200 Hz explicit PD, gas forces, measured leg friction, wheel delay/envelope,
gimbal motion and a declared regularized wheel baseline. The latter replaces
native PhysX wheel stiction in BOTH arms because installed Kamino maps
joint_friction to viscous damping. No production defaults are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend',choices=('physx','kamino'),required=True)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--output',type=Path,required=True)
p.add_argument('--snapshot',type=Path)
p.add_argument('--num-envs',type=int,default=256)
p.add_argument('--position-iterations',type=int,default=48)
p.add_argument('--velocity-iterations',type=int,default=24)
p.add_argument('--admm-iterations',type=int,default=200)
p.add_argument('--tolerance',type=float,default=1e-4)
p.add_argument('--contacts-per-world',type=int,default=128)
p.add_argument('--newton-contacts-per-world',type=int,default=128,
    help='Newton proximity-contact capacity, independently of active Kamino contacts')
p.add_argument('--warmstart-mode',choices=('containers','none','internal'),default='containers')
p.add_argument('--scenario',choices=('standing','drop'),default='standing')
p.add_argument('--rollouts',type=int,default=5)
p.add_argument('--warmup-rollouts',type=int,default=2)
p.add_argument('--steps',type=int,default=96,help='Physics calls, 96 = one 24 x 4 rollout')
p.add_argument('--no-graph',action='store_true')
p.add_argument('--sparse',action='store_true')
a=p.parse_args()
if min(a.num_envs,a.steps,a.rollouts,a.admm_iterations,a.contacts_per_world,
       a.newton_contacts_per_world)<1 or a.warmup_rollouts<0:
    p.error('Counts must be positive')
out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
root=a.root.resolve(); sys.path[:0]=[str(root/'scripts'),str(root/'src')]
shutil.copyfile(__file__,out/'benchmark_as_run.py')
shutil.copyfile(root/'src/wheeled_tasks/chassis/kamino_closed_chain_model.py',out/'model_adapter_as_run.py')
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
os.environ['OPENBLAS_NUM_THREADS']='1'
result=dict(schema='closed-chain-backend-mechanics-v1',status='starting',args={
    k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
    new_PPO=0,checkpoint_reads=[],behavior_qualified=False,
    scope='200Hz mechanics including explicit control, no observations/reward/PPO/task resets',
    wheel_friction='Both arms: native joint friction zero; measured dynamic floor * clamp(dq/0.01,-1,1), with impulse cap; no static stiction',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def save(): (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
save()
launcher=env=None
try:
    if a.backend=='physx':
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(dict(headless=True,device='cuda:0',enable_cameras=False,
            kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
    import numpy as np
    import torch
    import warp as wp
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(4)
    torch.manual_seed(106); np.random.seed(106)
    device='cuda:0'; dt=.005
    if a.backend=='kamino':
        # Both libraries share buffers for feedback and effort. All accesses
        # must use one CUDA stream, including graph replay and eager resets.
        wp.init()
        shared_torch_stream=torch.cuda.Stream()
        torch.cuda.set_stream(shared_torch_stream)
        shared_warp_stream=wp.stream_from_torch(shared_torch_stream)
        wp.set_stream(shared_warp_stream)
    from wheeled_tasks.chassis.v5_control import V5Control
    from wheeled_tasks.chassis.actuator_response import ActuatorResponse
    from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
    from wheeled_tasks.chassis.leg_motor_friction import LegMotorFriction
    from wheeled_tasks.chassis.gimbal_motion import GimbalMotion
    from wheeled_tasks.chassis.tensor_cuda_graph import TensorCudaGraph
    from wheeled_tasks.chassis.standing_support import wheel_bottom_clearance
    last_clearance=[None]

    class Controller:
        def __init__(self,cfg,manifest,names,q0):
            self.cfg,self.names=cfg,names
            directory=root/cfg['asset_directory']
            from wheeled_tasks.chassis.control_contract import load_chassis_control
            self.v5=V5Control(manifest,json.loads((directory/'model_spec.json').read_text()),
                json.loads((directory/'fit_10mpa.json').read_text()),
                load_chassis_control(root/cfg['control_math_source']),cfg['v5_control'],device,root=root)
            self.active=[names.index(n) for n in manifest['control_joint_names']]
            self.legs=[self.active[i] for i in self.v5.LEGS]
            self.wheel_ids=[self.active[i] for i in self.v5.WHEELS]
            self.spring=[names.index(n) for n in manifest['spring_joint_names']]
            self.gimbal_ids=[names.index(n) for n in cfg['gimbal_motion']['joint_names']]
            self.passive=[i for i,n in enumerate(names) if i not in self.active+self.gimbal_ids]
            self.target=q0[:,self.legs].clone()
            self.response=ActuatorResponse(a.num_envs,device,dt,cfg['actuator_response'])
            self.wheels=IdentifiedWheelResponse(a.num_envs,device,dt,cfg['identified_wheel_response'],root=root)
            self.friction=LegMotorFriction(a.num_envs,device,cfg['leg_motor_friction'],root=root)
            self.gimbal=GimbalMotion(a.num_envs,device,dt,cfg['gimbal_motion'],106)
            self.enabled=torch.ones(a.num_envs,device=device,dtype=torch.bool)
            self.zeros=torch.zeros(a.num_envs,2,device=device)
            native=self.wheels.native_parameters()
            self.floor=q0.new_tensor([native['dynamic_friction'][names[i]] for i in self.wheel_ids])
            self.phase=q0.new_tensor([1.,-1.,1.,-1.])
            ff=self.v5.gas_compensation
            # Pure math capture, same implementation used by production.
            ff.tensor_replay=TensorCudaGraph(ff._evaluate,q0[:,self.legs])
            ff.force_tensor_replay=TensorCudaGraph(ff._evaluate_force,self.v5.s0-q0[:,self.spring])
        def reset(self):
            self.response.reset(); self.wheels.reset()
            self.gimbal.rng.manual_seed(106 ^ 0x47494D42)
            self.gimbal.reset(torch.arange(a.num_envs,device=device))
        def __call__(self,q,dq,step,*,passive_damping):
            active_q,active_dq=q[:,self.active],dq[:,self.active]
            target=self.target + .008*np.sin(2*np.pi*step*dt)*self.phase
            tau=self.v5.motor_efforts(active_q,active_dq,target,self.zeros)
            tau=self.response.apply_torque(tau)
            tau.clamp_(min=-self.v5.current_motor_bounds,max=self.v5.current_motor_bounds)
            tau=self.wheels.apply_torque(tau,active_dq)
            effort=torch.zeros_like(q)
            effort[:,self.active]=tau
            effort[:,self.legs]+=self.friction.effort(active_dq[:,list(self.v5.LEGS)])
            speed=dq[:,self.wheel_ids]
            baseline=-torch.clamp(speed/.01,-1.,1.)*torch.minimum(
                self.floor,self.wheels.inertia*speed.abs()/dt)
            effort[:,self.wheel_ids]+=self.wheels.passive_effort+baseline
            effort[:,self.spring]=self.v5.spring_efforts(q[:,self.spring])
            effort[:,self.gimbal_ids]=self.gimbal.step(q[:,self.gimbal_ids],dq[:,self.gimbal_ids],self.enabled)
            if passive_damping:
                effort[:,self.passive]-=.002*dq[:,self.passive]
            return effort

    started=time.perf_counter()
    if a.backend=='physx':
        import train_prone_motion as entry
        from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
        from wheeled_tasks.chassis.prone_motion import validate
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
        from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
        from wheeled_tasks.chassis.root_kinematics import update_robot_data
        entry.base.torch=torch
        plan=resolve_plan(validate(json.loads((root/'contracts/wheel_leg_prone_motion_v4.json').read_text())),
            num_envs=a.num_envs,physics_hz=200,policy_hz=50,num_steps_per_env=24,seed=106,
            position_iterations=a.position_iterations,velocity_iterations=a.velocity_iterations)
        cfg,settings,bank=entry.configuration(out/'config_context',plan,num_envs=a.num_envs,seed=106)
        manifest=json.loads((root/cfg['asset_directory']/'manifest.json').read_text())
        mastery=RecoveryMastery(plan['multi_pose_recovery']['mastery_gate'])
        env=ProneMotionEnv(cfg,manifest,load_chassis_control(root/cfg['control_math_source']),root,
            stage_name='speed',num_envs=a.num_envs,device=device,seed=106,level=1.,run_role='train',
            bank=bank,settings=settings,recovery_mastery=mastery,global_curriculum=GlobalCurriculum(plan,mastery))
        with torch.inference_mode():
            env.reset(torch.arange(a.num_envs,device=device))
            env.sim.forward()
            update_robot_data(env,0.)
        names=list(env.robot.joint_names); body_names=list(env.robot.body_names)
        q0=env.robot.data.joint_pos.torch.clone()
        base0=env.robot.data.root_link_pose_w.torch.clone()
        if a.scenario=='drop': base0[:,2]+=.10
        zero_friction=torch.zeros_like(q0)
        env.robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=zero_friction,
            joint_dynamic_friction_coeff=zero_friction,joint_viscous_friction_coeff=zero_friction)
        # Snapshot is the shared physical initial condition, not a policy file.
        np.savez_compressed(out/'initial.npz',joint=q0.cpu().numpy(),base=base0.cpu().numpy())
        (out/'snapshot.json').write_text(json.dumps(dict(cfg=cfg,manifest=manifest,
            joint_names=names,body_names=body_names,scenario=a.scenario),indent=2)+'\n')
        result['native_startup']=env.startup_report
        result['joint_armatures']=wp.to_torch(env.robot.root_view.get_dof_armatures()).cpu().tolist()[0]
        controller=Controller(cfg,manifest,names,q0)
        def reset():
            env.robot.write_joint_state_to_sim_index(position=q0,velocity=torch.zeros_like(q0))
            env.robot.write_root_pose_to_sim_index(root_pose=base0)
            env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(a.num_envs,6,device=device))
            env.sim.forward()
            update_robot_data(env,0.)
            controller.reset()
        def step(i):
            q=env.robot.data.joint_pos.torch; dq=env.robot.data.joint_vel.torch
            effort=controller(q,dq,i,passive_damping=False)
            env.robot.set_joint_effort_target_index(target=effort)
            env.robot.write_data_to_sim()
            env.sim.step(render=False)
            update_robot_data(env,dt)
        def observe():
            last_clearance[0]=wheel_bottom_clearance(env.wheel_centers(),
                env.robot.data.body_link_pose_w.torch[:,env.wheel_ids,3:],env.origins[:,2:3])
            return env.closure_gap(),env.robot.data.root_link_pose_w.torch,env.robot.data.joint_vel.torch
    else:
        if a.snapshot is None: raise ValueError('Kamino requires a native PhysX reset snapshot')
        import newton
        from wheeled_tasks.chassis.kamino_closed_chain_model import import_robot
        snapshot=json.loads((a.snapshot/'snapshot.json').read_text())
        cfg,manifest=snapshot['cfg'],snapshot['manifest']
        data=np.load(a.snapshot/'initial.npz')
        if data['joint'].shape[0]<a.num_envs or snapshot['scenario']!=a.scenario:
            raise ValueError('Snapshot count or scenario differs')
        names=snapshot['joint_names']; body_names=snapshot['body_names']
        initial_joint=data['joint'][:a.num_envs].copy()
        initial_base=data['base'][:a.num_envs].copy()
        q0=torch.tensor(initial_joint,device=device); base0=torch.tensor(initial_base,device=device)
        robot,corrections,audit=import_robot(root/cfg['asset_directory'])
        result['frame_audit']=audit
        rb_names=[n.rsplit('/',1)[-1] for n in robot.body_label]
        joints={n.rsplit('/',1)[-1]:i for i,n in enumerate(robot.joint_label)}
        q_ids=[robot.joint_q_start[joints[n]] for n in names]
        u_ids=[robot.joint_qd_start[joints[n]] for n in names]
        controller=Controller(cfg,manifest,names,q0)
        native=controller.wheels.native_parameters()
        for n,armature in native['armature'].items():
            robot.joint_armature[robot.joint_qd_start[joints[n]]]=armature
        for s,b in enumerate(robot.shape_body):
            n=rb_names[b] if b>=0 else ''
            robot.shape_material_mu[s]=.7 if n in ('L_link3','R_link3') else .25
            robot.shape_material_restitution[s]=0.
        # Same 8m-cell 96m square triangle mesh as the native flat scene.
        from wheeled_tasks.chassis.getup_ground import flat_patch
        points,indices=flat_patch()
        mesh=newton.Mesh(points,indices,compute_inertia=False,is_solid=False)
        shape_cfg=newton.ModelBuilder.ShapeConfig(density=0.,mu=.5,restitution=0.,margin=0.,gap=0.)
        robot.add_shape_mesh(-1,mesh=mesh,cfg=shape_cfg,label='Terrain')
        builder=newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(a.num_envs): builder.add_world(robot)
        model=builder.finalize(skip_validation_joints=True)
        # Set before SolverKamino: otherwise its fallback sizes dense matrices
        # for every possible geometry contact and can overflow int32 at 256.
        model.rigid_contact_max=a.num_envs*a.contacts_per_world
        config=newton.solvers.SolverKamino.Config.from_model(model)
        config.use_fk_solver=True; config.use_collision_detector=False
        config.sparse_jacobian=a.sparse; config.sparse_dynamics=a.sparse
        config.padmm.max_iterations=a.admm_iterations
        config.padmm.primal_tolerance=a.tolerance
        config.padmm.dual_tolerance=a.tolerance
        config.padmm.compl_tolerance=a.tolerance
        config.padmm.warmstart_mode=a.warmstart_mode
        config.collect_solver_info=False; config.compute_solution_metrics=False
        solver=newton.solvers.SolverKamino(model,config=config)
        state=[model.state(),model.state()]; control=model.control()
        pipeline=newton.CollisionPipeline(model,broad_phase='explicit',
            rigid_contact_max=a.num_envs*a.newton_contacts_per_world)
        contacts=model.contacts(collision_pipeline=pipeline)
        result['model']=dict(worlds=model.world_count,bodies=model.body_count,joints=model.joint_count,
            shapes=model.shape_count,tree_dofs_per_world=20,ball_loops_per_world=6)
        shape_world=model.shape_world.numpy()
        pairs=model.shape_contact_pairs.numpy()
        result['model']['contact_pairs']=len(pairs)
        if len(pairs) and np.any(shape_world[pairs[:,0]]!=shape_world[pairs[:,1]]):
            raise ValueError('Cross-world collision pairs in the replicated model')
        result['model']['cross_world_pairs']=0
        result['model']['mass_per_world_kg']=float(model.body_mass.numpy().sum()/a.num_envs)
        if abs(result['model']['mass_per_world_kg']-manifest['total_mass_kg'])>1e-4:
            raise ValueError('Imported body mass differs from native manifest')
        result['solver_config']=str(config)
        seed_q=np.tile(np.asarray(robot.joint_q),(a.num_envs,1))
        seed_q[:,q_ids]=initial_joint; seed_q[:,:7]=initial_base
        # This probe resets complete, already closed body poses from tree FK.
        # Use the installed maximal-coordinate reset directly: the public
        # joint_q/base_q wrapper has different base-joint coordinate semantics.
        free_joints=[j for j,t in enumerate(robot.joint_type) if int(t)==4]
        if free_joints!=[0]: raise ValueError('Expected one leading free base joint')
        free_parent=np.asarray(robot.joint_X_p[0],dtype=float)
        free_child=np.asarray(robot.joint_X_c[0],dtype=float)
        if not np.allclose(free_parent[3:],[0,0,0,1],atol=1e-6) or not np.allclose(free_child,[0,0,0,0,0,0,1],atol=1e-6):
            raise ValueError('Unexpected free-joint frame; reset conversion must be reviewed')
        result['reset_boundary']=dict(free_parent_xyz=free_parent[:3].tolist(),
            method='Newton tree FK -> explicit origin/COM conversion -> Kamino maximal body-state reset',
            all_world_body_origin_pose_assertion_m=2e-6)
        initial_q=wp.array(seed_q.flatten(),dtype=wp.float32,device=device)
        initial_u=wp.zeros(model.joint_dof_count,dtype=wp.float32,device=device)
        zero_body_u=wp.zeros(model.body_count,dtype=wp.spatial_vectorf,device=device)
        qindex=torch.tensor(q_ids,device=device); uindex=torch.tensor(u_ids,device=device)
        bodyindex=torch.tensor([rb_names.index(n) for n in body_names],device=device)
        baseindex=rb_names.index('base_link')
        spec=json.loads((root/cfg['asset_directory']/'model_spec.json').read_text())
        spec_bodies={b['name']:b for b in spec['bodies']}
        wheel_names=('L_link3','R_link3')
        wheel_bodies=torch.tensor([rb_names.index(n) for n in wheel_names],device=device)
        wheel_offsets=torch.tensor(np.asarray([corrections[rb_names.index(n)].T@
            np.asarray([spec_bodies[n]['collisions'][0]['origin'][i][3] for i in range(3)])
            for n in wheel_names]),device=device,dtype=torch.float32)
        wheel_axes=torch.tensor(np.asarray([corrections[rb_names.index(n)].T@np.asarray([0.,0.,1.])
            for n in wheel_names]),device=device,dtype=torch.float32)
        ids=torch.tensor([[rb_names.index(c[f'body{e}']) for e in (0,1)] for c in spec['constraints']],device=device)
        locals_=torch.tensor(np.asarray([[corrections[rb_names.index(c[f'body{e}'])].T@
            np.asarray(c[f'local_pos{e}_m']) for e in (0,1)] for c in spec['constraints']]),
            device=device,dtype=torch.float32)
        current=[0]
        def physics(k):
            state[k].clear_forces()
            pipeline.collide(state[k],contacts)
            solver.step(state[k],state[1-k],control,contacts,dt)
        def reset():
            current[0]=0
            wp.copy(state[0].joint_q,initial_q); wp.copy(state[0].joint_qd,initial_u)
            # Maximal-coordinate FK must start from the complete tree pose,
            # including passive joints, rather than the previous episode.
            newton.eval_fk(model,initial_q,initial_u,state[0])
            solver._kamino.convert_body_origin_to_com(
                body_com=solver._model_kamino.bodies.i_r_com_i,
                body_q=state[0].body_q,body_q_com=state[0].body_q)
            ks=solver._kamino.StateKamino.from_newton(solver._model_kamino.size,model,state[0],
                initialize_state_prev=True)
            solver._solver_kamino.reset(state_out=ks,bodies_q=ks.q_i,bodies_u=zero_body_u)
            # The maximal-body reset clears generalized coordinates. Restore
            # the matching tree coordinates and multi-turn references before
            # the first controller feedback; bodies alone are insufficient.
            wp.copy(state[0].joint_q,initial_q)
            wp.copy(state[0].joint_qd,initial_u)
            wp.copy(ks.q_j_p,initial_q)
            solver._kamino.convert_body_com_to_origin(
                body_com=solver._model_kamino.bodies.i_r_com_i,
                body_q_com=state[0].body_q,body_q=state[0].body_q)
            # Installed Kamino leaves these two reset calls commented out.
            # This harness resets every world, so clearing all caches is valid.
            for name in ('_ws_limits','_ws_contacts'):
                cache=getattr(solver._solver_kamino,name,None)
                if cache is not None: cache.reset()
            controller.reset()
        graphs=None
        reset()
        if 'body' in data.files:
            from scipy.spatial.transform import Rotation
            native_body=data['body'][:a.num_envs]
            candidate=state[0].body_q.numpy().reshape(a.num_envs,-1,7)[:,[rb_names.index(n) for n in body_names]]
            matrices=Rotation.from_quat(candidate[...,3:].reshape(-1,4)).as_matrix().reshape(a.num_envs,-1,3,3)
            original=matrices@np.asarray([corrections[rb_names.index(n)].T for n in body_names])
            native_matrices=Rotation.from_quat(native_body[...,3:].reshape(-1,4)).as_matrix().reshape(a.num_envs,-1,3,3)
            joint_actual=state[0].joint_q.numpy().reshape(a.num_envs,-1)[:,q_ids]
            joint_error=joint_actual-initial_joint
            for j,n in enumerate(names):
                if n not in manifest['spring_joint_names']:
                    joint_error[:,j]=np.arctan2(np.sin(joint_error[:,j]),np.cos(joint_error[:,j]))
            fidelity=dict(body_position_max_error_m=float(np.max(np.abs(candidate[...,:3]-native_body[...,:3]))),
                body_rotation_matrix_max_error=float(np.max(np.abs(original-native_matrices))),
                joint_coordinate_max_error=float(np.max(np.abs(joint_error))))
            result['initial_joint_coordinates']={n:dict(native=float(initial_joint[0,j]),
                kamino=float(joint_actual[0,j]),max_error=float(np.max(np.abs(joint_error[:,j]))))
                for j,n in enumerate(names)}
            result['initial_model_fidelity']=fidelity;save()
            if max(fidelity.values())>2e-5:
                raise ValueError(f'Native body/joint initial conditions do not match: {fidelity}')
        for i in range(4):
            k=i%2; physics(k)
        wp.synchronize()
        if not a.no_graph:
            graphs=[]
            for k in (0,1):
                with wp.ScopedCapture(device=device) as capture: physics(k)
                graphs.append(capture.graph)
        result['physics_cuda_graph']=graphs is not None
        def step(i):
            k=current[0]
            q=wp.to_torch(state[k].joint_q).reshape(a.num_envs,-1)[:,qindex]
            dq=wp.to_torch(state[k].joint_qd).reshape(a.num_envs,-1)[:,uindex]
            effort=controller(q,dq,i,passive_damping=True)
            f=wp.to_torch(control.joint_f).reshape(a.num_envs,-1)
            f.zero_(); f[:,uindex]=effort
            if graphs is None: physics(k)
            else: wp.capture_launch(graphs[k])
            current[0]=1-k
        def observe():
            poses=wp.to_torch(state[current[0]].body_q).reshape(a.num_envs,-1,7)
            wheels=poses[:,wheel_bodies]
            def rotate(local):
                cross=torch.linalg.cross(wheels[...,3:6],local.expand(a.num_envs,-1,-1))
                return local+2*(wheels[...,6:]*cross+torch.linalg.cross(wheels[...,3:6],cross))
            centers=wheels[...,:3]+rotate(wheel_offsets)
            axial_z=rotate(wheel_axes)[...,2].clamp(-1.,1.)
            extent=.06*(1-axial_z.square()).clamp_min(0.).sqrt()+.012500000186264515*axial_z.abs()
            last_clearance[0]=centers[...,2]-extent
            pose=poses[:,ids]
            local=locals_.expand(a.num_envs,-1,-1,-1)
            cross=torch.linalg.cross(pose[...,3:6],local)
            world=pose[...,:3]+local+2*(pose[...,6:]*cross+torch.linalg.cross(pose[...,3:6],cross))
            gap=(world[:,:,0]-world[:,:,1]).norm(dim=-1).amax(-1)
            dq=wp.to_torch(state[current[0]].joint_qd).reshape(a.num_envs,-1)[:,uindex]
            return gap,poses[:,baseindex],dq

    result['initialization_s']=time.perf_counter()-started
    save()
    writer=SummaryWriter(str(out))
    timed=[]; trace={k:[] for k in ('gap_m','base_pose','joint_speed_peak','wheel_clearance_m')}
    with torch.inference_mode():
        for rollout in range(a.warmup_rollouts+a.rollouts):
            torch.cuda.synchronize(); t=time.perf_counter(); reset(); torch.cuda.synchronize()
            reset_s=time.perf_counter()-t
            initial_gap,initial_base,initial_speed=observe()
            if a.backend=='physx' and rollout==0:
                np.savez_compressed(out/'initial.npz',joint=q0.cpu().numpy(),base=base0.cpu().numpy(),
                    body=env.robot.data.body_link_pose_w.torch.cpu().numpy())
            initial=dict(gap_max_m=initial_gap.max().item(),base_first=initial_base[0].cpu().tolist(),
                speed_peak=initial_speed.abs().max().item(),base_target_max_abs_error=
                (initial_base-base0).abs().max().item())
            if initial['base_target_max_abs_error']>2e-6 or initial['gap_max_m']>.003:
                result['failed_initial_state']=initial
                result['expected_initial_base']=base0[0].cpu().tolist()
                raise RuntimeError('Reset failed to reproduce the shared root pose or closed chain')
            gaps=[]; bases=[]; speeds=[]; clearances=[]
            counts=[]; newton_counts=[]
            torch.cuda.synchronize(); t=time.perf_counter()
            for i in range(a.steps):
                step(i)
                gap,base,speed=observe()
                gaps.append(gap.clone()); bases.append(base.clone()); speeds.append(speed.abs().amax(-1))
                clearances.append(last_clearance[0].clone())
                if a.backend=='kamino':
                    counts.append(wp.to_torch(solver._contacts_kamino._data.world_active_contacts).clone())
                    newton_counts.append(wp.to_torch(contacts.rigid_contact_count).clone())
            torch.cuda.synchronize(); wall=time.perf_counter()-t
            gap=torch.stack(gaps); base=torch.stack(bases); speed=torch.stack(speeds)
            clearance=torch.stack(clearances)
            if not bool(torch.isfinite(gap).all() and torch.isfinite(base).all() and torch.isfinite(speed).all()):
                raise RuntimeError('Nonfinite native dynamics')
            row=dict(rollout=rollout,warmup=rollout<a.warmup_rollouts,wall_s=wall,reset_s=reset_s,initial=initial,
                gap_max_m=gap.max().item(),gap_rms_m=gap.square().mean().sqrt().item(),
                over3mm_env_physics=int((gap>.003).sum()),
                envs_over3mm=int((gap>.003).any(0).sum()),base_min_z_m=base[...,2].min().item(),
                base_final_mean_z_m=base[-1,:,2].mean().item(),joint_speed_peak=speed.max().item())
            row['wheel_penetration_peak_m']=(-clearance.min()).clamp_min(0.).item()
            if counts:
                contact_counts=torch.stack(counts)
                row['contacts_per_world_peak']=int(contact_counts.max())
                row['contact_env_physics']=int((contact_counts>0).sum())
                row['newton_contacts_global_peak']=int(torch.stack(newton_counts).max())
                if row['contacts_per_world_peak']>=a.contacts_per_world:
                    raise RuntimeError('Contact capacity exhausted; this timing is invalid')
                if row['newton_contacts_global_peak']>=a.num_envs*a.newton_contacts_per_world:
                    raise RuntimeError('Newton contact capacity exhausted; this timing is invalid')
            for tag in ('wall_s','reset_s','gap_max_m','gap_rms_m','envs_over3mm','base_final_mean_z_m','joint_speed_peak'):
                writer.add_scalar('ClosedChain/'+tag,row[tag],rollout)
            writer.flush()
            result.setdefault('rollouts',[]).append(row); save()
            print(json.dumps(row),flush=True)
            if rollout>=a.warmup_rollouts:
                timed.append(row)
                trace['gap_m'].append(gap.cpu().numpy())
                trace['base_pose'].append(base.cpu().numpy())
                trace['joint_speed_peak'].append(speed.cpu().numpy())
                trace['wheel_clearance_m'].append(clearance.cpu().numpy())
        result['timing']=dict(median_rollout_s=float(np.median([r['wall_s'] for r in timed])),
            median_reset_s=float(np.median([r['reset_s'] for r in timed])),
            rollout_physics_calls=a.steps,per_env_simulated_s=a.steps*dt)
        result['effects']=dict(gap_max_m=max(r['gap_max_m'] for r in timed),
            over3mm_env_physics=sum(r['over3mm_env_physics'] for r in timed),
            evaluated_env_physics=a.num_envs*a.steps*a.rollouts,
            reset_on_guard=False,guard_semantics='Count would-terminate samples; keep fixed duration for matched mechanics probe')
    np.savez_compressed(out/'trace.npz',**{k:np.asarray(v) for k,v in trace.items()})
    writer.close(); result['status']='complete'; save()
    print('CLOSED_CHAIN_BENCHMARK_COMPLETE',json.dumps(result['timing']),json.dumps(result['effects']),flush=True)
except BaseException as e:
    result.update(status='failed',error=repr(e),traceback=traceback.format_exc());save()
    print(result['traceback'],flush=True)
    raise
finally:
    if env is not None: env.close()
    if launcher is not None: launcher.app.close()
