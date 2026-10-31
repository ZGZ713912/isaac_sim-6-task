#!/usr/bin/env python3
"""Small native positive/negative controls for articulation self-contact."""
import argparse
import json
from pathlib import Path
import sys
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from simulate_traditional_self_righting import digest,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--broadphase',choices=['SAP','MBP'],default='MBP')
    parser.add_argument('--reset-joints',action='store_true')
    parser.add_argument('--forward-after-reset',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'sources').mkdir();shutil.copy2(__file__,args.output/'sources'/Path(__file__).name)
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(dict(headless=True,device='cpu'))
    try:
        import numpy as np
        import torch
        import warp as wp
        from pxr import Gf,UsdGeom,UsdPhysics,PhysxSchema
        import isaaclab.sim as su
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab_physx.physics import PhysxCfg,PhysxManager
        torch.set_num_threads(2)
        sim=su.SimulationContext(su.SimulationCfg(device='cpu',dt=.001,render_interval=1000,
            physics=PhysxCfg(solver_type=1,enable_external_forces_every_iteration=True)))
        for prim in sim.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):PhysxSchema.PhysxSceneAPI.Apply(prim).CreateBroadphaseTypeAttr(args.broadphase)
        cases=[dict(name='enabled',enabled=True),dict(name='disabled',enabled=False),
            dict(name='enabled_pair_filtered',enabled=True,filtered=True),
            dict(name='enabled_after_false',enabled=True,replaced=True),
            dict(name='root_filters_first',enabled=True,root_filters=True),
            dict(name='first_filters_root',enabled=True,root_target=True),
            dict(name='root_collider_filters_first',enabled=True,root_collider_filter=True)]
        for index,case in enumerate(cases):
            prefix=f'/World/envs/env_{index}/Robot'
            UsdGeom.Xform.Define(sim.stage,prefix)
            for name,offset,mass in [('base',(0.,0.,0.),10.),('first',(.3,0.,0.),1.),('second',(.3,.07,0.),1.)]:
                prim=UsdGeom.Xform.Define(sim.stage,prefix+'/'+name)
                prim.AddTranslateOp().Set(Gf.Vec3d(index*2.+offset[0],offset[1],1.))
                body=prim.GetPrim();UsdPhysics.RigidBodyAPI.Apply(body)
                props=PhysxSchema.PhysxRigidBodyAPI.Apply(body);props.CreateDisableGravityAttr(True)
                props.CreateMaxDepenetrationVelocityAttr(1.)
                inertia=UsdPhysics.MassAPI.Apply(body);inertia.CreateMassAttr(mass)
                inertia.CreateDiagonalInertiaAttr(Gf.Vec3f(.01,.01,.01))
                shape=UsdGeom.Cube.Define(sim.stage,prefix+'/'+name+'/collision');shape.CreateSizeAttr(.14)
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                if name!='base':
                    joint=UsdPhysics.RevoluteJoint.Define(sim.stage,prefix+'/'+name+'_joint')
                    joint.CreateBody0Rel().SetTargets([prefix+'/base']);joint.CreateBody1Rel().SetTargets([prefix+'/'+name])
                    joint.CreateAxisAttr('Z');joint.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,0.))
                    joint.CreateLocalPos1Attr(Gf.Vec3f(-offset[0],-offset[1],0.))
                    joint.CreateLocalRot0Attr(Gf.Quatf(1.));joint.CreateLocalRot1Attr(Gf.Quatf(1.))
                if case.get('filtered') and name=='first':
                    UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets([prefix+'/second'])
                if case.get('root_filters') and name=='base':
                    UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets([prefix+'/first'])
                if case.get('root_target') and name=='first':
                    UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets([prefix+'/base'])
                if case.get('root_collider_filter') and name=='base':
                    UsdPhysics.FilteredPairsAPI.Apply(shape.GetPrim()).CreateFilteredPairsRel().SetTargets([prefix+'/first/collision'])
            root=sim.stage.GetPrimAtPath(prefix+'/base')
            UsdPhysics.ArticulationRootAPI.Apply(root)
            api=PhysxSchema.PhysxArticulationAPI.Apply(root)
            if case.get('replaced'):api.CreateEnabledSelfCollisionsAttr(False)
            api.CreateEnabledSelfCollisionsAttr(case['enabled'])
            api.CreateSolverPositionIterationCountAttr(64);api.CreateSolverVelocityIterationCountAttr(32)
        robot=Articulation(ArticulationCfg(prim_path='/World/envs/env_[0-9]+/Robot',spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={'.*':0.},joint_vel={'.*':0.}),
            actuators=dict(passive=IdealPDActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,
                effort_limit=100.,effort_limit_sim=100.))))
        sim.reset();robot.update(0.)
        if args.reset_joints:
            robot.write_joint_position_to_sim_index(position=torch.zeros((len(cases),2)))
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros((len(cases),2)))
            if args.forward_after_reset:sim.forward()
        initial=robot.data.joint_pos.torch.numpy().copy();trace=[]
        for tick in range(100):
            robot.write_data_to_sim();sim.step(render=False);robot.update(.001)
            trace.append(np.c_[robot.data.joint_pos.torch.numpy(),robot.data.joint_vel.torch.numpy()])
        order=[int(p.split('/env_')[1].split('/')[0])for p in robot.root_view.prim_paths]
        records=[]
        values=np.array(trace)
        for i,index in enumerate(order):
            position_change=float(np.max(np.abs(values[:,i,:2]-initial[i])))
            moved=max(position_change,float(np.max(np.abs(initial[i]))))
            expected=cases[index]['enabled'] and not any(cases[index].get(k)for k in ('filtered','root_filters','root_target'))
            records.append(dict(**cases[index],initial_position_rad=initial[i].tolist(),max_position_change_rad=position_change,
                max_speed_rad_s=float(np.max(np.abs(values[:,i,2:]))),expected_motion=expected,
                control_passed=bool(moved>.05 if expected else moved<1e-5)))
        before=robot.data.joint_pos.torch.numpy().copy()
        robot.set_joint_effort_target_index(target=torch.ones((len(cases),2)))
        for tick in range(20):
            robot.write_data_to_sim();sim.step(render=False);robot.update(.001)
        driven_change=np.abs(robot.data.joint_pos.torch.numpy()-before).max(1).tolist()
        np.savez_compressed(args.output/'trace.npz',values=values)
        save(args.output/'receipt.json',dict(status='completed',source_sha256=digest(__file__),records=records,
            all_controls_passed=all(r['control_passed']for r in records),
            broadphase=args.broadphase,external_torque_smoke_position_change_rad=driven_change,
            reset_joints=args.reset_joints,forward_after_reset=args.forward_after_reset,
            trace_sha256=digest(args.output/'trace.npz'),scope='Small self-contact positive/negative controls; no vehicle or trajectory qualification'))
        print(json.dumps(records),flush=True)
    finally:
        launcher.app.close()


if __name__=='__main__':main()
