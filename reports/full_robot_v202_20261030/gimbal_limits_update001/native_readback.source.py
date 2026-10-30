"""Read the movable gimbal limits from native PhysX without running a policy."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202,digest


def main():
    import numpy as np
    from isaaclab.app import AppLauncher
    asset = FullRobotV202()
    launcher = AppLauncher(dict(headless=True,device="cpu",enable_cameras=False))
    try:
        import warp as wp
        import isaaclab.sim as su
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        sim = su.SimulationContext(su.SimulationCfg(device="cpu",dt=.001,gravity=(0.,0.,0.)))
        robot = Articulation(ArticulationCfg(prim_path="/World/Robot",
            spawn=su.UsdFileCfg(usd_path=str(asset.bundle/"robot_gimbal_free.usda"),
                articulation_props=su.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False,solver_position_iteration_count=64,solver_velocity_iteration_count=32)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,2.),joint_pos=asset.manifest["nominal_joint_pos"],joint_vel={".*":0.}),
            actuators=dict(passive=IdealPDActuatorCfg(joint_names_expr=[".*"],stiffness=0.,damping=0.,
                effort_limit=0.,effort_limit_sim=0.,velocity_limit_sim=100.)),soft_joint_pos_limit_factor=1.))
        sim.reset()
        limits = wp.to_torch(robot.root_view.get_dof_limits()).cpu().numpy()[0]
        rows = {n:limits[robot.joint_names.index(n)].tolist() for n in ("M_joint1","M_joint2")}
        assert rows["M_joint1"][0]<-1e20 and rows["M_joint1"][1]>1e20
        np.testing.assert_allclose(rows["M_joint2"],np.deg2rad([-30.,35.]),atol=1e-7,rtol=0.)
        assert len(robot.joint_names)==20 and len(robot.body_names)==21
        report = dict(passed=True,scope="native limit import/readback only; no stop torque or policy test",
            asset_manifest_sha256=digest(asset.bundle/"manifest.json"),
            usd_sha256=digest(asset.bundle/"robot_gimbal_free.usda"),joint_limits_rad=rows,
            joint_names=robot.joint_names,body_names=robot.body_names,new_PPO=0,hardware_motion=False,
            explicit_env_steps=0,script_sha256=digest(__file__))
        Path(__file__).with_name("native_readback.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    finally:
        launcher.app.close()


if __name__=="__main__":
    main()
