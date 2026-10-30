"""Measure sliding deceleration after the same native material update as reset."""
from pathlib import Path
import json
import sys

import numpy as np

from isaaclab.app import AppLauncher


def main():
    launcher = AppLauncher(dict(headless=True, device="cpu", enable_cameras=False))
    try:
        import torch
        import warp as wp
        from pxr import UsdGeom, UsdPhysics
        import isaaclab.sim as su
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab_physx.physics import PhysxCfg

        dt = .001
        sim = su.SimulationContext(su.SimulationCfg(device="cpu", dt=dt, physics=PhysxCfg(solver_type=1)))
        plane = UsdGeom.Plane.Define(sim.stage, "/World/Plane")
        plane.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
        ground = su.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0., friction_combine_mode="multiply")
        ground.func("/World/GroundMaterial", ground)
        su.bind_physics_material("/World/Plane", "/World/GroundMaterial")
        objects = []
        for i in range(3):
            material = su.RigidBodyMaterialCfg(static_friction=.3, dynamic_friction=.3, restitution=0., friction_combine_mode="multiply")
            obj = RigidObject(RigidObjectCfg(prim_path=f"/World/Slider{i}",
                spawn=su.CuboidCfg(size=(.1,.1,.1), rigid_props=su.RigidBodyPropertiesCfg(),
                    mass_props=su.MassPropertiesCfg(mass=1.),
                    collision_props=su.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.), physics_material=material),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,i*.3,.055))))
            objects.append(obj)
        sim.reset()
        for _ in range(200):
            sim.step(render=False)
            for obj in objects:
                obj.update(dt)
        rows = []
        for obj, mu in zip(objects, [0.,.2,.4]):
            material = wp.to_torch(obj.root_view.get_material_properties()).clone().cpu()
            material[...,0:2] = mu
            material[...,2] = 0.
            obj.root_view.set_material_properties(wp.from_torch(material, dtype=wp.float32),
                wp.from_torch(torch.tensor([0], dtype=torch.int32), dtype=wp.int32))
            np.testing.assert_allclose(wp.to_torch(obj.root_view.get_material_properties()).numpy(),material.numpy(),atol=1e-7)
            velocity = torch.zeros((1,6),dtype=torch.float32)
            velocity[0,0] = 1.
            obj.write_root_com_velocity_to_sim_index(root_velocity=velocity)
            obj.write_data_to_sim()
            rows.append(dict(requested_effective_friction=mu,initial_velocity_m_s=1.))
        for _ in range(100):
            sim.step(render=False)
            for obj in objects:
                obj.update(dt)
        for obj, row in zip(objects, rows):
            final = float(obj.data.root_link_lin_vel_w.torch[0,0])
            measured = (1.-final)/(.1*9.81)
            row.update(final_velocity_m_s=final,measured_effective_friction=measured)
        passed = all(abs(r["measured_effective_friction"]-r["requested_effective_friction"])<.015 for r in rows)
        receipt = dict(schema="native-contact-sliding-probe/v1",ground_coefficient=1.,combine_mode="multiply",
            physics_dt_s=dt,measurement_ticks=100,material_updates="same PhysX view API as episode randomization",
            sliders=rows,passed=passed,scope="contact combination check; no controller or robot behavior qualification")
        output = Path(sys.argv[1])
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(receipt,indent=2)+"\n")
        if not passed:
            raise RuntimeError("Measured sliding friction differs from effective pair coefficient")
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
