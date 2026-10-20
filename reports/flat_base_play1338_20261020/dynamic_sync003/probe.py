"""Bounded native diagnosis: physics link poses versus USD before/after publishing."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                  OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")

import play_v5_grounded

OriginalView = play_v5_grounded.PolicyView


class ProbeView(OriginalView):
    def __init__(self, env, args):
        super().__init__(env, args)
        import numpy as np
        import omni.physx
        import torch
        from pxr import Usd, UsdGeom

        topology = {}
        for name in env.robot.body_names:
            visual = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/" + name + "/Visual")
            topology[name] = dict(valid=bool(visual), meshes=0, reset_xform_paths=[])
            if visual:
                for prim in Usd.PrimRange(visual):
                    if prim.IsA(UsdGeom.Mesh):
                        topology[name]["meshes"] += 1
                    if prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).GetResetXformStack():
                        topology[name]["reset_xform_paths"].append(str(prim.GetPath()))
        (OUT / "visual_topology.json").write_text(json.dumps(topology, indent=2) + "\n")
        self.rows = []
        self.probe_time = 0.
        self.capture_pending = None
        self.captured = set()
        original_render = env.sim.render
        interface = omni.physx.get_physx_interface()
        (OUT / "native_api.txt").write_text(str(interface.update_transformations.__doc__))

        def compare():
            values = []
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            state = env.robot.data.body_link_pose_w.torch[0].cpu().numpy().copy()
            for name, pose in zip(env.robot.body_names, state):
                prim = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/" + name)
                matrix = cache.GetLocalToWorldTransform(prim)
                position = np.asarray(matrix.ExtractTranslation())
                q = matrix.ExtractRotationQuat()
                orientation = np.asarray([*q.GetImaginary(), q.GetReal()])
                error = 2 * np.arccos(np.clip(abs(np.dot(pose[3:], orientation)) /
                    max(np.linalg.norm(pose[3:]) * np.linalg.norm(orientation), 1e-12), 0., 1.))
                values.append(dict(name=name, position_error_m=float(np.linalg.norm(position - pose[:3])),
                    angle_error_deg=float(np.degrees(error)), physics_pose=pose.tolist(),
                    usd_pose=[*position.tolist(), *orientation.tolist()]))
            return values

        def render(*args, **kwargs):
            before = compare()
            q = env.robot.data.joint_pos.torch.clone()
            v = env.robot.data.joint_vel.torch.clone()
            pose = env.robot.data.body_link_pose_w.torch.clone()
            fixed = False  # Observe the production fix without an extra publisher.
            if fixed:
                interface.update_transformations(False, True)
            after = compare()
            unchanged = (torch.equal(q, env.robot.data.joint_pos.torch)
                and torch.equal(v, env.robot.data.joint_vel.torch)
                and torch.equal(pose, env.robot.data.body_link_pose_w.torch))
            self.rows.append(dict(time_s=self.probe_time, explicit_publish=fixed,
                physics_unchanged=unchanged, before=before, after=after))
            (OUT / "transforms.json").write_text(json.dumps(self.rows, indent=2) + "\n")
            result = original_render(*args, **kwargs)
            key = "before" if .3 <= self.probe_time < .6 else (
                "forward" if self.probe_time >= 1.1 and self.probe_time < 1.3 else (
                "height_yaw" if self.probe_time >= 2.5 else None))
            if key and key not in self.captured:
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                capture_viewport_to_file(get_active_viewport(), str(OUT / (key + ".png")))
                self.captured.add(key)
            return result

        env.sim.render = render

    def command(self, seconds, dt):
        self.probe_time = seconds
        return ([0., 0., .305] if seconds < .6 else
                [.5, 0., .305] if seconds < 1.5 else
                [0., .5, .35] if seconds < 3. else [0., 0., .305])


play_v5_grounded.PolicyView = ProbeView
import play_flat_base
sys.argv = [str(ROOT / "scripts/play_flat_base.py"), "--output", str(OUT / "gui"), "--gui-seconds", "40"]
raise SystemExit(play_flat_base.main())
