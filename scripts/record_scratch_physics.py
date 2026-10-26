"""Read actual PhysX scene/collider settings without changing the simulation."""
import hashlib
from pathlib import Path


def record(env):
    stage = env.sim.stage
    scene = stage.GetPrimAtPath(env.sim.cfg.physics_prim_path)

    def attributes(prim, names):
        result = {}
        for name in names:
            if not prim.IsValid():
                result[name] = None
                continue
            attr = prim.GetAttribute(name)
            value = attr.Get() if attr.IsValid() else None
            if value is not None and not isinstance(value, (bool, int, float, str)):
                value = str(value)
            result[name] = value
        return result

    flags = attributes(scene, (
        "physxScene:solverType", "physxScene:enableExternalForcesEveryIteration",
        "physxScene:enableStabilization", "physxScene:enableCCD",
        "physxScene:minPositionIterationCount", "physxScene:maxPositionIterationCount",
        "physxScene:minVelocityIterationCount", "physxScene:maxVelocityIterationCount"))
    articulation = stage.GetPrimAtPath("/World/envs/env_0/Robot/base_link")
    iterations = attributes(articulation, (
        "physxArticulation:solverPositionIterationCount", "physxArticulation:solverVelocityIterationCount"))
    wheels = {}
    for body in ("L_link3", "R_link3"):
        path = f"/World/envs/env_0/Robot/{body}/Collision_000"
        prim = stage.GetPrimAtPath(path)
        wheels[body] = dict(path=path, valid=prim.IsValid(), type=str(prim.GetTypeName()) if prim.IsValid() else None,
            attributes=attributes(prim, ("radius", "height", "axis", "physxConvexGeometry:margin")))
    cylinder = dict(value=None, status="unavailable")
    try:
        from carb.settings import get_settings
        from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
        value = get_settings().get(SETTING_COLLISION_APPROXIMATE_CYLINDERS)
        cylinder = dict(setting=SETTING_COLLISION_APPROXIMATE_CYLINDERS, value=value,
            status="observed" if isinstance(value, bool) else "not_boolean")
    except (ImportError, AttributeError, RuntimeError) as error:
        cylinder["reason"] = str(error)
    return dict(schema="scratch-native-physics-settings-v1", physical_dt_s=env.dt,
        policy_dt_s=env.policy_dt, scene_flags=flags, articulation_iterations=iterations,
        cylinder_convex_approximation=cylinder, wheel_colliders=wheels,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Read-only runtime configuration; no contact, geometry, solver, action or acceptance changes")
