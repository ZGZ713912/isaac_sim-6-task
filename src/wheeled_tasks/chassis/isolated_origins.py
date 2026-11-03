"""Bounded getup placement with complete native collision isolation.

The separate 96 m floors retain their geometry and per-environment material.
Isaac Lab's native inverted collision groups isolate the complete Robot and
Terrain hierarchy for each environment before PhysX creates its scene.
"""
import math

OVERLAP = "isolated_overlap_v1"
GRID = "isolated_grid4m_v1"


def layout_origins(count, mode=OVERLAP):
    """Keep old contracts at zero; spread new batches without large coordinates.

    Collision filtering is still required: the unchanged 96 m floors overlap.
    At 512 environments the furthest origin is 44 m from zero on each axis.
    This bounds float32 coordinates; it does not assert identical trajectories.
    """
    if type(count) is not int or not 1 <= count <= 512:
        raise ValueError("Isolated native getup batches are bounded to 1..512 environments")
    if mode == OVERLAP:
        return [(0., 0., 0.)] * count
    if mode != GRID:
        raise ValueError("Unknown native environment layout")
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    return [((i % columns - (columns - 1) / 2) * 4.,
             (i // columns - (rows - 1) / 2) * 4., 0.) for i in range(count)]


def validate_isolated_origins(config, count):
    enabled = config.get("full_rl_isolated_origins", False)
    if enabled and (not config.get("full_rl_square_ground") or not config.get("flat_triangle_mesh")):
        raise ValueError("Isolated getup origins require the explicit square triangle-mesh task")
    if enabled:
        layout_origins(count, config.get("native_environment_layout", OVERLAP))
    elif "native_environment_layout" in config:
        raise ValueError("Native environment layout requires complete collision isolation")
    return enabled


def isolate_native_worlds(stage, physics_scene_path, count, *, mode=OVERLAP, shared_ground=None):
    origins = layout_origins(count, mode)
    from isaaclab.cloner import filter_collisions
    from pxr import UsdPhysics

    root = "/World/getup_collision_groups"
    if stage.GetPrimAtPath(root).IsValid():
        raise RuntimeError("Getup collision groups must be created exactly once")
    env_paths = [f"/World/envs/env_{i}" for i in range(count)]
    global_paths = []
    if shared_ground is not None:
        global_paths = [str(p.GetPath()) for p in stage.Traverse()
                        if str(p.GetPath()).startswith(shared_ground)
                        and p.HasAPI(UsdPhysics.CollisionAPI)]
    filter_collisions(stage, physics_scene_path, root, env_paths, global_paths=global_paths)
    if stage.GetPrimAtPath(physics_scene_path).GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True:
        raise RuntimeError("Native inverted collision filtering was not authored")

    colliders = [[] for _ in range(count)]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if shared_ground is not None and path.startswith(shared_ground):
            continue
        if not path.startswith("/World/envs/env_"):
            raise RuntimeError("Isolated getup world contains an ungrouped collider: " + path)
        index = int(path.split("/env_", 1)[1].split("/", 1)[0])
        colliders[index].append(prim.GetPath())
    for i, env_path in enumerate(env_paths):
        group_path = f"{root}/group{i}"
        group = UsdPhysics.CollisionGroup.Get(stage, group_path)
        collection = group.GetCollidersCollectionAPI()
        includes = [str(path) for path in collection.GetIncludesRel().GetTargets()]
        allowed = [env_path] + (global_paths if shared_ground is not None else [])
        filtered = [str(path) for path in group.GetFilteredGroupsRel().GetTargets()]
        if (group_path not in filtered
                or any(path.startswith(root + "/group") and path != group_path for path in filtered)
                or env_path not in includes
                or (shared_ground is None and any(path not in allowed for path in includes))):
            raise RuntimeError("Getup collision groups must allow only their own environment")
        query = collection.ComputeMembershipQuery()
        if ((shared_ground is None and not any(str(path).startswith(env_path + "/Terrain/") for path in colliders[i]))
                or not any(str(path).startswith(env_path + "/Robot/") for path in colliders[i])
                or not all(query.IsPathIncluded(path) for path in colliders[i])):
            raise RuntimeError("Native getup collision group does not contain its complete robot and floor")
    return dict(mode=mode, group_count=count, origins_m=origins[0] if mode == OVERLAP else origins,
                grid_spacing_m=4. if mode == GRID else 0.,
                max_abs_origin_xy_m=max(abs(v) for origin in origins for v in origin[:2]),
                max_abs_support_coordinate_m=48. + max(abs(v) for origin in origins for v in origin[:2]),
                inverted_filter=True, global_collision_paths=[],
                complete_robot_and_terrain_membership=True,
                collision_shapes_per_environment=[len(paths) for paths in colliders])
