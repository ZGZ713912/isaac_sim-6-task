"""Keep independent getup worlds near zero without cross-environment contacts.

The separate 96 m floors retain their geometry and per-environment material.
Isaac Lab's native inverted collision groups isolate the complete Robot and
Terrain hierarchy for each environment before PhysX creates its scene.
"""


def validate_isolated_origins(config, count):
    enabled = config.get("full_rl_isolated_origins", False)
    if enabled and (not config.get("full_rl_square_ground") or not config.get("flat_triangle_mesh")):
        raise ValueError("Isolated getup origins require the explicit square triangle-mesh task")
    if enabled and not 1 <= count <= 512:
        raise ValueError("Isolated native getup batches are bounded to 1..512 environments")
    return enabled


def isolate_native_worlds(stage, physics_scene_path, count):
    from isaaclab.cloner import filter_collisions
    from pxr import UsdPhysics

    root = "/World/getup_collision_groups"
    if stage.GetPrimAtPath(root).IsValid():
        raise RuntimeError("Getup collision groups must be created exactly once")
    env_paths = [f"/World/envs/env_{i}" for i in range(count)]
    filter_collisions(stage, physics_scene_path, root, env_paths, global_paths=[])
    if stage.GetPrimAtPath(physics_scene_path).GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True:
        raise RuntimeError("Native inverted collision filtering was not authored")

    colliders = [[] for _ in range(count)]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if not path.startswith("/World/envs/env_"):
            raise RuntimeError("Isolated getup world contains an ungrouped collider: " + path)
        index = int(path.split("/env_", 1)[1].split("/", 1)[0])
        colliders[index].append(prim.GetPath())
    for i, env_path in enumerate(env_paths):
        group_path = f"{root}/group{i}"
        group = UsdPhysics.CollisionGroup.Get(stage, group_path)
        collection = group.GetCollidersCollectionAPI()
        if ([str(path) for path in group.GetFilteredGroupsRel().GetTargets()] != [group_path]
                or [str(path) for path in collection.GetIncludesRel().GetTargets()] != [env_path]):
            raise RuntimeError("Getup collision groups must allow only their own environment")
        query = collection.ComputeMembershipQuery()
        if (not any(str(path).startswith(env_path + "/Terrain/") for path in colliders[i])
                or not any(str(path).startswith(env_path + "/Robot/") for path in colliders[i])
                or not all(query.IsPathIncluded(path) for path in colliders[i])):
            raise RuntimeError("Native getup collision group does not contain its complete robot and floor")
    return dict(mode="isolated_overlap_v1", group_count=count, origins_m=[0., 0., 0.],
                inverted_filter=True, global_collision_paths=[],
                complete_robot_and_terrain_membership=True,
                collision_shapes_per_environment=[len(paths) for paths in colliders])
