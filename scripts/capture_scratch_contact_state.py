"""Read native contact buffers and full joint state at each existing 1 ms boundary.

This is a diagnostic observer. It never advances physics, queries observations,
writes state, or changes the policy, contact force, reward or acceptance.
"""
from pathlib import Path


def select_wheel_contact_pairs(counts, starts, rows):
    """Convert SDK UInt32 buffers before advanced indexing on CUDA."""
    if counts.shape != starts.shape or counts.ndim != 2 or rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("Unexpected sensor/filter contact pair layout")
    return counts.long()[rows], starts.long()[rows]


def reduce_contact_points(force, normal, separation, counts, starts, *, maximum=64):
    """Reduce the installed tensor API's packed sensor/filter contact buffers.

    Counts and starts have shape [environment, wheel, filter]. Unused buffer
    entries are never counted. Overflow is returned explicitly, never accepted
    as a zero-force contact. The real API returns forces first, unlike the
    unrelated mock interface's tuple order.
    """
    import torch

    if (counts.shape != starts.shape or counts.ndim != 3 or counts.shape[1] != 2
            or force.ndim != 2 or force.shape[1] != 1
            or normal.shape != (len(force), 3) or separation.shape != force.shape
            or not len(force) or maximum <= 0):
        raise ValueError("Unexpected packed native contact buffers")
    counts, starts = counts.long(), starts.long()
    offsets = torch.arange(maximum, device=counts.device)
    present = offsets < counts[..., None]
    indices = starts[..., None] + offsets
    invalid = present & ((indices < 0) | (indices >= len(force)))
    overflow = invalid.any((-1, -2)) | (counts > maximum).any(-1) | (counts < 0).any(-1)
    safe = indices.clamp(0, len(force)-1)
    valid = present & ~invalid
    # Native buffers only initialize the counted records. A masked-out NaN
    # must not contaminate the sum through NaN*0.
    vectors = torch.where(valid[..., None], force[safe] * normal[safe], 0.)
    distance = separation[safe, 0]
    return dict(point_count=counts.sum(-1), force=vectors.sum((-2, -3)),
        minimum_separation=torch.where(valid, distance, torch.inf).amin((-1, -2)),
        overflow=overflow)


def scene_properties(env):
    result = {}
    names = ("physxRigidBody:sleepThreshold", "physxRigidBody:stabilizationThreshold",
             "physxArticulation:sleepThreshold", "physxArticulation:stabilizationThreshold",
             "physxCollision:contactOffset", "physxCollision:restOffset")
    for body in ("base_link", "L_link3", "R_link3"):
        for suffix in ("", "/Collision_000"):
            path = f"/World/envs/env_0/Robot/{body}{suffix}"
            prim = env.sim.stage.GetPrimAtPath(path)
            if prim.IsValid():
                result[path] = {name: prim.GetAttribute(name).Get()
                                for name in names if prim.GetAttribute(name).IsValid()}
    return result


def contact_view_bindings(env):
    """Verify the native sensor ordering before assigning any packed records."""
    bindings, all_lanes = [], []
    for ids, view in env.contact_views:
        lanes = ids.cpu().tolist()
        expected = [env.env_paths[lane] + '/Robot/' + body
                    for lane in lanes for body in env.robot.body_names]
        if view.sensor_count != len(expected) or list(view.sensor_paths) != expected:
            raise ValueError('Native contact sensor paths or order differ')
        actual_filters = set()

        def collect(value):
            if isinstance(value, str):
                actual_filters.add(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            else:
                raise ValueError('Unknown native contact filter path format')

        collect(view.filter_paths)
        expected_filters = {env.env_paths[lane] + f'/Terrain/surface_{index}/geometry/mesh'
                            for lane in lanes for index in range(len(env.surfaces[lane]))}
        if (actual_filters != expected_filters
                or any(view.filter_count != len(env.surfaces[lane]) for lane in lanes)):
            raise ValueError('Native contact filter membership differs')
        all_lanes.extend(lanes)
        bindings.append(dict(lanes=lanes, sensor_count=view.sensor_count, filter_count=view.filter_count,
            max_contact_data_count=view.max_contact_data_count, exact_sensor_order=True, exact_filter_membership=True))
    if sorted(all_lanes) != list(range(env.num_envs)):
        raise ValueError('Native contact views omit or repeat an environment')
    return bindings


def native_joint_properties(env):
    """Observe both friction APIs; a zero new model does not verify the old one."""
    import torch
    import warp as wp

    result = {}
    for name, method in (
            ('legacy_friction_coefficient', 'get_dof_friction_coefficients'),
            ('friction_efforts_static_dynamic_viscous', 'get_dof_friction_properties'),
            ('drive_damping', 'get_dof_dampings'), ('armature', 'get_dof_armatures')):
        try:
            value = wp.to_torch(getattr(env.robot.root_view, method)()).detach().cpu()
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite native parameter')
            result[name] = dict(status='observed', values=value[:21].tolist())
        except Exception as error:
            # Missing optional property evidence cannot be presented as zero.
            result[name] = dict(status='unavailable', reason=str(error))
    return result


def evaluate_with_contact_state(env, actor, output, writer, completed, *, evaluate, digest):
    import numpy as np
    import torch
    import warp as wp

    count = 21
    if env.num_envs < count or env.dt != .001 or env.decimation != 5:
        raise ValueError("Need the original21 cases and five existing1ms boundaries")
    native_gap, native_step = env.closure_gap, env.step
    traces = {name: [] for name in (
        "joint_position", "joint_velocity", "root_link_pose_local", "root_com_velocity_world",
        "wheel_link_pose_local", "wheel_net_force", "wheel_point_force",
        "wheel_contact_point_count", "wheel_minimum_contact_separation", "contact_buffer_overflow",
        "filtered_wheel_force", "closure_gap", "valid")}
    inside, valid = False, None
    properties = scene_properties(env)
    bindings = contact_view_bindings(env)
    joint_properties = native_joint_properties(env)

    def gap():
        value = native_gap()
        if not inside:
            return value
        net = torch.zeros_like(env.contact_force)
        point = torch.zeros(env.num_envs, 2, 3, device=env.device)
        number = torch.zeros(env.num_envs, 2, dtype=torch.long, device=env.device)
        distance = torch.full((env.num_envs, 2), torch.inf, device=env.device)
        overflow = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        for ids, view in env.contact_views:
            net[ids] = wp.to_torch(view.get_net_contact_forces(dt=env.dt)).reshape(len(ids), env.body_count, 3)
            force, _, normals, separation, counts, starts = view.get_contact_data(dt=env.dt)
            counts, starts = wp.to_torch(counts), wp.to_torch(starts)
            rows = (torch.arange(len(ids), device=env.device)[:, None]*env.body_count
                    + torch.as_tensor(env.wheel_ids, device=env.device)[None])
            wheel_counts, wheel_starts = select_wheel_contact_pairs(counts, starts, rows)
            reduced = reduce_contact_points(wp.to_torch(force), wp.to_torch(normals),
                wp.to_torch(separation), wheel_counts, wheel_starts)
            point[ids], number[ids] = reduced["force"], reduced["point_count"]
            distance[ids], overflow[ids] = reduced["minimum_separation"], reduced["overflow"]
        pose = env.robot.data.root_link_pose_w.torch[:count].clone()
        pose[:, :3] -= env.origins[:count]
        wheels = env.robot.data.body_link_pose_w.torch[:count, env.wheel_ids].clone()
        wheels[:, :, :3] -= env.origins[:count, None]
        sample = dict(joint_position=env.robot.data.joint_pos.torch[:count],
            joint_velocity=env.robot.data.joint_vel.torch[:count],
            root_link_pose_local=pose, root_com_velocity_world=env.robot.data.root_com_vel_w.torch[:count],
            wheel_link_pose_local=wheels, wheel_net_force=net[:count, env.wheel_ids],
            wheel_point_force=point[:count], wheel_contact_point_count=number[:count],
            wheel_minimum_contact_separation=distance[:count], contact_buffer_overflow=overflow[:count],
            filtered_wheel_force=env.contact_force[:count, env.wheel_ids], closure_gap=value[:count], valid=valid)
        for name, item in sample.items():
            traces[name].append(item.detach().clone())
        return value

    def step(actions):
        nonlocal inside, valid
        if inside:
            raise RuntimeError("Recursive native contact observer")
        valid = ~env.finished[:count].clone()
        before = len(traces["valid"])
        inside = True
        try:
            result = native_step(actions)
        finally:
            inside = False
        if len(traces["valid"]) - before != 5:
            raise RuntimeError("Contact observer missed a native physical boundary")
        return result

    env.closure_gap, env.step = gap, step
    try:
        result = evaluate(env, actor, output, writer, completed)
    finally:
        env.closure_gap, env.step = native_gap, native_step
    arrays = {name: torch.stack(items).cpu().numpy() for name, items in traces.items()}
    with np.load(Path(output) / "native_trace.npz") as physical:
        fine_force = physical["force"].reshape(-1, count, 2, 3)
        fine_valid = np.repeat(physical["valid"], 5, axis=0)
        if (not np.array_equal(arrays["filtered_wheel_force"], fine_force)
                or not np.array_equal(arrays["valid"], fine_valid)):
            raise RuntimeError("Independent contact observer timing differs from the original trace")
    if arrays["contact_buffer_overflow"].any():
        raise RuntimeError("Packed contact data overflow; cannot interpret missing support")
    for name, values in arrays.items():
        if name == "wheel_minimum_contact_separation":
            counted = arrays["wheel_contact_point_count"] > 0
            if not np.isfinite(values[counted]).all() or not np.isposinf(values[~counted]).all():
                raise RuntimeError("Invalid native contact separation")
        elif not np.isfinite(values).all():
            raise RuntimeError("Nonfinite native contact-state sample: " + name)
    path = Path(output) / "contact_state_trace.npz"
    np.savez_compressed(path, **arrays)
    selected = next(i for i, case in enumerate(result["cases"])
                    if case["height_m"] == .305 and case["pitch_deg"] == 0.)
    active = arrays["valid"][:, selected]
    filtered = np.linalg.norm(arrays["filtered_wheel_force"][active, selected], axis=-1)
    net = np.linalg.norm(arrays["wheel_net_force"][active, selected], axis=-1)
    missing = filtered <= 2.
    writer.add_scalar("ContactState/mid_zero_pitch/missing_wheel_samples", int(missing.sum()), completed)
    writer.add_scalar("ContactState/mid_zero_pitch/missing_filter_but_supported_net_samples",
        int((missing & (net > 2.)).sum()), completed)
    writer.flush()
    result["contact_state_trace"] = dict(path=str(path), sha256=digest(path),
        source_sha256=digest(Path(__file__)), physical_hz=1000, joint_names=list(env.robot.joint_names),
        wheel_names=[env.robot.body_names[i] for i in env.wheel_ids], scene_properties=properties,
        contact_view_bindings=bindings,
        native_joint_properties=joint_properties,
        original_force_and_valid_samples_exact=True, buffer_overflow=False,
        maximum_reduced_points_per_sensor_filter=64,
        no_contact_minimum_separation="positive infinity when no reported contact point exists",
        scope="Read-only same-boundary native joint/root state, net contact force, packed point normals and counts; no physics, state, policy, reward or acceptance change")
    return result
