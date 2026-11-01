"""Experimental standalone Kamino model adapter; never selected by training.

The installed Kamino converter changes child frames sequentially, including
loop joints. Canonicalize tree frames before conversion and give ball joints
equal orientation frames (their rotations are unconstrained). Preserve every
body mass, COM, inertia, shape and joint anchor under these frame changes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _matrix(q):
    return Rotation.from_quat(np.asarray(q, dtype=float)).as_matrix()


def _transform(value):
    v = np.asarray(value, dtype=float)
    out = np.eye(4)
    out[:3, :3] = _matrix(v[3:])
    out[:3, 3] = v[:3]
    return out


def _warp_transform(matrix):
    import warp as wp
    return wp.transform(wp.vec3(*matrix[:3, 3]),
                        wp.quat(*Rotation.from_matrix(matrix[:3, :3]).as_quat()))


def canonicalize_frames(builder):
    """Return old-to-new local rotations plus numerical preservation audit."""
    root_rotations = {}
    corrections = [None] * builder.body_count
    for j, typ in enumerate(builder.joint_type):
        if int(typ) == 2:  # ball loops: do not determine tree frames
            continue
        parent, child = builder.joint_parent[j], builder.joint_child[j]
        if parent < 0:
            root_rotations[child] = np.eye(3)
        else:
            if parent not in root_rotations or child in root_rotations:
                raise ValueError('Expected one parent-before-child articulation tree')
            xp, xc = _transform(builder.joint_X_p[j]), _transform(builder.joint_X_c[j])
            root_rotations[child] = root_rotations[parent] @ xp[:3,:3] @ xc[:3,:3].T
        corrections[child] = root_rotations[child].T
    if any(c is None for c in corrections):
        raise ValueError('Every rigid body must have a tree frame')
    transforms = []
    audit = dict(max_world_com_error_m=0., max_world_inertia_error=0.,
                 max_world_shape_transform_error=0., max_world_anchor_error_m=0.,
                 max_scalar_joint_frame_rotation_error=0.)
    for b, c in enumerate(corrections):
        t = np.eye(4); t[:3,:3] = c
        transforms.append(t)
        old = _transform(builder.body_q[b])
        com = np.asarray(builder.body_com[b],dtype=float)
        inertia = np.asarray(builder.body_inertia[b],dtype=float).reshape(3,3)
        new = old @ t
        new_com = c.T @ com
        new_inertia = c.T @ inertia @ c
        audit['max_world_com_error_m'] = max(audit['max_world_com_error_m'],
            float(np.max(np.abs((old[:3,:3]@com+old[:3,3])-(new[:3,:3]@new_com+new[:3,3])))))
        audit['max_world_inertia_error'] = max(audit['max_world_inertia_error'],
            float(np.max(np.abs(old[:3,:3]@inertia@old[:3,:3].T-new[:3,:3]@new_inertia@new[:3,:3].T))))
        builder.body_q[b] = _warp_transform(new)
        import warp as wp
        builder.body_com[b] = wp.vec3(*new_com)
        builder.body_inertia[b] = wp.mat33(*new_inertia.flatten())
    for s, body in enumerate(builder.shape_body):
        if body < 0:
            continue
        old = _transform(builder.shape_transform[s])
        new = np.linalg.inv(transforms[body]) @ old
        audit['max_world_shape_transform_error'] = max(audit['max_world_shape_transform_error'],
            float(np.max(np.abs(transforms[body] @ new-old))))
        builder.shape_transform[s] = _warp_transform(new)
    for j, typ in enumerate(builder.joint_type):
        for field, body in (('joint_X_p',builder.joint_parent[j]),
                            ('joint_X_c',builder.joint_child[j])):
            old = _transform(getattr(builder,field)[j])
            t = np.eye(4) if body < 0 else transforms[body]
            new = np.linalg.inv(t) @ old
            audit['max_world_anchor_error_m'] = max(audit['max_world_anchor_error_m'],
                float(np.max(np.abs((t@new)[:3,3]-old[:3,3]))))
            if int(typ) == 2:
                new[:3,:3] = np.eye(3)  # spherical joint has no angular restriction
            getattr(builder,field)[j] = _warp_transform(new)
        if int(typ) in (0,1):
            rp = _transform(builder.joint_X_p[j])[:3,:3]
            rc = _transform(builder.joint_X_c[j])[:3,:3]
            audit['max_scalar_joint_frame_rotation_error'] = max(
                audit['max_scalar_joint_frame_rotation_error'],float(np.max(np.abs(rp-rc))))
    if max(audit.values()) > 2e-6:
        raise ValueError(f'Frame canonicalization did not preserve the model: {audit}')
    return np.asarray(corrections), audit


def import_robot(directory, *, canonical_frames=True):
    import json
    from pxr import Usd, UsdGeom
    import newton
    stage = Usd.Stage.Open(str(directory/'robot.usda'))
    # Layer-wrapper USD files do not inherit stage upAxis metadata from their
    # sublayers. Author only this in-memory session; the declared asset is Z-up.
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.add_usd(stage,floating=True,enable_self_collisions=False,
        collapse_fixed_joints=False,joint_ordering='dfs',load_visual_shapes=False,
        force_position_velocity_actuation=True)
    manifest = json.loads((directory/'manifest.json').read_text())
    names = [n.rsplit('/',1)[-1] for n in builder.joint_label]
    bodies = [n.rsplit('/',1)[-1] for n in builder.body_label]
    tree = [names[i] for i,t in enumerate(builder.joint_type) if int(t) in (0,1)]
    loops = sum(int(t)==2 for t in builder.joint_type)
    if set(bodies)!=set(manifest['rigid_body_names']) or set(tree)!=set(manifest['tree_joint_names']) or loops!=6:
        raise ValueError('Native import lost bodies, tree joints or closed-chain joints')
    corrections,audit = (canonicalize_frames(builder) if canonical_frames else
        (np.tile(np.eye(3),(builder.body_count,1,1)),dict(canonicalization=False)))
    return builder,corrections,audit
