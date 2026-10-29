"""Normalize v2.0.1 leg joint axes against the reviewed chassis convention.

Run with the isaacsim6 Python environment. --apply writes the three model files;
the original export is preserved in before/. No simulation or hardware is used.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'model/wheel_leg_v2.0.1/轮腿urdf.SLDASM'
URDF = PACKAGE / 'urdf/轮腿urdf.SLDASM.urdf'
CSV = URDF.with_suffix('.csv')
YAML = PACKAGE / 'config/joint_names_轮腿urdf.SLDASM.yaml'
BASELINE = ROOT / 'model/纯底盘_v7_flat_base107/urdf/robot.urdf'
EXPECTED_SHA = '6e2b437dc2f4f26b78e2a3ca1ac4e35625e4304a7d574ace0049d67997f5a8ff'

# Reuse the existing offline URDF kinematics helper without modifying it.
helper = ROOT / 'reports/full_robot_axis_inertia_20261029/update_axes.py'
spec = importlib.util.spec_from_file_location('reviewed_axis_math', helper)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)
fk, vector, joint_blocks = math.fk, math.vector, math.joint_blocks


def sha(data):
    return hashlib.sha256(data).hexdigest()


def manifest():
    return {str(p.relative_to(PACKAGE)): sha(p.read_bytes())
            for p in sorted(PACKAGE.rglob('*')) if p.is_file()}


def main(apply):
    raw = URDF.read_bytes()
    if sha(raw) != EXPECTED_SHA:
        raise RuntimeError('URDF differs from the inspected v2.0.1 export')
    original_csv, original_yaml = CSV.read_bytes(), YAML.read_bytes()
    original_manifest = manifest()
    baseline_bytes = BASELINE.read_bytes()
    baseline = ET.fromstring(baseline_bytes)
    original = ET.fromstring(raw)
    frames = fk(original)
    bf = fk(baseline)
    left_joint = next(j for j in baseline.findall('joint') if j.get('name') == 'L_joint1')
    align = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    left = align.T @ (bf[left_joint.find('child').get('link')][:3, :3]
                     @ vector(left_joint.find('axis').get('xyz')))
    left /= np.linalg.norm(left)
    assert left[0] > .99
    desired = {'L': left, 'R': -left}
    changes, replacements = [], {}
    old_blocks = joint_blocks(raw.decode('utf-8'))
    for joint in original.findall('joint'):
        name, child = joint.get('name'), joint.find('child').get('link')
        if child.startswith('M_'):
            continue
        assert joint.get('type') == ('fixed' if name == 'RR_joint1' else 'continuous')
        side = child[0]
        old_axis = vector(joint.find('axis').get('xyz'))
        world = frames[child][:3, :3] @ old_axis
        new_axis = frames[child][:3, :3].T @ desired[side]
        new_axis /= np.linalg.norm(new_axis)
        new_axis[np.abs(new_axis) < 5e-15] = 0.
        new_axis /= np.linalg.norm(new_axis)
        block, count = re.subn(r'(<axis\s+xyz=")[^"]*(")',
                              lambda m: m[1] + ' '.join(format(x, '.17g') for x in new_axis) + m[2],
                              old_blocks[name])
        assert count == 1
        if name == 'RR_joint1':
            block, count = re.subn('type="fixed"', 'type="continuous"', block, count=1)
            assert count == 1
        norm = float(np.linalg.norm(world))
        dot = float(world @ desired[side] / norm) if norm > 1e-15 else None
        correction = (float(np.degrees(np.arctan2(np.linalg.norm(np.cross(world, desired[side])),
                                                 abs(world @ desired[side]))))
                      if norm > 1e-15 else None)
        changes.append(dict(name=name, child=child, side=side, old_type=joint.get('type'),
                            new_type='continuous', old_local_axis=old_axis.tolist(),
                            new_local_axis=new_axis.tolist(), old_root_axis=world.tolist(),
                            target_root_axis=desired[side].tolist(), old_target_dot=dot,
                            old_to_new_q_sign=None if dot is None else (1 if dot >= 0 else -1),
                            unoriented_axis_correction_deg=correction))
        replacements[name] = block
    assert len(changes) == 18
    text = re.sub(r'<joint\b[\s\S]*?</joint>',
                  lambda m: replacements.get(ET.fromstring(m[0]).get('name'), m[0]), raw.decode('utf-8'))
    changed = ET.fromstring(text)
    assert all(j.get('type') == 'continuous' for j in changed.findall('joint'))
    new_blocks = joint_blocks(text)
    for old, new in zip(original.findall('link'), changed.findall('link')):
        assert ET.tostring(old) == ET.tostring(new)
    for name in ('M_joint1', 'M_joint2'):
        assert old_blocks[name] == new_blocks[name]
    for joint, new_joint in zip(original.findall('joint'), changed.findall('joint')):
        # No changes to joint poses, parent/child links, limits or other properties.
        for old_el, new_el in zip(joint, new_joint):
            if old_el.tag != 'axis':
                assert ET.tostring(old_el) == ET.tostring(new_el)

    reader = csv.DictReader(io.StringIO(original_csv.decode('utf-8-sig'), newline=''))
    columns, rows = reader.fieldnames, list(reader)
    changes_by_name = {c['name']: c for c in changes}
    for row in rows:
        c = changes_by_name.get(row['Joint Name'])
        if c:
            row['Joint Type'] = c['new_type']
            for key, value in zip(('Joint Axis X', 'Joint Axis Y', 'Joint Axis Z'), c['new_local_axis']):
                row[key] = format(value, '.17g')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows)
    new_csv = stream.getvalue().encode('utf-8')
    names = [j.get('name') for j in changed.findall('joint')]
    assert len(names) == len(set(names)) == 20
    new_yaml = ('controller_joint_names: [' + ', '.join(repr(n) for n in names) + ']\r\n').encode('utf-8')
    assert ast.literal_eval(new_yaml.decode().split(':', 1)[1]) == names

    new_frames = fk(changed)
    zero_error = max(float(np.abs(frames[n] - new_frames[n]).max()) for n in frames)
    assert zero_error == 0.
    max_fd_error = 0.
    for c in changes:
        moved = fk(changed, {c['name']: 1e-3})[c['child']][:3, :3]
        omega = Rotation.from_matrix(moved @ new_frames[c['child']][:3, :3].T).as_rotvec() / 1e-3
        max_fd_error = max(max_fd_error, float(np.linalg.norm(omega - desired[c['side']])))
    rng = np.random.default_rng(20261029)
    joints = {j.get('name'): j for j in changed.findall('joint')}
    max_axis_error = 0.
    for _ in range(64):
        f = fk(changed, {c['name']: float(rng.uniform(-np.pi, np.pi)) for c in changes})
        for c in changes:
            actual = f[c['child']][:3, :3] @ vector(joints[c['name']].find('axis').get('xyz'))
            max_axis_error = max(max_axis_error, float(np.linalg.norm(actual - desired[c['side']])))
    assert max_fd_error < 1e-10 and max_axis_error < 1e-12
    receipt = dict(
        source_sha256=sha(raw), output_sha256=sha(text.encode('utf-8')),
        baseline_path=str(BASELINE), baseline_sha256=sha(baseline_bytes),
        helper_path=str(helper), helper_sha256=sha(helper.read_bytes()),
        scope='Leg joint axes, RR_joint1 mobility and matching CSV/YAML only',
        outward_root_axes={k: v.tolist() for k, v in desired.items()},
        changes=changes, inward_axes_reversed=sum(c['old_to_new_q_sign'] == -1 for c in changes),
        restored_movable_joint='RR_joint1', controller_joint_names=names,
        zero_pose_max_abs_transform_difference=zero_error,
        all_link_elements_identical=True, gimbal_joint_blocks_identical=True,
        positive_rotation_finite_difference_max_error=max_fd_error,
        independent_random_tree_poses=64, random_pose_max_axis_vector_error=max_axis_error)
    if not apply:
        print(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False))
        return
    backup = OUT / 'before'
    backup.mkdir(exist_ok=False)
    for p, data in ((URDF, raw), (CSV, original_csv), (YAML, original_yaml)):
        (backup / p.name).write_bytes(data)
    (OUT / 'package_before_sha256.json').write_text(
        json.dumps(original_manifest, indent=2, ensure_ascii=False) + '\n')
    URDF.write_bytes(text.encode('utf-8'))
    CSV.write_bytes(new_csv)
    YAML.write_bytes(new_yaml)
    after = manifest()
    changed_files = [p for p in original_manifest if original_manifest[p] != after[p]]
    assert set(after) == set(original_manifest)
    assert set(changed_files) == {str(p.relative_to(PACKAGE)) for p in (URDF, CSV, YAML)}
    assert sha(BASELINE.read_bytes()) == sha(baseline_bytes)
    receipt['changed_package_files'] = changed_files
    receipt['all_other_package_files_identical'] = True
    receipt['files'] = [dict(path=str(p), before_sha256=sha(data), after_sha256=sha(p.read_bytes()))
                        for p, data in ((URDF, raw), (CSV, original_csv), (YAML, original_yaml))]
    (OUT / 'axis_changes.json').write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ('changes', 'controller_joint_names', 'files')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
