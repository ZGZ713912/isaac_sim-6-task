"""Normalize full-vehicle leg axes; preserve geometry, inertia and source backup.

No simulator or hardware access. --apply performs the explicitly requested edit.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'model/wheel_legv2.0.0'
URDF = PACKAGE / 'urdf/轮腿urdf.SLDASM.urdf'
CSV = URDF.with_suffix('.csv')
YAML = PACKAGE / 'config/joint_names_轮腿urdf.SLDASM.yaml'
BASELINE = ROOT / 'model/纯底盘_v7_flat_base107/urdf/robot.urdf'
EXPECTED_ORIGINAL_SHA = '777d875faaf4c150225f606076cae5a7e827b34dc3eb02b31712f2c1b9a601f4'
ALIGN = Rotation.from_euler('z', 90, degrees=True).as_matrix()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def vector(text):
    return np.fromstring(text or '0 0 0', sep=' ')


def fk(robot, q=None):
    q = q or {}
    frames = {'base_link': np.eye(4)}
    pending = list(robot.findall('joint'))
    while pending:
        count = len(pending)
        for j in pending[:]:
            parent = j.find('parent').get('link')
            if parent not in frames:
                continue
            child = j.find('child').get('link')
            o = j.find('origin')
            t = np.eye(4)
            t[:3, :3] = Rotation.from_euler('xyz', vector(o.get('rpy'))).as_matrix()
            t[:3, 3] = vector(o.get('xyz'))
            if j.get('type') in ('continuous', 'revolute'):
                a = vector(j.find('axis').get('xyz'))
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(a / np.linalg.norm(a) * q.get(j.get('name'), 0.)).as_matrix()
            elif j.get('type') == 'prismatic':
                a = vector(j.find('axis').get('xyz'))
                t[:3, 3] += t[:3, :3] @ a * q.get(j.get('name'), 0.)
            frames[child] = frames[parent] @ t
            pending.remove(j)
        if len(pending) == count:
            raise ValueError('Unresolved joint tree')
    return frames


def joint_blocks(text):
    return {ET.fromstring(b).get('name'): b for b in re.findall(r'<joint\b[\s\S]*?</joint>', text)}


def main(apply):
    raw = URDF.read_bytes()
    if sha(raw) != EXPECTED_ORIGINAL_SHA:
        raise RuntimeError('Full URDF differs from the reviewed original; refusing to overwrite')
    original = ET.fromstring(raw)
    old_frames = fk(original)
    baseline_bytes = BASELINE.read_bytes()
    baseline = ET.fromstring(baseline_bytes)
    bf = fk(baseline)
    bj = {j.get('name'): j for j in baseline.findall('joint')}
    j = bj['L_joint1']
    outward_left = ALIGN.T @ (bf[j.find('child').get('link')][:3, :3] @ vector(j.find('axis').get('xyz')))
    outward_left /= np.linalg.norm(outward_left)
    assert outward_left[0] > .99
    desired = {'L': outward_left, 'R': -outward_left}
    changes = []
    new_blocks = {}
    before_blocks = joint_blocks(raw.decode('utf-8'))
    for j in original.findall('joint'):
        name = j.get('name')
        child = j.find('child').get('link')
        if child.startswith('M_'):
            continue
        side = child[0]
        old_axis = vector(j.find('axis').get('xyz'))
        r = old_frames[child][:3, :3]
        old_world = r @ old_axis
        new_axis = r.T @ desired[side]
        new_axis /= np.linalg.norm(new_axis)
        new_axis[np.abs(new_axis) < 5e-15] = 0.
        new_axis /= np.linalg.norm(new_axis)
        block, n = re.subn(r'(<axis\s+xyz=")[^"]*(")',
                           lambda m: m[1] + ' '.join(format(x, '.17g') for x in new_axis) + m[2],
                           before_blocks[name])
        assert n == 1
        if name == 'RR_joint1':
            block, n = re.subn(r'type="fixed"', 'type="continuous"', block, count=1)
            assert n == 1
        old_dot = float(old_world @ desired[side] / np.linalg.norm(old_world))
        changes.append(dict(name=name, child=child, side=side, old_type=j.get('type'),
                            new_type='continuous' if name == 'RR_joint1' else j.get('type'),
                            old_local_axis=old_axis.tolist(), new_local_axis=new_axis.tolist(),
                            target_root_axis=desired[side].tolist(), old_target_dot=old_dot,
                            old_to_new_q_sign=None if name == 'RR_joint1' else (1 if old_dot >= 0 else -1),
                            unoriented_axis_correction_deg=float(np.degrees(np.arctan2(
                                np.linalg.norm(np.cross(old_world, desired[side])), abs(old_world @ desired[side]))))))
        new_blocks[name] = block
    text = re.sub(r'<joint\b[\s\S]*?</joint>',
                  lambda m: new_blocks.get(ET.fromstring(m[0]).get('name'), m[0]), raw.decode('utf-8'))
    changed = ET.fromstring(text)
    assert len(changes) == 18
    original_csv = CSV.read_bytes()
    reader = csv.DictReader(io.StringIO(original_csv.decode('utf-8'), newline=''))
    columns = reader.fieldnames
    rows = list(reader)
    change_map = {c['name']: c for c in changes}
    for row in rows:
        c = change_map.get(row['Joint Name'])
        if c:
            row['Joint Type'] = c['new_type']
            for k, x in zip(('Joint Axis X', 'Joint Axis Y', 'Joint Axis Z'), c['new_local_axis']):
                row[k] = format(x, '.17g')
    s = io.StringIO(newline='')
    writer = csv.DictWriter(s, fieldnames=columns, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = s.getvalue().encode('utf-8')
    original_yaml = YAML.read_bytes()
    names = [j.get('name') for j in changed.findall('joint') if j.get('type') != 'fixed']
    yaml_bytes = ('controller_joint_names: [' + ', '.join(repr(n) for n in names) + ']\r\n').encode('utf-8')

    # Meaningful geometric checks: zero pose, all positive-rotation directions,
    # and axes throughout independent random tree poses; no loop is introduced.
    new_frames = fk(changed)
    zero_error = max(float(np.max(np.abs(old_frames[n] - new_frames[n]))) for n in old_frames)
    assert zero_error == 0.
    for l, nl in zip(original.findall('link'), changed.findall('link')):
        assert ET.tostring(l) == ET.tostring(nl)
    cb = joint_blocks(text)
    for n in ('M_joint1', 'M_joint2'):
        assert cb[n] == before_blocks[n]
    max_fd_error = 0.
    for c in changes:
        moved = fk(changed, {c['name']: 1e-3})[c['child']][:3, :3]
        rotvec = Rotation.from_matrix(moved @ new_frames[c['child']][:3, :3].T).as_rotvec() / 1e-3
        max_fd_error = max(max_fd_error, float(np.linalg.norm(rotvec - desired[c['side']])))
    rng = np.random.default_rng(20261029)
    max_axis_error = 0.
    cj = {j.get('name'): j for j in changed.findall('joint')}
    for _ in range(64):
        f = fk(changed, {c['name']: float(rng.uniform(-np.pi, np.pi)) for c in changes})
        for c in changes:
            a = vector(cj[c['name']].find('axis').get('xyz'))
            w = f[c['child']][:3, :3] @ a
            max_axis_error = max(max_axis_error, float(np.linalg.norm(w - desired[c['side']])))
    assert max_fd_error < 1e-10 and max_axis_error < 1e-12
    assert ast.literal_eval(yaml_bytes.decode().split(':', 1)[1]) == names
    assert len(names) == 20 and len(set(names)) == 20

    if not apply:
        print(json.dumps(dict(proposed_changes=changes, zero_pose_error=zero_error), indent=2))
        return
    backup = OUT / 'before'
    backup.mkdir(parents=True, exist_ok=False)
    for p, b in ((URDF, raw), (CSV, original_csv), (YAML, original_yaml)):
        (backup / p.name).write_bytes(b)
    URDF.write_bytes(text.encode('utf-8'))
    CSV.write_bytes(csv_bytes)
    YAML.write_bytes(yaml_bytes)
    assert sha(BASELINE.read_bytes()) == sha(baseline_bytes)
    receipt = dict(scope='Leg rotation axes only; no closed-loop, inertia, mesh, root-frame or gimbal changes',
                   source_sha256=sha(raw), output_sha256=sha(URDF.read_bytes()),
                   baseline=str(BASELINE), baseline_sha256=sha(baseline_bytes),
                   baseline_root_to_full_root=ALIGN.T.tolist(), outward_full_root_axes={k: v.tolist() for k, v in desired.items()},
                   changes=changes, changed_leg_joint_count=18, restored_movable_joint='RR_joint1',
                   zero_pose_max_abs_transform_difference=zero_error,
                   all_link_elements_identical=True, gimbal_joint_blocks_identical=True,
                   positive_rotation_finite_difference_max_error=max_fd_error,
                   independent_random_tree_poses=64, random_pose_max_axis_vector_error=max_axis_error,
                   controller_joint_names=names,
                   files=[dict(path=str(p), before_sha256=sha(b), after_sha256=sha(p.read_bytes()))
                          for p, b in ((URDF, raw), (CSV, original_csv), (YAML, original_yaml))])
    (OUT / 'axis_changes.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('changes', 'files', 'controller_joint_names')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
