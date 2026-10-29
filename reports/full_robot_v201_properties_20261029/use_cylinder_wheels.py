"""Apply the user's requested chassis-style cylindrical wheel colliders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
URDF = ROOT / 'model/wheel_leg_v2.0.1/轮腿urdf.SLDASM/urdf/轮腿urdf.SLDASM.urdf'
BASELINE = ROOT / 'model/纯底盘_v7_flat_base107/urdf/robot.urdf'
EXPECTED_SHA = 'b46af65f0b625b3dcbe996dd0339ff1b30d59f67b4b7da0ae7e6f395393e0a8b'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    raw = URDF.read_bytes()
    if digest(raw) != EXPECTED_SHA:
        raise RuntimeError('URDF differs from the axis-corrected v2.0.1 model')
    before = ET.fromstring(raw)
    baseline_bytes = BASELINE.read_bytes()
    baseline = ET.fromstring(baseline_bytes)
    old_blocks = re.findall(r'<link\b[\s\S]*?</link>', raw.decode())
    replacements, records = {}, []
    for name in ('L_link3', 'R_link3'):
        link = before.find(f"link[@name='{name}']")
        source = baseline.find(f"link[@name='{name}']/collision")
        origin = source.find('origin').attrib
        cylinder = source.find('geometry/cylinder').attrib
        mesh_path = URDF.parent.parent / link.find('visual/geometry/mesh').get('filename').split('/', 3)[3]
        mesh = trimesh.load_mesh(mesh_path, process=False)
        assert abs(float(cylinder['radius']) - np.linalg.norm(mesh.vertices[:, :2], axis=1).max()) < 1e-7
        assert abs(float(cylinder['length']) - np.ptp(mesh.vertices[:, 2])) < 1e-9
        assert abs(float(origin['xyz'].split()[2]) - mesh.bounds[:, 2].mean()) < 1e-9
        collision = ('<collision>\n'
                     '      <origin\n'
                     f'        xyz="{origin["xyz"]}"\n'
                     f'        rpy="{origin["rpy"]}" />\n'
                     '      <geometry>\n'
                     f'        <cylinder radius="{cylinder["radius"]}" length="{cylinder["length"]}" />\n'
                     '      </geometry>\n'
                     '    </collision>')
        block = next(b for b in old_blocks if ET.fromstring(b).get('name') == name)
        changed, count = re.subn(r'<collision\b[\s\S]*?</collision>', collision, block)
        assert count == 1
        replacements[name] = changed
        records.append(dict(name=name, radius_m=float(cylinder['radius']),
                            length_m=float(cylinder['length']), origin=origin,
                            scheme='Exact copy of corresponding chassis v7 cylinder',
                            visual_mesh_sha256=digest(mesh_path.read_bytes())))
    text = re.sub(r'<link\b[\s\S]*?</link>',
                  lambda m: replacements.get(ET.fromstring(m[0]).get('name'), m[0]), raw.decode())
    after = ET.fromstring(text)
    assert [ET.tostring(j) for j in before.findall('joint')] == [ET.tostring(j) for j in after.findall('joint')]
    for old, new in zip(before.findall('link'), after.findall('link')):
        assert ET.tostring(old.find('inertial')) == ET.tostring(new.find('inertial'))
        assert ET.tostring(old.find('visual')) == ET.tostring(new.find('visual'))
        if old.get('name') not in replacements:
            assert ET.tostring(old) == ET.tostring(new)
    backup = OUT / 'before_cylinder'
    backup.mkdir(exist_ok=False)
    (backup / URDF.name).write_bytes(raw)
    URDF.write_bytes(text.encode())
    assert digest(BASELINE.read_bytes()) == digest(baseline_bytes)
    receipt = dict(urdf=str(URDF), source_sha256=digest(raw), output_sha256=digest(URDF.read_bytes()),
                   baseline=str(BASELINE), baseline_sha256=digest(baseline_bytes), wheels=records,
                   all_inertial_elements_identical=True, all_visual_elements_identical=True,
                   all_joint_elements_identical=True, other_link_elements_identical=True,
                   csv_note='SW exporter CSV cannot express primitive cylinders; its wheel mesh columns remain CAD export provenance. Final collision definitions are in the URDF.')
    (OUT / 'wheel_collision_changes.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
