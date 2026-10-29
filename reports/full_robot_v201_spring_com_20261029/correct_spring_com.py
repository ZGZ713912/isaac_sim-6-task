"""Mirror the lower gas-spring COM using the old CAD value, not its mesh prior."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
URDF = ROOT / 'model/wheel_leg_v2.0.1/轮腿urdf.SLDASM/urdf/轮腿urdf.SLDASM.urdf'
CSV = URDF.with_suffix('.csv')
OLD = ROOT / 'model/纯底盘_v7_flat_base107/urdf'
EXPECTED_SHA = '93be05a257f6458cad709406c4081fc449787a91d7424e211a4459d6c641ff9b'
TARGET = 'LLL_link2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def vector(text):
    return np.array([float(x) for x in text.split()])


def tensor(link):
    a = {k: float(v) for k, v in link.find('inertial/inertia').attrib.items()}
    return np.array([[a['ixx'], a['ixy'], a['ixz']],
                     [a['ixy'], a['iyy'], a['iyz']],
                     [a['ixz'], a['iyz'], a['izz']]])


def main():
    raw, csv_raw = URDF.read_bytes(), CSV.read_bytes()
    if sha(raw) != EXPECTED_SHA:
        raise RuntimeError('Expected the axis-corrected model with cylindrical wheels')
    old_files = {p: p.read_bytes() for p in (
        OLD / 'robot.urdf', OLD / 'source_urdf_v5.0.urdf', OLD / 'inertial_sources.json')}
    cad = ET.fromstring(old_files[OLD / 'source_urdf_v5.0.urdf'])
    baseline = ET.fromstring(old_files[OLD / 'robot.urdf'])
    provenance = json.loads(old_files[OLD / 'inertial_sources.json'])
    before = ET.fromstring(raw)
    # Old rod coordinates use +Y; the new local frame uses +X.
    old_to_new = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    mirror = np.diag([1., 1., -1.])
    left = before.find(f"link[@name='{TARGET}']")
    right = before.find("link[@name='RRR_link2']")
    c0 = vector(left.find('inertial/origin').get('xyz'))
    cad_left = vector(cad.find(f"link[@name='{TARGET}']/inertial/origin").get('xyz'))
    c1 = old_to_new @ cad_left
    assert np.allclose(c1, mirror @ vector(right.find('inertial/origin').get('xyz')),
                       rtol=0., atol=1e-15)
    assert np.array_equal(c1, np.array([.070864, 0., -.0035025]))
    old_origin = b'xyz="0.060132 0 -0.0035121"'
    new_origin = b'xyz="0.070864 0 -0.0035025"'
    old_csv = b'LLL_link2,0.060132,0,-0.0035121,'
    new_csv = b'LLL_link2,0.070864,0,-0.0035025,'
    assert raw.count(old_origin) == csv_raw.count(old_csv) == 1
    output = raw.replace(old_origin, new_origin)
    csv_output = csv_raw.replace(old_csv, new_csv)
    after = ET.fromstring(output)
    # Verify the entire tree: only the selected inertial xyz may change.
    restored = deepcopy(after)
    restored.find(f"link[@name='{TARGET}']/inertial/origin").set('xyz', '0.060132 0 -0.0035121')
    assert ET.tostring(restored) == ET.tostring(before)
    rows_before = list(csv.DictReader(io.StringIO(csv_raw.decode(), newline='')))
    rows_after = list(csv.DictReader(io.StringIO(csv_output.decode(), newline='')))
    csv_changes = []
    for a, b in zip(rows_before, rows_after, strict=True):
        for key in a:
            if a[key] != b[key]:
                csv_changes.append(dict(link=a['Link Name'], field=key,
                                        before=a[key], after=b[key]))
    assert {(r['link'], r['field']) for r in csv_changes} == {
        (TARGET, 'Center of Mass X'), (TARGET, 'Center of Mass Z')}
    for link in after.findall('link'):
        eig = np.linalg.eigvalsh(tensor(link))
        assert eig[0] > 0 and eig[0] + eig[1] >= eig[2] - 1e-14
    left_after = after.find(f"link[@name='{TARGET}']")
    assert np.array_equal(tensor(left_after), mirror @ tensor(right) @ mirror)
    assert len(after.findall('link')) == 21
    assert len(after.findall('joint')) == 20
    assert len(after.findall('link/collision/geometry/cylinder')) == 2
    m = float(left.find('inertial/mass').get('value'))
    assert m == float(right.find('inertial/mass').get('value')) == .057
    total_mass = sum(float(l.find('inertial/mass').get('value'))
                     for l in after.findall('link'))
    I_C = tensor(left)
    shift = lambda c: I_C + m * (float(c @ c) * np.eye(3) - np.outer(c, c))
    I_O_before, I_O_after = shift(c0), shift(c1)
    hinge = next(j for j in after.findall('joint') if j.find('child').get('link') == TARGET)
    axis = vector(hinge.find('axis').get('xyz'))
    axis /= np.linalg.norm(axis)
    J_before, J_after = float(axis @ I_O_before @ axis), float(axis @ I_O_after @ axis)
    source_coms = {}
    for name in (TARGET, 'RRR_link2'):
        source_coms[name] = dict(
            raw_CAD_com_m=vector(cad.find(f"link[@name='{name}']/inertial/origin").get('xyz')).tolist(),
            processed_v7_com_m=vector(baseline.find(f"link[@name='{name}']/inertial/origin").get('xyz')).tolist(),
            processed_v7_method=provenance[name]['method'],
            hardware_identified=provenance[name]['hardware_identified'])
    receipt = dict(
        urdf=str(URDF), source_sha256=sha(raw), output_sha256=sha(output),
        csv=str(CSV), csv_source_sha256=sha(csv_raw), csv_output_sha256=sha(csv_output),
        assumption='Identical mirrored lower spring parts; inherit 70.864 mm from old raw CAD and current right side. Not a new physical COM measurement.',
        source_coms=source_coms, old_frame_to_new_frame_rotation=old_to_new.tolist(),
        changed_link=TARGET, com_before_m=c0.tolist(), com_after_m=c1.tolist(),
        mass_kg=m, total_robot_mass_kg=total_mass,
        central_tensor_preserved_kg_m2=I_C.tolist(),
        connection_hinge=dict(joint=hinge.get('name'), local_axis=axis.tolist(),
            tensor_before_kg_m2=I_O_before.tolist(), tensor_after_kg_m2=I_O_after.tolist(),
            J_before_kg_m2=J_before, J_after_kg_m2=J_after,
            change_percent=(J_after / J_before - 1.) * 100.,
            scope='Contribution of this 57 g part only, not the whole leg or motor-equivalent inertia'),
        max_gravity_torque_change_Nm=m * 9.80665 * float(np.linalg.norm(c1 - c0)),
        whole_robot_com_change_upper_bound_m=m * float(np.linalg.norm(c1 - c0)) / total_mass,
        validation=dict(only_target_inertial_xyz_changed=True, csv_changes=csv_changes,
            all_masses_and_central_tensors_preserved=True, all_21_tensors_positive_definite_and_triangle_valid=True,
            lower_spring_COM_mirrored_atol_m=1e-15, lower_spring_tensors_exactly_mirrored=True,
            geometry_collisions_and_all_joint_elements_preserved=True,
            cylindrical_wheel_colliders=2, old_baseline_files_preserved=True,
            simulation_run=False, hardware_measured=False),
        baseline_sha256={str(p): sha(b) for p, b in old_files.items()},
        references=[
            'https://github.com/ros2/ros2_documentation/blob/rolling/source/Capabilities/Simulation/URDF/Adding-Physical-and-Collision-Properties-to-a-URDF-Model.rst',
            'https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_rotational_inertia.html'])
    backup = OUT / 'before'
    backup.mkdir(exist_ok=False)
    (backup / URDF.name).write_bytes(raw)
    (backup / CSV.name).write_bytes(csv_raw)
    URDF.write_bytes(output)
    CSV.write_bytes(csv_output)
    assert URDF.read_bytes() == output and CSV.read_bytes() == csv_output
    assert all(p.read_bytes() == b for p, b in old_files.items())
    (OUT / 'correction.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('output_sha256', 'changed_link',
                                            'com_after_m', 'connection_hinge', 'validation')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
