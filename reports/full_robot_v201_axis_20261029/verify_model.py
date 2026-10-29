"""Read back the corrected v2.0.1 export, validate assets, and write the review."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'model/wheel_leg_v2.0.1/轮腿urdf.SLDASM'
URDF = PACKAGE / 'urdf/轮腿urdf.SLDASM.urdf'
CSV = URDF.with_suffix('.csv')
YAML = PACKAGE / 'config/joint_names_轮腿urdf.SLDASM.yaml'
PREVIOUS = ROOT / 'model/wheel_legv2.0.0/urdf/轮腿urdf.SLDASM.urdf'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(link):
    a = {k: float(v) for k, v in link.find('inertial/inertia').attrib.items()}
    return np.array([[a['ixx'], a['ixy'], a['ixz']],
                     [a['ixy'], a['iyy'], a['iyz']],
                     [a['ixz'], a['iyz'], a['izz']]])


def main():
    robot = ET.parse(URDF).getroot()
    original = ET.parse(OUT / 'before' / URDF.name).getroot()
    previous = ET.parse(PREVIOUS).getroot()
    links = {e.get('name'): e for e in robot.findall('link')}
    joints = {e.get('name'): e for e in robot.findall('joint')}
    old_joints = {e.get('name'): e for e in original.findall('joint')}
    before_links = {e.get('name'): e for e in original.findall('link')}
    previous_links = {e.get('name'): e for e in previous.findall('link')}
    receipt = json.loads((OUT / 'axis_changes.json').read_text())
    assert digest(URDF) == receipt['output_sha256']
    assert len(links) == len(robot.findall('link')) == 21
    assert len(joints) == len(robot.findall('joint')) == 20
    assert all(ET.tostring(links[n]) == ET.tostring(before_links[n]) for n in links)
    children = [e.find('child').get('link') for e in joints.values()]
    assert len(set(children)) == len(children)
    assert set(links) - set(children) == {'base_link'}
    reached = {'base_link'}
    while True:
        nxt = reached | {j.find('child').get('link') for j in joints.values()
                         if j.find('parent').get('link') in reached}
        if nxt == reached:
            break
        reached = nxt
    assert reached == set(links)
    assert all(j.get('type') == 'continuous' for j in joints.values())
    assert all(np.isclose(np.linalg.norm(np.fromstring(j.find('axis').get('xyz'), sep=' ')), 1.)
               for j in joints.values())
    assert old_joints['RR_joint1'].get('type') == 'fixed'
    assert joints['RR_joint1'].get('type') == 'continuous'

    with CSV.open(encoding='utf-8-sig', newline='') as stream:
        current_rows = list(csv.DictReader(stream))
    with (OUT / 'before' / CSV.name).open(encoding='utf-8-sig', newline='') as stream:
        before_rows = list(csv.DictReader(stream))
    assert len(current_rows) == len(before_rows) == 21
    allowed_csv = {'Joint Axis X', 'Joint Axis Y', 'Joint Axis Z', 'Joint Type'}
    csv_changed_fields = set()
    for old, new in zip(before_rows, current_rows):
        assert old['Link Name'] == new['Link Name']
        fields = {key for key in old if old[key] != new[key]}
        assert fields <= allowed_csv
        csv_changed_fields |= fields
        name = new['Joint Name']
        if name:
            joint = joints[name]
            assert joint.get('type') == new['Joint Type']
            assert np.allclose(np.fromstring(joint.find('axis').get('xyz'), sep=' '),
                               [float(new[k]) for k in ('Joint Axis X', 'Joint Axis Y', 'Joint Axis Z')],
                               atol=0, rtol=0)
    names = ast.literal_eval(YAML.read_text().split(':', 1)[1])
    assert names == list(joints) and '' not in names and len(set(names)) == 20

    inertias, restored, mesh_records, wheels = [], [], [], []
    for name, link in links.items():
        I = tensor(link)
        mass = float(link.find('inertial/mass').get('value'))
        eigenvalues = np.linalg.eigvalsh(I)
        margin = float(eigenvalues[0] + eigenvalues[1] - eigenvalues[2])
        assert mass > 0 and np.isfinite(I).all()
        assert eigenvalues[0] > 0 and margin >= -1e-12
        inertias.append(dict(name=name, mass_kg=mass, tensor_kg_m2=I.tolist(),
                             principal_moments_kg_m2=eigenvalues.tolist(), triangle_margin_kg_m2=margin))
        previous_I = tensor(previous_links[name])
        for index, key in enumerate(('ixx', 'iyy', 'izz')):
            if previous_I[index, index] == 0:
                assert I[index, index] > 0
                restored.append(dict(name=name, component=key, previous_value=0.,
                                     new_value=float(I[index, index])))
        for mesh in link.findall('.//mesh'):
            uri = mesh.get('filename')
            relative = uri.split('/', 3)[3] if uri.startswith('package://') else uri
            path = PACKAGE / relative if uri.startswith('package://') else URDF.parent / relative
            assert path.is_file()
            mesh_records.append(dict(link=name, uri=uri, path=str(path.resolve()), sha256=digest(path)))
    assert len(restored) == 16
    for name in ('L_link3', 'R_link3'):
        for label, urdf, package in (('v2.0.1', URDF, PACKAGE),
                                     ('v2.0.0', PREVIOUS, PREVIOUS.parent.parent)):
            robot_part = ET.parse(urdf).getroot().find(f"link[@name='{name}']")
            element = robot_part.find('visual/geometry/mesh')
            uri = element.get('filename')
            mesh_path = package / uri.split('/', 3)[3]
            mesh = trimesh.load_mesh(mesh_path, process=False)
            scale = np.fromstring(element.get('scale', '1 1 1'), sep=' ')
            vertices = np.asarray(mesh.vertices) * scale
            radius = float(np.linalg.norm(vertices[:, :2], axis=1).max())
            wheels.append(dict(model=label, name=name, radius_m=radius,
                               diameter_m=radius * 2, mesh_path=str(mesh_path), sha256=digest(mesh_path)))
            assert abs(radius - (.06 if label == 'v2.0.1' else .0525)) < 1e-7

    before_manifest = json.loads((OUT / 'package_before_sha256.json').read_text())
    after_manifest = {str(p.relative_to(PACKAGE)): digest(p)
                      for p in sorted(PACKAGE.rglob('*')) if p.is_file()}
    assert before_manifest.keys() == after_manifest.keys()
    modified = [p for p in before_manifest if before_manifest[p] != after_manifest[p]]
    assert set(modified) == {str(p.relative_to(PACKAGE)) for p in (URDF, CSV, YAML)}
    result = dict(
        urdf=str(URDF), urdf_sha256=digest(URDF), links=21, joints=20,
        joint_types={'continuous': 20}, leg_joint_count=18, rooted_tree_valid=True,
        all_axes_unit_length=True, controller_joint_names_match=True,
        csv_joint_fields_match_urdf=True, csv_changed_fields=sorted(csv_changed_fields),
        original_link_elements_identical=True, only_expected_package_files_changed=True,
        modified_package_files=modified, inertia_positive_definite_count=21,
        inertia_triangle_inequality_pass_count=21, zero_diagonal_count=0,
        previous_zero_diagonal_components_restored=restored,
        minimum_principal_moment_kg_m2=min(e['principal_moments_kg_m2'][0] for e in inertias),
        total_mass_kg=sum(e['mass_kg'] for e in inertias), inertias=inertias,
        all_mesh_references_resolve=True, mesh_references=mesh_records, wheels=wheels,
        kinematic_checks=receipt,
        validation_scope='Offline XML/CSV/YAML, FK, tensor and mesh checks; no simulator or hardware run')
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    table = []
    for change in receipt['changes']:
        if change['old_type'] == 'fixed':
            action = 'fixed / 零轴 → continuous / 朝外'
        elif change['old_to_new_q_sign'] == -1:
            action = '向内 → 朝外，q/dq/力矩正方向反号'
        else:
            action = '保持朝外，并统一微小轴线偏差'
        table.append(f"| {change['name']} | {action} |")
    body = f'''# 全车 v2.0.1 关节轴统一与导出检查

日期：2026-10-29。源目录为 `{PACKAGE}`。

## 修改结果

- 18 个腿部转动关节统一为与纯底盘 v7 相同的左右朝外约定；10 个原向内关节反号。
- `RR_joint1` 原为 fixed 且 axis=[0,0,0]，现恢复 continuous 并设置右侧朝外轴。本次只发现这 1 个误设 fixed 的关节。
- 同步 URDF、配套 CSV 和 YAML；清除 YAML 开头空名称、补回 RR_joint1，清单与 20 个可动关节完全对应。
- 21 个 link 元素、质量、质心、惯量、网格、碰撞、关节 origin、limit 和云台两关节均保留本次 SW 导出值；全部包文件哈希检查确认仅上述 3 份文件改变。
- root 沿用 CAD 坐标；朝外方向在全车 root 中左侧约为 `[+0.999989403, +0.000003673, -0.004603657]`，右侧反向，与此前修正的 v2.0.0 一致。局部 axis 需按各 joint frame 计算，不能全部机械填写同一向量。

## 小数精度与轮径

- 21/21 惯量张量正定，21/21 满足主惯量三角不等式，最小主惯量为 {result['minimum_principal_moment_kg_m2']:.12g} kg·m²。
- v2.0.0 中原为零的 16 个对角分量已全部恢复非零；两件气弹簧上半件现为 `(4.8e-7, 3.039e-5, 3.064e-5)` kg·m²，并保留非零惯性积。
- 两轮 STL 的径向最大半径约 60.000 mm，即直径 120.000 mm；v2.0.0 为半径 52.500 mm、直径 105.000 mm。这里读取的是 URDF 实际引用网格及其 scale。
- 新轮惯量对角项为 `(0.000301, 0.000301, 0.00058325)` kg·m²；全车总质量 {result['total_mass_kg']:.8g} kg。本次没有覆写任何物性。

## 逐关节处理

| 关节名 | 处理 |
|---|---|
{chr(10).join(table)}

既有名称 `R_jonit2`、`LLL_jointt1`、`LL_link4` 沿用源文件，避免额外改变关节接口。

## 验证与边界

- 修改前后 21 个 link 的零位变换最大差为 0。
- 18 个关节分别做正角度有限差分，方向误差峰值 {receipt['positive_rotation_finite_difference_max_error']:.3g}。
- 64 组独立随机关节姿态中，18 条腿轴均保持对应侧朝外，方向误差峰值 {receipt['random_pose_max_axis_vector_error']:.3g}。
- XML 树、所有网格引用、CSV 字段、YAML 清单、全部惯量基本物理检查均通过。
- 本次为离线模型检查，没有启动仿真、训练或实车；闭链与气弹簧伸缩副仍按此前约定留待后续。
- 轴反号关节的 q、dq、力矩正方向对应变化见回执，已有这些关节的非零轨迹应同步换号。没有修改已存在的训练资产或控制配置。

## 原始文件与回执

- 原始 SW 导出 URDF SHA-256：`{receipt['source_sha256']}`。
- 修正后 URDF SHA-256：`{receipt['output_sha256']}`。
- [修正后的 URDF]({URDF})
- [关节清单]({YAML})
- [原始导出备份]({OUT / 'before' / URDF.name})
- [逐关节修改与方向映射]({OUT / 'axis_changes.json'})
- [完整验证与物性/网格数据]({OUT / 'validation.json'})
'''
    (OUT / 'REVIEW.md').write_text(body)
    print(json.dumps({k: result[k] for k in (
        'urdf_sha256', 'links', 'joints', 'joint_types', 'inertia_positive_definite_count',
        'inertia_triangle_inequality_pass_count', 'zero_diagonal_count',
        'minimum_principal_moment_kg_m2', 'total_mass_kg',
        'all_mesh_references_resolve', 'controller_joint_names_match',
        'csv_joint_fields_match_urdf', 'only_expected_package_files_changed')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
