"""Compare COM inertias in a common physical frame; never edit inertial data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from update_axes import ALIGN, BASELINE, OUT, PACKAGE, ROOT, URDF, Rotation, fk, vector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def props(link):
    inertial = link.find('inertial')
    o = inertial.find('origin')
    mass = float(inertial.find('mass').get('value'))
    a = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
    matrix = np.array([[a['ixx'], a['ixy'], a['ixz']],
                       [a['ixy'], a['iyy'], a['iyz']],
                       [a['ixz'], a['iyz'], a['izz']]])
    r = Rotation.from_euler('xyz', vector(o.get('rpy'))).as_matrix()
    return mass, vector(o.get('xyz')), r @ matrix @ r.T


def elements(robot):
    return {l.get('name'): l for l in robot.findall('link')}


def valid(tensor):
    eig = np.linalg.eigvalsh(tensor)
    return dict(principal_moments_kg_m2=eig.tolist(), positive_definite=bool(eig[0] > 0),
                triangle_valid=bool(eig[0] + eig[1] >= eig[2] - 1e-12))


def about_point(tensor, mass, com):
    return tensor + mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))


def main():
    paths = dict(full=URDF, baseline_v7=BASELINE,
                 baseline_v6=ROOT/'model/纯底盘_v6_232mm/urdf/robot.urdf',
                 historical_v5=ROOT/'model/纯底盘_v5_232mm/urdf/robot.urdf')
    robots = {k: ET.parse(p).getroot() for k, p in paths.items()}
    links = {k: elements(r) for k, r in robots.items()}
    frames = {k: fk(r) for k, r in robots.items()}
    joints = {j.find('child').get('link'): j for j in robots['full'].findall('joint')}
    provenance_path = BASELINE.parent/'inertial_sources.json'
    provenance = json.loads(provenance_path.read_text())
    same_baselines = all(ET.tostring(links['baseline_v6'][n].find('inertial')) ==
                         ET.tostring(l.find('inertial')) for n, l in links['baseline_v7'].items())
    assert same_baselines
    entries = []
    for name, link in links['baseline_v7'].items():
        if name == 'base_link':
            continue
        nm, nc, ni = props(links['full'][name])
        om, oc, oi = props(link)
        nf = frames['full'][name]
        of = frames['baseline_v7'][name]
        full_r_aligned = ALIGN @ nf[:3, :3]
        q = full_r_aligned.T @ of[:3, :3]
        offset = full_r_aligned.T @ (of[:3, 3] - ALIGN @ nf[:3, 3])
        old_i = q @ oi @ q.T
        old_c = q @ oc + offset
        axis = vector(joints[name].find('axis').get('xyz'))
        axis /= np.linalg.norm(axis)
        old_axis_i = float(axis @ old_i @ axis)
        new_axis_i = float(axis @ ni @ axis)
        old_p = float(axis @ about_point(old_i, om, old_c) @ axis)
        new_p = float(axis @ about_point(ni, nm, nc) @ axis)
        st = provenance[name]
        entry = dict(name=name, old_mass_kg=om, full_mass_kg=nm, mass_delta_g=(nm-om)*1000,
                     mass_change_percent=(nm/om-1)*100,
                     baseline_method=st['method'], baseline_hardware_identified=st.get('hardware_identified', False),
                     baseline_raw_source_com_m=st['source_com_m'],
                     baseline_raw_source_tensor=st['source_tensor'],
                     full_com_m=nc.tolist(), old_com_in_full_link_m=old_c.tolist(),
                     com_delta_mm=((nc-old_c)*1000).tolist(), com_distance_mm=float(np.linalg.norm(nc-old_c)*1000),
                     old_link_to_full_link_rotation=q.tolist(), old_link_to_full_link_translation_m=offset.tolist(),
                     full_com_tensor_kg_m2=ni.tolist(), old_com_tensor_in_full_link_kg_m2=old_i.tolist(),
                     full_validation=valid(ni), old_validation=valid(old_i),
                     full_axis_local=axis.tolist(), old_axis_parallel_com_inertia_kg_m2=old_axis_i,
                     full_axis_parallel_com_inertia_kg_m2=new_axis_i,
                     axis_parallel_com_inertia_change_percent=(new_axis_i/old_axis_i-1)*100,
                     old_about_full_joint_origin_inertia_kg_m2=old_p, full_about_joint_origin_inertia_kg_m2=new_p,
                     joint_origin_inertia_change_percent=(new_p/old_p-1)*100)
        mesh_path = PACKAGE/'meshes'/f'{name}.STL'
        mesh = trimesh.load_mesh(mesh_path, process=True)
        mesh_data = dict(path=str(mesh_path), sha256=digest(mesh_path),
                         watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                         volume_m3=float(mesh.volume), assumption='Uniform density, current full mesh and current CAD mass; diagnostic only')
        if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0:
            it = mesh.moment_inertia * nm/mesh.volume
            mesh_data.update(com_m=mesh.center_mass.tolist(), com_tensor_kg_m2=it.tolist(),
                             axis_parallel_com_inertia_kg_m2=float(axis @ it @ axis),
                             validation=valid(it),
                             axis_inertia_change_from_old_percent=(float(axis @ it @ axis)/old_axis_i-1)*100)
        entry['full_owned_mesh_diagnostic'] = mesh_data
        entries.append(entry)
    summary = dict(
        baseline_v6_v7_inertial_elements_identical=same_baselines,
        old_leg_mass_kg=sum(e['old_mass_kg'] for e in entries),
        full_leg_mass_kg=sum(e['full_mass_kg'] for e in entries),
        leg_mass_delta_g=sum(e['mass_delta_g'] for e in entries),
        old_base_mass_kg=props(links['baseline_v7']['base_link'])[0],
        full_base_mass_kg=props(links['full']['base_link'])[0],
        full_gimbal_mass_kg=sum(props(l)[0] for n,l in links['full'].items() if n.startswith('M_')),
        full_invalid_links=[e['name'] for e in entries if not all(e['full_validation'][k] for k in ('positive_definite','triangle_valid'))],
        baseline_mesh_prior_links=[e['name'] for e in entries if 'research_prior' in e['baseline_method']],
        historical_v5_RR_link2_mass_kg=props(links['historical_v5']['RR_link2'])[0],
        baseline_v6_v7_RR_link2_mass_kg=props(links['baseline_v7']['RR_link2'])[0],
        full_RR_link2_mass_kg=props(links['full']['RR_link2'])[0])
    data = dict(scope='Read-only mass, COM and inertia comparison; no closed-loop or inertia modifications',
                tensor_rule='Old tensor rotated into current full link frame; both tensors about each own COM, not about root or joint origin',
                joint_origin_rule='Separate parallel-axis comparison about current full joint origin',
                inputs=[dict(name=k,path=str(p),sha256=digest(p)) for k,p in paths.items()]
                       +[dict(name='inertial_provenance',path=str(provenance_path),sha256=digest(provenance_path))],
                summary=summary, links=entries)
    (OUT/'inertia_comparison.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write_report(data)
    plot_axes(robots['full'])
    plot_inertia(entries)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def write_report(data):
    axis = json.loads((OUT/'axis_changes.json').read_text())
    s = data['summary']
    rows = {e['name']:e for e in data['links']}
    report = [
        '# 全车腿轴统一朝外与连杆质量/惯量对比', '',
        '日期：2026-10-29。按本轮要求已修改腿部关节轴，闭链留待后续；惯量只做对比，本轮未改数值。', '',
        '## 已完成的轴向修改', '',
        f'- 修改文件：[全车 URDF]({URDF})、配套导出 CSV 和控制关节清单。原始三文件保存在本报告目录 `before/`。',
        '- 18 个腿部旋转关节采用旧 V7 的同侧统一朝外约定，包含主腿、传动杆、轮和气簧铰。',
        '- 原误设为 fixed 的 RR_joint1 恢复为 continuous，校正其约 15° 的轴线偏差，并补回关节清单。',
        '- 当前全车 root 约定下，左侧轴约 `[+0.999989403, +0.000003673, -0.004603657]`，右侧取相反方向。这里左右沿 root X；没有改 root 坐标。',
        '- 保留共同铰轴相对 root 的 CAD 小倾角，未擅自将实体腿铰转成严格水平。',
        '- 只修改 joint axis / RR_joint1 type 及配套清单。所有 link 元素（质量、质心、惯量、visual、collision）、关节 origin、云台关节和网格均保持原值。',
        f'- 校验：零位 21 个 link 的变换差为 {axis["zero_pose_max_abs_transform_difference"]:.1f}；64 组随机树姿态的最大轴向向量误差 {axis["random_pose_max_axis_vector_error"]:.3g}；逐关节正向微转的方向误差 {axis["positive_rotation_finite_difference_max_error"]:.3g}。',
        '- 旧 V6/V7 等模型保持原文件；未运行物理步或接入实车。',
        '- 新旧 q/dq/力矩的符号与微小轴线校正逐关节记录在 [axis_changes.json]('+str(OUT/'axis_changes.json')+')；RR_joint1 原本没有运动自由度，不能给它定义旧运动坐标的简单翻号。', '',
        '![轴向修改前后]('+str(OUT/'axes_before_after.png')+')', '',
        '## 质量：腿部只差 2 克', '',
        '**以下以实际处理后的 V6/V7 底盘为主对照。两者全部惯性元素逐字节相同，因此本表同时适用于 V6 和 V7。**', '',
        '| 同类构件（左右每件） | 旧底盘 kg | 全车 kg | 每件变化 g |', '|---|---:|---:|---:|']
    groups = [('L/R_link1 大腿','L_link1'),('L/R_link2 主小腿','L_link2'),('L/R_link3 轮','L_link3'),
              ('LL/RR_link1 输入曲柄','LL_link1'),('LL/RR_link2 传动杆','LL_link2'),
              ('LL/RR_link3 摇杆','LL_link3'),('LL/RR_link4 末杆','LL_link4'),
              ('LLL/RRR_link1 气簧上半件','LLL_link1'),('LLL/RRR_link2 气簧下半件','LLL_link2')]
    for label,name in groups:
        e=rows[name]
        report.append(f'| {label} | {e["old_mass_kg"]:.4f} | {e["full_mass_kg"]:.4f} | {e["mass_delta_g"]:+.3f} |')
    report += ['', f'两侧全部腿部（含轮、传动杆、气簧）合计：**{s["old_leg_mass_kg"]:.3f} → {s["full_leg_mass_kg"]:.3f} kg**，减少 {abs(s["leg_mass_delta_g"]):.3f} g，约 0.0883%。',
               '', '全车比旧底盘增加的质量主要来自 base_link：11.620 → 16.904 kg，以及新增的 2.2738 kg 显式云台；不是腿部质量显著增加。', '',
               '历史 V5 / V5_232mm 的 RR_link2 曾为 0.840 kg；V6/V7 已改为 0.084 kg，新全车为 0.0836 kg。不能把 V5 的这项历史异常和 V6/V7 的正常值混为同一个对照。', '',
               '## 惯量对比的坐标与来源', '',
               '- 每个张量都以各自质心为参考点；先把旧底盘惯量旋转到新全车同名 link 的坐标系，再比较。没有直接混用旧长杆 Y 向和新长杆 X 向的 Ixx/Iyy。',
               '- 表中 J 是**穿过各自质心、平行该侧铰轴**的转动惯量；不是整条腿的等效惯量，也不是关于电机轴原点的惯量。完整张量、主惯量、质心差以及单独使用平行轴定理换算到相同关节原点的结果均见 JSON。',
               '- 旧底盘 10 件惯量来自 `owned_mesh_uniform_density_research_prior`：气簧两半、传动杆 link2/link3/link4，均为左右各一。它们是均匀密度网格估计，尚非实测真值。',
               '- 旧主大腿、主小腿、轮、输入曲柄保留旧 CAD 值；新全车为当前 CAD 导出值。', '',
               '## 逐件转动惯量：质量接近，参数差异仍明显', '',
               '**下表 J 的数字单位统一为 10⁻⁶ kg·m²，例如 1000 表示 0.001 kg·m²。新网格列为同一新 STL、同一新质量下的均匀密度积分诊断，不写入 URDF。**', '',
               '| Link | 旧 J | 全车 J | 变化 | 新网格估计 J | 全车张量检查 |', '|---|---:|---:|---:|---:|---|']
    for e in data['links']:
        mesh=e['full_owned_mesh_diagnostic'].get('axis_parallel_com_inertia_kg_m2')
        mesh_s='未用非封闭网格积分' if mesh is None else f'{mesh*1e6:.3f}'
        v=e['full_validation'];state='通过基本检查' if v['positive_definite'] and v['triangle_valid'] else '不合格：零主惯量'
        if not v['triangle_valid']:state+=' / 三角不等式不满足'
        report.append(f'| {e["name"]} | {e["old_axis_parallel_com_inertia_kg_m2"]*1e6:.3f} | {e["full_axis_parallel_com_inertia_kg_m2"]*1e6:.3f} | {e["axis_parallel_com_inertia_change_percent"]:+.2f}% | {mesh_s} | {state} |')
    report += ['', '具体解释：', '',
               '- 主大腿绕该侧铰轴的质心惯量保持约 0.001 kg·m²，但完整张量并不相同：新长轴方向 Ixx 为 0.00010361，旧换到同坐标后约 0.000078，增加约 32.8%；另一方向约减少 4.76%，原非零惯量积也有变化。',
               '- 主小腿绕铰轴惯量约 0.004960 → 0.005000 kg·m²，增加约 0.81%；其长轴主惯量和惯量积仍有明显变化。',
               '- 轮的轴向惯量 0.000575 → 0.000400 kg·m²，减少 30.43%；横向惯量各从 0.000296 → 0.000200，减少 32.43%。新轮半径已从旧 60 mm 改为 52.5 mm，故不能把轮惯量差全部归为导出误差。',
               '- 四类传动杆绕铰轴惯量分别约增加 32.66%、25.27%、32.05%、55.52%，而质量只改变不到 1%。新车对应网格的均匀密度估计与旧值很接近：link2/link3 的差基本对应当前质量减少，link4 几乎相同。说明这里主要暴露参数/来源差异，不能据此认定实物惯量真的增加了 25%–56%。',
               '- 全车两个气簧上半件的惯量张量完全为零；两下半件及左右传动杆均有零主惯量。其中 12 件不满足有限三维刚体的基本惯量要求，本轮没有把旧估计值直接覆盖进去。', '',
               '![惯量对比]('+str(OUT/'inertia_comparison.png')+')', '',
               '## 完整张量与左右关系也有差异', '',
               '下表把旧值旋转到新全车同名 link frame 后列出六个分量，顺序为 `(Ixx, Iyy, Izz; Ixy, Ixz, Iyz)`，单位仍为 10⁻⁶ kg·m²。', '',
               '| Link | 旧六分量 | 全车六分量 |', '|---|---|---|']
    for name in ['L_link1','R_link1','L_link2','R_link2','LL_link1','RR_link1']:
        e=rows[name]
        def components(t):
            a=np.asarray(t)*1e6
            return f'({a[0,0]:.3f}, {a[1,1]:.3f}, {a[2,2]:.3f}; {a[0,1]:.3f}, {a[0,2]:.3f}, {a[1,2]:.3f})'
        report.append(f'| {name} | {components(e["old_com_tensor_in_full_link_kg_m2"])} | {components(e["full_com_tensor_kg_m2"])} |')
    report += ['',
               '- 新 L_link2 与 R_link2 的 Ixz 都为 −0.0002 kg·m²；旧值在相同物理局部坐标中分别约 +0.000191 / −0.000191。理想 Z 镜像件的 Ixz 应反号，因此新左右小腿没有保留旧模型的惯量积镜像关系，需要结合 CAD 材料分布核对，不能仅检查三个对角分量。',
               '- LL_link1 与 RR_link1 同重，当前网格均匀密度估计的绕铰轴惯量几乎相同；但全车填入的 Iyy 分别为 0.00012422 / 0.00010000，相差 24.22%。这也是需要核对的左右物性差异。', '',
               '## 主惯量对比：不受坐标旋转影响', '',
               '单位仍为 10⁻⁶ kg·m²，三个数按从小到大排序。零主惯量在旋转坐标后仍为零，因此不是 X/Y 互换造成的假象。', '',
               '| Link | 旧三个主惯量 | 全车三个主惯量 |', '|---|---|---|']
    for e in data['links']:
        old=' / '.join(f'{x*1e6:.3f}' for x in e['old_validation']['principal_moments_kg_m2'])
        new=' / '.join(f'{x*1e6:.3f}' for x in e['full_validation']['principal_moments_kg_m2'])
        report.append(f'| {e["name"]} | {old} | {new} |')
    report += ['', '## 质心与物性来源需要区分', '',
               '气簧上半件 CAD 局部质心 X 为 49.165 mm，而旧底盘采用的均匀密度网格质心约 28.355 mm，差约 20.81 mm；下半件对应约 70.864 与 56.621 mm，差约 14.24 mm。相同网格若包含不同材料/密度分布，实际质心可以偏离均匀密度结果，因此不能把旧网格估计当作唯一正确值。', '',
               '这部分差异来自旧资产选择的估计方法，不是本轮改轴造成的质量或质心变化。本轮所有惯性 XML 元素与修改前完全一致。', '',
               '## 回执与边界', '',
               '- [逐关节轴修改及符号映射]('+str(OUT/'axis_changes.json')+')',
               '- [完整质量、质心、张量、主惯量及网格诊断]('+str(OUT/'inertia_comparison.json')+')',
               '- [原始文件备份]('+str(OUT/'before')+')',
               '- 闭链按本轮要求未处理。本轮验证是静态文件、变换及网格积分，不是动力学或实车验证。', '']
    (OUT/'REVIEW.md').write_text('\n'.join(report))


def plot_axes(after):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    before=ET.parse(OUT/'before'/URDF.name).getroot()
    fig=plt.figure(figsize=(13,6.8))
    for i,(robot,title) in enumerate(((before,'Before: mixed directions; RR_joint1 fixed'),(after,'After: all 18 leg axes outward'))):
        ax=fig.add_subplot(1,2,i+1,projection='3d');f=fk(robot)
        for j in robot.findall('joint'):
            child=j.find('child').get('link')
            if child.startswith('M_'):continue
            parent=j.find('parent').get('link');p=ALIGN@f[child][:3,3];p0=ALIGN@f[parent][:3,3]
            a=ALIGN@f[child][:3,:3]@vector(j.find('axis').get('xyz'));a/=np.linalg.norm(a)
            ax.plot(*np.stack([p0,p]).T,color='#a6b9a5',lw=1)
            color='#c83737' if child[0]=='L' else '#3265b7'
            if j.get('type')=='fixed':color='#d49a24'
            ax.quiver(*p,*a,length=.09,color=color,arrow_length_ratio=.25,linewidth=1.8)
            ax.scatter(*p,color=color,s=12)
        ax.set(xlabel='forward X (m)',ylabel='left Y (m)',zlabel='up Z (m)',title=title)
        ax.set_xlim(-.26,.03);ax.set_ylim(-.32,.32);ax.set_zlim(-.22,.1)
        ax.view_init(elev=17,azim=-50);ax.set_box_aspect([.35,.7,.35])
    fig.suptitle('Leg joint rotation axes at unchanged CAD zero pose',fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n,lw=2) for n,c in
                        [('left leg','#c83737'),('right leg','#3265b7'),('old fixed-axis field','#d49a24')]],
               loc='lower center',ncols=3)
    fig.subplots_adjust(left=.01,right=.98,bottom=.1,top=.86,wspace=.02)
    fig.savefig(OUT/'axes_before_after.png',dpi=180)
    plt.close(fig)


def plot_inertia(entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['L_link1','L_link2','L_link3','LL_link1','LL_link2','LL_link3','LL_link4','LLL_link1','LLL_link2']
    rows={e['name']:e for e in entries};x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(12,5.7))
    for offset,color,key,label in [(-.24,'#4266a3','old_axis_parallel_com_inertia_kg_m2','Chassis V6/V7'),
                                   (0,'#d28b44','full_axis_parallel_com_inertia_kg_m2','Full URDF'),
                                   (.24,'#4d956e',None,'Full mesh, uniform density diagnostic')]:
        vals=[rows[n][key]*1e6 if key else rows[n]['full_owned_mesh_diagnostic'].get('axis_parallel_com_inertia_kg_m2',np.nan)*1e6 for n in names]
        ax.bar(x+offset,vals,width=.23,color=color,label=label)
        for i,v in enumerate(vals):
            if v==0:ax.text(i+offset,1.2,'0',ha='center',color='#b54832',fontweight='bold')
    ax.set_yscale('log');ax.set_ylim(1,10000);ax.set_xticks(x,names,rotation=20,ha='right')
    ax.set_ylabel('Inertia about COM, parallel to hinge axis (10^-6 kg m^2)')
    ax.set_title('Left leg examples; right leg follows the same pattern (no inertial data edited)')
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right');fig.tight_layout()
    fig.savefig(OUT/'inertia_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    main()
