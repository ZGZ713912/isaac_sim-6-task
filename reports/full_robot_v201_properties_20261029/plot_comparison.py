"""Plot source data comparisons; plots are diagnostics, not simulation results."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
data = json.loads((OUT / 'comparison.json').read_text())
entries = {e['name']: e for e in data['comparison']}
summary = data['summary']
fig, axes = plt.subplots(2, 2, figsize=(13.2, 9), constrained_layout=True)
ax = axes[0, 0]
parts = [('Base', [summary['old_base_mass_kg'], summary['new_base_mass_kg']], '#5579ac'),
         ('Legs + wheels + springs', [summary['old_leg_mass_kg'], summary['new_leg_mass_kg']], '#59a694'),
         ('Gimbal', [0, summary['new_gimbal_mass_kg']], '#edaf54')]
bottom = np.zeros(2)
for label, values, color in parts:
    ax.bar(['Chassis v7', 'Full v2.0.1'], values, bottom=bottom, label=label, color=color, width=.5)
    bottom += values
for i, value in enumerate(bottom):
    ax.text(i, value + .35, f'{value:.3f} kg', ha='center', weight='bold')
ax.set_ylim(0, 25)
ax.set_ylabel('Mass (kg)')
ax.set_title('Mass increase is concentrated in the upper assembly')
ax.legend(frameon=False, fontsize=9)

ax = axes[0, 1]
names = ['L_link1', 'L_link2', 'L_link3', 'LL_link1', 'LL_link2', 'LL_link3', 'LL_link4', 'LLL_link1', 'LLL_link2']
labels = ['Main upper leg', 'Main lower leg', 'Wheel', 'Rod 1', 'Rod 2', 'Rod 3', 'Rod 4', 'Spring upper', 'Spring lower']
changes = [entries[n]['hinge_parallel_com_inertia_change_percent'] for n in names]
ax.barh(labels, changes, color=['#59a694' if c >= 0 else '#5579ac' for c in changes])
ax.invert_yaxis()
ax.axvline(0, color='0.3', lw=.7)
ax.set_xlim(-4.5, 5.5)
ax.set_xlabel('Change in hinge-parallel COM inertia (%)')
ax.set_title('Leg axial inertias remain close (left side shown)')
for y, value in enumerate(changes):
    ax.text(value + (.1 if value >= 0 else -.1), y, f'{value:+.2f}%',
            va='center', ha='left' if value >= 0 else 'right', fontsize=9)

ax = axes[1, 0]
ax.bar(np.arange(3) - .18, [.178, .126, .25526], .36, color='#5579ac', label='Chassis v7')
ax.bar(np.arange(3) + .18, [.29731, .32743, .53495], .36, color='#edaf54', label='Full v2.0.1 base only')
ax.set_xticks(np.arange(3), ['Roll (X forward)', 'Pitch (Y left)', 'Yaw (Z up)'])
ax.set_ylabel('COM inertia (kg m²)')
ax.set_title('Base inertias, expressed in the same physical axes')
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
full_path = ROOT / 'model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1/urdf/轮腿整车URDFv2.0.1.urdf'
link = ET.parse(full_path).getroot().find("link[@name='base_link']")
path = full_path.parent.parent / link.find('visual/geometry/mesh').get('filename').split('/', 3)[3]
mesh = trimesh.load_mesh(path, process=True)
# Full-root -> X-forward chassis-root: x=-y, y=x, z=z.
v = np.column_stack((-mesh.vertices[:, 1], mesh.vertices[:, 0], mesh.vertices[:, 2]))
rng = np.random.default_rng(20261029)
sample = v[rng.choice(len(v), min(12000, len(v)), replace=False)]
ax.scatter(sample[:, 0] * 1000, sample[:, 2] * 1000, s=.25, alpha=.3, color='#edaf54', label='New detailed base STL')
h = ConvexHull(v[:, [0, 2]])
polygon = v[np.r_[h.vertices, h.vertices[0]]][:, [0, 2]]
ax.plot(polygon[:, 0] * 1000, polygon[:, 1] * 1000, '--', color='#a34b39', lw=1.7,
        label='Single-hull silhouette (illustration)')
for i, rec in enumerate(data['collision_source']['chassis']):
    if rec['name'] != 'base_link':
        continue
    old_mesh = trimesh.load_mesh(rec['path'], process=True)
    old_hull = ConvexHull(old_mesh.vertices[:, [0, 2]])
    old_polygon = old_mesh.vertices[np.r_[old_hull.vertices, old_hull.vertices[0]]][:, [0, 2]]
    ax.plot(old_polygon[:, 0] * 1000, old_polygon[:, 1] * 1000, color='#5579ac', lw=1,
            label='Old base collision pieces' if i == 0 else None)
ax.set_aspect('equal')
ax.set_xlabel('Forward X (mm)')
ax.set_ylabel('Up Z (mm)')
ax.set_title('Base collision source: hulls fill concave gaps')
ax.legend(frameon=False, fontsize=8, loc='lower left')
for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Full v2.0.1 versus chassis v7 — offline model comparison', fontsize=15)
fig.savefig(OUT / 'comparison.png', dpi=170)
print(OUT / 'comparison.png')
