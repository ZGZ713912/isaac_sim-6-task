"""Reproduce coverage of CAD vertices by the generated collision envelopes."""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
bundle = ROOT / "model/wheel_leg_v2.0.2/isaac"
spec = json.loads((bundle / "model_spec.json").read_text())
rows = []
for body in spec["bodies"]:
    if body["collisions"][0]["type"] != "mesh":
        continue
    vertices = trimesh.load(bundle / body["mesh"],process=False,force="mesh").vertices
    contained = np.zeros(len(vertices),dtype=bool)
    maximum_vertices = 0
    for col in body["collisions"]:
        hull = trimesh.load(bundle / col["file"],process=False,force="mesh")
        maximum_vertices = max(maximum_vertices,len(hull.vertices))
        equations = ConvexHull(hull.vertices).equations
        contained |= np.all(vertices@equations[:,:3].T+equations[:,3] <= 1e-7,axis=1)
    rows.append(dict(body=body["name"],cad_vertices=len(vertices),outside_count=int((~contained).sum()),
        maximum_hull_vertices=maximum_vertices,collision_count=len(body["collisions"])))
assert not any(row["outside_count"] for row in rows)
print(json.dumps(rows,indent=2))
