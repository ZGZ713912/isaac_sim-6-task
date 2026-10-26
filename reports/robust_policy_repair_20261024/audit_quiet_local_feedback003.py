"""Read-only actor Jacobians on recorded native standing trajectories."""
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import quiet_gate

torch.set_num_threads(1)
base = Path('runs/robust_policy_repair_20261024')
run = base / 'diagnosis015_development21_baseline'
progress = json.loads((run/'progress.json').read_text())
cp = Path(progress['checkpoint'])
model, _ = load_mean(cp)
model.eval()
z = np.load(run/'evaluation/trajectory.npz')
report = json.loads((run/'evaluation/report.json').read_text())
rows=[]
for lane in report['lanes']:
    if lane['name'] not in ('stand_high__wobble','stand_high__clean','stand_low__roll_plus2','stand_low__clean'):
        continue
    i=lane['lane']
    for t in (.25,.5,1.,2.,3.,4.,6.,7.5):
        k=int(np.abs(z['time'][:,i]-t).argmin())
        obs=torch.tensor(z['observations'][k,i:i+1],requires_grad=True)
        out=model(obs)
        jac=torch.stack([torch.autograd.grad(out[0,j],obs,retain_graph=True)[0][0] for j in range(6)])
        memory=jac[:,22:28].detach().numpy()
        e=np.linalg.eigvals(memory)
        rows.append(dict(name=lane['name'],time=float(z['time'][k,i]),quiet_gate=float(quiet_gate(obs).item()),
            body_vx=float(z['velocity'][k,i,0]),local_position=z['local_position'][k,i].tolist(),
            memory_spectral_radius=float(abs(e).max()),memory_eigenvalues=[[float(v.real),float(v.imag)] for v in e],
            memory_jacobian=memory.tolist(),
            pose_and_velocity_jacobian=jac[:,4:22].detach().tolist(),
            position_feedback_jacobian=jac[:,14:16].detach().tolist(),
            action=out.detach().tolist()[0]))
receipt=dict(schema='recorded-quiet-actor-local-jacobian-v1',checkpoint=str(cp),
    sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),run=str(run),rows=rows,
    limitation='Actor partial derivatives at recorded states only; memory spectral radius is not a closed-loop physical stability proof.',new_PPO=0)
Path('reports/robust_policy_repair_20261024/quiet_local_feedback003.json').write_text(json.dumps(receipt,indent=2)+'\n')
for r in rows:print(r['name'],r['time'],round(r['quiet_gate'],3),round(r['body_vx'],3),round(r['memory_spectral_radius'],4))
