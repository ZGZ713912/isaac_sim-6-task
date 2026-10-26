"""Preserve per-case gains and losses, including right censoring diagnostics."""
from pathlib import Path
import json,hashlib
import numpy as np
base=Path('runs/robust_policy_repair_20261024');names=['diagnosis012_final_arithmetic_rmcs','diagnosis015_bias015_rmcs'];runs=[]
for n in names:
 p=base/n/'evaluation';r=json.loads((p/'report.json').read_text());t=p/'trajectory.npz';assert hashlib.sha256(t.read_bytes()).hexdigest()==r['trajectory_sha256'];runs.append((r,np.load(t)))
rows=[]
for a,b in zip(runs[0][0]['lanes'],runs[1][0]['lanes'],strict=True):
 assert a['name']==b['name']
 if all(a[k]==b[k] for k in ['survived','recovered','followup_success']):continue
 row=dict(name=a['name'],actual_family=a['post_settle_family'])
 for label,(report,z),l in zip(['before','after'],runs,[a,b],strict=True):
  i=l['lane'];alive=z['alive'][:,i]&~z['passive'][:,i];stable=z['stable'][:,i]&alive;tail=alive&(z['time'][:,i]>=6.);best=cur=0
  for v in stable:cur=cur+1 if v else 0;best=max(best,cur)
  rec=np.flatnonzero(z['recovered'][:,i]);row[label]=dict(survived=l['survived'],recovered=l['recovered'],followup=l['followup_success'],termination=l['termination_causes'],longest_stable_s=best*.005,first_recovered_s=float(z['time'][rec[0],i]) if len(rec) else None,tail_stable_fraction=float(stable[tail].mean()) if tail.any() else None,tail_height_m=float(z['height'][tail,i].mean()) if tail.any() else None,tail_wheel_clearance_m=z['wheel_clearance'][tail,i].mean(0).tolist() if tail.any() else None)
 rows.append(row)
out=dict(schema='RMCS-bias015-per-case-delta-v1',source_paths=names,rows=rows,scope='Fixed8s complete trajectories; tail descriptors do not establish future12s outcomes',new_PPO=0,qualification=None,hardware_started=False);Path('reports/robust_policy_repair_20261024/bias015_rmcs_regression_audit001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
