"""Separate the first-second displacement from later drift, without optimization."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flat_base_20261019'
rows={}
for run in ('baseline_native002','feedback832_basic001','feedback896_basic001'):
 path=RUN/run/'traces.npz';z=np.load(path);times=z['episode_ticks']*.02
 displacement=np.linalg.norm(z['position'][:,:,:2]-z['position'][0:1,:,:2],axis=-1)
 metrics={}
 for name in ('stand_230mm','stand_305mm','stand_430mm'):
  lanes=z['groups']==name;windows={}
  for label,lo,hi in (('0_1s',0.,1.),('1_3s',1.,3.),('3_10s',3.,10.)):
   mask=z['active']&lanes[None,:]&(times>lo)&(times<=hi);count=int(mask.sum())
   if count:
    windows[label]=dict(sample_frames=count,root_vx_mae_m_s=float(np.abs(z['velocity'][:,:,0][mask]).mean()),
      root_vx_mean_m_s=float(z['velocity'][:,:,0][mask].mean()),
      displacement_from_first_recorded_frame_max_m=float(displacement[mask].max()))
  metrics[name]=windows
 rows[run]=dict(traces_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics)
receipt=dict(schema='flat-static896-birth-and-settled-trace-audit-v1',recorded_at=datetime.now(timezone.utc).isoformat(),rows=rows,
 scope='Read-only active deterministic frames relative to first recorded post-action position; fixed acceptance drift remains authoritative.',
 optimizer_updates_performed=0,used_for_optimization=False,behavior_qualified=False)
Path(__file__).with_name('static832_896_window_audit001.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({run:{name:{window:{k:round(v,6) if isinstance(v,float) else v for k,v in value.items()} for window,value in windows.items()} for name,windows in row['metrics'].items()} for run,row in rows.items()},indent=2))
