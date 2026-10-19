"""Read deterministic trajectories to separate start-up error from steady error."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flat_base_20261019'
rows={};initial={}
for run in ('envelope_source002','feedback832_envelope001','feedback896_envelope001'):
 path=RUN/run/'traces.npz';z=np.load(path);obs=z['observations'];active=z['active'];groups=z['groups'];times=z['episode_ticks']*.02
 demand=np.maximum(np.abs(obs[:,:,0])/.75,np.abs(obs[:,:,2])/1.5)
 turn=obs[:,:,2]*(.4373/2)
 expected=np.stack((obs[:,:,0]-turn,-obs[:,:,0]-turn),-1)/.6
 wheel_error=np.max(np.abs(obs[:,:,20:22]-expected),-1)*.6
 gate=np.maximum(np.clip(1-demand,0,1)**2,np.clip((wheel_error-.015)/.085,0,1))
 metrics={};initial[run]={}
 for name in ('forward_3','backward_1','backward_2','backward_3','rotate_4_reverse'):
  lanes=groups==name;initial[run][name]=obs[0,lanes].copy()
  windows={}
  vxerr=np.abs(z['velocity'][:,:,0]-z['commands'][:,:,0]);yawerr=np.abs(z['omega'][:,:,2]-z['commands'][:,:,1])
  for label,lo,hi in (('0_1s',0.,1.),('1_3s',1.,3.),('3_5s',3.,5.),('5_9s',5.,9.)):
   mask=active&lanes[None,:]&(times>lo)&(times<=hi)
   count=int(mask.sum())
   if not count:continue
   zero=mask&(gate==0.)
   windows[label]=dict(sample_frames=count,root_vx_mean_m_s=float(z['velocity'][:,:,0][mask].mean()),
      root_vx_mae_m_s=float(vxerr[mask].mean()),root_yaw_mae_rad_s=float(yawerr[mask].mean()),
      wheel_rim_error_mean_m_s=float(wheel_error[mask].mean()),zero_feedback_gate_fraction=float((gate[mask]==0.).mean()),
      root_vx_error_over005_and_gate_zero_fraction=float(((vxerr[mask]>.05)&(gate[mask]==0.)).mean()),
      zero_gate_sample_frames=int(zero.sum()),root_vx_mae_when_gate_zero_m_s=float(vxerr[zero].mean()) if zero.any() else None)
  metrics[name]=windows
 rows[run]=dict(traces_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics)
checks={name:all(np.array_equal(initial['envelope_source002'][name],initial[run][name]) for run in ('feedback832_envelope001','feedback896_envelope001')) for name in initial['envelope_source002']}
receipt=dict(schema='flat-feedback896-motion-window-audit-v1',recorded_at=datetime.now(timezone.utc).isoformat(),policy_dt_s=.02,initial35D_observation_exact_across_policies=checks,rows=rows,optimizer_updates_performed=0,used_for_optimization=False,behavior_qualified=False,scope='Read-only active first-episode deterministic frames; this grouping is diagnostic and does not replace the fixed acceptance metrics.')
Path(__file__).with_name('feedback896_motion_window_audit001.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(receipt,indent=2))
