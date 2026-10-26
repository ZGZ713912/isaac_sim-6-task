"""Describe existing sensor/physical feedback in failures; no policy edits."""
from pathlib import Path
import json,sys,time
sys.path[:0]=['src','scripts']
import numpy as np
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');rows=[]
selections={'diagnosis035_current_source_control':['real_start_3__clean','real_start_5__pitch_plus2','real_start_6__pitch_plus2'],
 'diagnosis038_quiet_stance050_development':['stand_low__roll_plus2','stand__roll_plus2','stand_high__pitch_minus2','stand_high__combined']}
for name,cases in selections.items():
 p=B/name/'evaluation';report=json.loads((p/'report.json').read_text());t=np.load(p/'trajectory.npz')
 for case in cases:
  n=next(x['lane'] for x in report['lanes'] if x['name']==case);mask=t['alive'][:,n]&(t['time'][:,n]>=6.)
  obs=t['observations'][mask,n];actual=t['omega'][mask,n,:2];measured=obs[:,4:6]*2.
  enc=obs[:,16:20]*10.;common=enc.reshape(-1,2,2).mean(-1)
  rows.append(dict(run=name,name=case,frames=len(obs),physical_body_omegaXY_RMS_rad_s=float(np.sqrt((actual**2).sum(-1).mean())),
   measured_gyroXY_RMS_rad_s=float(np.sqrt((measured**2).sum(-1).mean())),
   measured_leg_common_speed_RMS_rad_s=float(np.sqrt((common**2).mean())),
   measured_leg_common_peak_rad_s=float(np.abs(common).max()),
   body_observed_gravity_z_mean=float(obs[:,9].mean()),physical_gravity_z_mean=float(t['gravity'][mask,n,2].mean()),
   encoder_common_activity_fraction_above05=float((np.abs(common).max(-1)>.5).mean())))
out=dict(schema='quiet-feedback-observed-activity-v1',rows=rows,
 provenance=[dict(path=str(B/n/'evaluation/trajectory.npz'),sha256=digest(B/n/'evaluation/trajectory.npz')) for n in selections],
 interpretation='Descriptive recorded input/physical signal statistics only. No gate selected, no causal noise separation or qualification claimed.',
 new_PPO=0,qualification=None,hardware_started=False,created_unix=time.time());write_json(R/'quiet_sensor_activity001.json',out);print(json.dumps(out,indent=2))
