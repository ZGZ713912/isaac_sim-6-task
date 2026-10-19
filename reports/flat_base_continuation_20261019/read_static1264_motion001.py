"""Locate displacement before and after settling using existing fixed traces."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
OUT = HERE / 'static1264_motion_windows001.json'
if OUT.exists():
    raise SystemExit('Preserve the completed static motion readout')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = {}
for cohort in ('deployment_source001','deployment1248_basic001','deployment1264_basic001',
               'rmcs_default1264_envelope001','full_gate1264_basic001'):
    trace_path = RUN / cohort / 'traces.npz'
    evaluation_path = RUN / cohort / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation['status'] != 'evaluated':
        raise SystemExit('Wait for the completed fixed traces')
    values = np.load(trace_path)
    cases = {}
    for name in ('stand_230mm','stand_305mm','stand_430mm'):
        lanes = np.flatnonzero(values['groups'] == name)
        windows = {}
        for label,lower,upper in (('first1s',0,1),('1to3s',1,3),('3to5s',3,5),('5to10s',5,10)):
            selected = (values['active'][:,lanes].astype(bool)
                & (values['episode_ticks'][:,lanes] * .02 > lower)
                & (values['episode_ticks'][:,lanes] * .02 <= upper))
            velocity = values['velocity'][:,lanes][selected]
            observations = values['observations'][:,lanes][selected]
            encoder = .3 * (observations[:,20] - observations[:,21])
            windows[label] = dict(frames=len(velocity),body_vx_mean_m_s=float(velocity[:,0].mean()),
                body_vy_mean_m_s=float(velocity[:,1].mean()),
                encoder_vx_mean_m_s=float(encoder.mean()),
                encoder_body_abs_difference_mean_m_s=float(np.abs(encoder-velocity[:,0]).mean()),
                wheel_encoder_speed_std_m_s=float(encoder.std()),
                root_planar_speed_mean_m_s=float(np.linalg.norm(velocity[:,:2],axis=-1).mean()))
        position = values['position'][:,lanes,:2]
        endpoints = {}
        for seconds in (1,3,5,10):
            displacements = []
            for index,lane in enumerate(lanes):
                ticks = values['episode_ticks'][:,lane]
                valid = np.flatnonzero(values['active'][:,lane] & (ticks*.02 <= seconds))
                at = valid[-1]
                displacements.append((position[at,index]-position[0,index]).tolist())
            endpoints[str(seconds)] = dict(displacement_xy_m=displacements,
                displacement_norm_max_m=float(np.linalg.norm(displacements,axis=-1).max()))
        cases[name] = dict(windows=windows,displacement_from_first_poststep_frame=endpoints,
            unchanged_acceptance_drift_m=evaluation['result']['cases'][name]['stand_drift_max_m'],
            unchanged_acceptance_passed=evaluation['result']['cases'][name]['passed'])
    rows[cohort] = dict(evaluation_sha256=sha(evaluation_path),trace_sha256=sha(trace_path),cases=cases)
    values.close()
result = dict(schema='flat-static1264-displacement-window-readout-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),cohorts=rows,
    scope='Existing traces only. Window means and first-poststep displacement locate drift; they do not change the original whole-episode drift verdict or prove a learning cause.',
    new_native_steps=0,new_optimizer_updates=0,behavior_qualified=False)
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for cohort,row in rows.items():
    print(cohort)
    for name,c in row['cases'].items():
        print(name,'drift',round(c['unchanged_acceptance_drift_m'],5),
              'last5s vx',round(c['windows']['5to10s']['body_vx_mean_m_s'],6),
              'first1smax',round(c['displacement_from_first_poststep_frame']['1']['displacement_norm_max_m'],5))
