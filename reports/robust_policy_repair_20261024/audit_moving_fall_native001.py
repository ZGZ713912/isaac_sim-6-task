"""Read the recorded moving-fall failures without changing impact gates."""
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
import numpy as np
import torch
from train_robust_getup import digest,write_json
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import tilt_gate

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
paths=dict(best=B/'diagnosis026_motion_damping080_nominal_pair/robust',
    entry_control=B/'diagnosis020_entry_support_nominal_pair/reference',
    entry_support=B/'diagnosis020_entry_support_nominal_pair/robust')
runs={}
for role,path in paths.items():
    report=json.loads((path/'report.json').read_text());trace=path/'trajectory.npz'
    assert digest(trace)==report['trajectory_sha256']
    d=np.load(trace);rows=[]
    for lane in report['lanes']:
        i=lane['lane']
        if not lane.get('push',False):
            continue
        active=d['alive'][:,i]&~d['passive'][:,i]
        fallen=active&d['in_motion_fall'][:,i]
        recovered=active&d['recovered_after_push'][:,i]
        first=int(np.flatnonzero(fallen)[0]) if fallen.any() else None
        end=int(np.flatnonzero(active)[-1])
        # Gates use the recorded pre-action observation, not simulator truth.
        x=torch.from_numpy(d['observations'][:,i].copy())
        entry=handoff_entry_gate(x).ravel().numpy()
        actual_fallen=tilt_gate(x,True,True).ravel().numpy()
        if first is not None:
            segment=np.arange(first,end+1)
            early=segment[d['time'][segment,i]<=d['time'][first,i]+.1]
        else:
            early=np.zeros(0,dtype=np.int64)
        rows.append(dict(lane=i,name=lane['name'],survived=lane['survived'],termination_causes=lane['termination_causes'],
            first_fall_s=float(d['time'][first,i]) if first is not None else None,
            first_recovered_after_push_s=float(d['time'][np.flatnonzero(recovered)[0],i]) if recovered.any() else None,
            observed_until_s=float(d['time'][end,i]),
            seconds_available_after_fall=float(d['time'][end,i]-d['time'][first,i]) if first is not None else None,
            entry_gate_first100ms_peak=float(entry[early].max()) if len(early) else None,
            fallen_gate_first100ms_mean=float(actual_fallen[early].mean()) if len(early) else None,
            impact_peak_n=float(d['impact'][active,i].max()),impulse_peak_ns=float(d['impulse'][active,i].max()),
            leg_speed_peak_rad_s=float(d['leg_speed_peak_rad_s'][active,i].max()) if 'leg_speed_peak_rad_s' in d else None,
            body_tilt_peak_deg=float(np.rad2deg(np.arccos(np.clip(-d['gravity'][active,i,2],-1,1))).max())))
    runs[role]=dict(trace=str(trace),sha256=digest(trace),checkpoint_sha256=report.get('checkpoint_sha256'),rows=rows)
write_json(R/'moving_fall_native_diagnosis001.json',dict(schema='native-moving-fall-window-diagnosis-v1',runs=runs,
    scope='Recorded real PhysX trajectories, same nominal case design. Different policies change the state before the push, so this is not a paired sole-cause attribution.',
    gates_unchanged=True,new_PPO=0,qualification=None,hardware_started=False,created_unix=time.time()))
print(json.dumps(runs,indent=2))
