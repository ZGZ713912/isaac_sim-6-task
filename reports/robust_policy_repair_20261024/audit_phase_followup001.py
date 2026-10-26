"""Explain unchanged strict followup failures from complete native traces."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
models = [('corrected_source','diagnosis012_final_arithmetic_fixed'),
          ('phase_constrained','diagnosis024_entry_phase_fit_fixed')]
results = []
for label,directory in models:
    run = B/directory
    path = run/'evaluation/trajectory.npz'
    z = np.load(path)
    report = json.loads((path.parent/'report.json').read_text())
    writer = SummaryWriter(str(path.parent/'metrics'),filename_suffix='.followup-break-diagnosis')
    rows = []
    for lane in report['lanes']:
        if not lane['recovery'] or not lane['recovered'] or lane['behavior_case'].startswith('real_start'):
            continue
        i = lane['lane']
        t = z['time'][:,i]
        first = int(np.flatnonzero(z['recovered'][:,i])[0])
        valid = z['alive'][:,i]&~z['passive'][:,i]&(t>=t[first])
        conditions = dict(tilt=z['gravity'][:,i,2]<-math.cos(math.radians(15)),
            body_angular_speed=np.linalg.norm(z['omega'][:,i,:2],axis=-1)<.7,
            height=z['height'][:,i]>.19,both_wheel_force_contact=z['wheel_contact'][:,i].all(-1),
            leg_speed=np.abs(z['motor_velocity'][:,i][:,[0,1,3,4]]).max(-1)<3.,
            forward_tracking=np.abs(z['velocity'][:,i,0]-z['commands'][:,i,0])<.25,
            yaw_tracking=np.abs(z['omega'][:,i,2]-z['commands'][:,i,1])<.4)
        follow = valid.copy()
        for condition in conditions.values():
            follow &= condition
        longest = current = 0
        for value in follow:
            current = current+1 if value else 0
            longest = max(current,longest)
        settled = valid&(t>=max(float(t[first])+.5,4.))
        contact_break = settled & ~conditions['both_wheel_force_contact']
        row = dict(name=lane['name'],actual_family=lane['post_settle_family'],
            recovered_at_s=float(t[first]),survived=lane['survived'],
            followup_success=lane['followup_success'],longest_follow_s=longest*.005,
            post4s_valid_frames=int(settled.sum()),
            post4s_failure_frames={k:int((settled&~v).sum()) for k,v in conditions.items()},
            force_contact_break_wheel_clearance_peak_m=float(z['wheel_clearance_peak'][contact_break,i].max()) if contact_break.any() else None,
            force_contact_break_wheel_air_run_peak_s=float(z['wheel_air_run_peak'][contact_break,i].max()) if contact_break.any() else None)
        for key,value in row.items():
            if isinstance(value,(int,float)):
                writer.add_scalar('FollowupDiagnosis/'+row['name']+'/'+key,value,1501)
        for key,value in row['post4s_failure_frames'].items():
            writer.add_scalar('FollowupDiagnosis/'+row['name']+'/failure_frames/'+key,value,1501)
        rows.append(row)
    writer.close()
    results.append(dict(model=label,trace=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows))
out = dict(schema='strict-native-followup-break-audit-v1',models=results,
    conclusion='The phase edit increases completed recovery, but strict uninterrupted two-second followup fails. Most otherwise stable recovered lanes have brief force-contact breaks; the new right_symmetric combined lane also has real pitch-rate and leg-speed breaks. Neither the contact nor followup threshold is changed.',
    scope='Recorded paired trajectories and condition decomposition; no unique causal claim about solver/contact/controller.',
    new_PPO=0,actual_experiment_consumed=2603,qualification=None,hardware_started=False,created_unix=time.time())
(R/'entry_phase_followup_audit001.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
