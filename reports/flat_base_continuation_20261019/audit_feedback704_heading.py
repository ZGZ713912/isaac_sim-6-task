"""Read recorded straight-line exits and same-input wheel correction components."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from wheeled_algo.command_residual import mean_from_checkpoint

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/flat_base_20261019'
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
models={n:mean_from_checkpoint(torch.load(RUN/p,map_location='cpu',weights_only=True),source).eval()
        for n,p in ((640,'pilot013/model_000640.pt'),(704,'pilot014/model_000704.pt'))}
traces={n:np.load(RUN/f'feedback{n}_envelope001/traces.npz') for n in models}
report=dict(schema='flat-feedback704-straight-heading-audit-v1',optimizer_updates_performed=0,
 same_input_only_is_not_a_closed_loop_intervention=True,cases={})
for name in ('forward_3','backward_3','curve_low_reverse'):
    rows={}
    for n,d in traces.items():
        ids=np.flatnonzero(d['groups']==name)
        alive=d['active'][:,ids]
        times=d['episode_ticks'][:,ids]*.02
        obs=d['observations'][:,ids]
        terminal=[]
        for j,i in enumerate(ids):
            k=np.flatnonzero(alive[:,j])[-1]
            terminal.append(dict(time_s=float(times[k,j]),position_m=d['position'][k,i].tolist(),
                integrated_gyro_z_rad=float((d['omega'][:k+1,i,2]*.02).sum())))
        mask=alive&(times>=3)&(times<9)
        x=torch.from_numpy(obs[mask])
        values={}
        with torch.inference_mode():
            base=models[n].command.base(x)
            command=models[n].command.residual_actions(x)
            feedback=models[n].feedback_actions(x)
            for label,v in (('source',base),('command',command),('feedback',feedback)):
                values[label]=dict(mean_policy_action=v.mean(0).tolist(),
                    kinematic_wheel_target_forward_m_s=float((.3*(v[:,4]-v[:,5])).mean()),
                    kinematic_wheel_target_yaw_rad_s=float((-.6/.4373*(v[:,4]+v[:,5])).mean()))
            values['other_checkpoint_feedback_at_same_inputs']=dict(
                mean_policy_action=models[704 if n==640 else 640].feedback_actions(x).mean(0).tolist())
        rows[str(n)]=dict(terminal_samples=terminal,samples_3_to_9=len(x),
            gyro_z_mean_rad_s=float(d['omega'][:,ids,2][mask].mean()),
            components_3_to_9=values)
    a,b=[traces[n] for n in (640,704)]
    initial_equal=np.array_equal(a['observations'][0,a['groups']==name],b['observations'][0,b['groups']==name])
    report['cases'][name]=dict(initial_observation_bit_exact=initial_equal,cohorts=rows)
report['trace_sha256']={str(n):sha(RUN/f'feedback{n}_envelope001/traces.npz') for n in models}
Path(__file__).with_name('feedback704_straight_heading_audit001.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:dict(initial_equal=v['initial_observation_bit_exact'],cohorts={n:dict(terminal_times=[t['time_s'] for t in x['terminal_samples']],terminal_y=[t['position_m'][1] for t in x['terminal_samples']],gyro=x['gyro_z_mean_rad_s'],feedback_kinematic_yaw_target=x['components_3_to_9']['feedback']['kinematic_wheel_target_yaw_rad_s']) for n,x in v['cohorts'].items()}) for k,v in report['cases'].items()}))
