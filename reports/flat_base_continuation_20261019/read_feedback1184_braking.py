"""Read archived braking actions at50Hz; no simulation, fitting or optimizer work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/flat_base_20261019'
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
output=HERE/'feedback1184_braking_readout001.json'
if output.exists():raise SystemExit('Preserve completed archived braking readout')
cases=('start_stop_1','start_stop_2','start_stop_3')
rows={};checks={};initial_inputs={}
for step,trial in ((1120,'pilot021'),(1184,'pilot022')):
    checkpoint=RUN/trial/f'model_{step:06d}.pt'
    evaluation=RUN/f'rmcs_default{step}_envelope001/evaluation.json'
    trace=evaluation.with_name('traces.npz')
    result=read(evaluation)
    checks[str(step)+'_completed_same_checkpoint']=result['status']=='evaluated' and result['checkpoint_sha256']==sha(checkpoint)
    checks[str(step)+'_all12_braking_episodes_full']=all(result['result']['cases'][n]['full_horizon_episodes']==4 for n in cases)
    checks[str(step)+'_same_native_clock']=result['physics_hz']==1000 and result['policy_hz']==50
    with np.load(trace) as z:
        groups=z['groups'];ids=np.flatnonzero(np.isin(groups,cases))
        # Cache each compressed member exactly once before indexing frames.
        data={k:z[k][:,ids].copy() for k in ('observations','actions','velocity','active','commands','raw_requested_commands','motor_effort')}
    obs,actions,velocity=data['observations'],data['actions'],data['velocity']
    initial_inputs[step]=obs[0].copy()
    actor=mean_from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=True),
        ROOT/'models/wheel_leg_flat_base_20261018/source.onnx').eval()
    with torch.inference_mode():
        inputs=torch.from_numpy(obs.reshape(-1,35))
        predicted=actor(inputs).numpy().reshape(actions.shape)
        command=actor.command(inputs).numpy().reshape(actions.shape)
        correction=actor.feedback_actions(inputs).numpy().reshape(actions.shape)
        gate=actor.tracking_feedback_gate(inputs).numpy().reshape(obs.shape[:2])
        mix=actor.fast_mix(inputs).numpy().reshape(obs.shape[:2])
        features=(inputs[:,:28]-actor.input_center)*actor.input_scale
        raw_low=torch.tanh(actor.feedback(features)).numpy().reshape(actions.shape)
    checks[str(step)+'_CPU_reproduces_archived_actions']=np.allclose(predicted,actions,atol=2e-5,rtol=2e-5)
    checks[str(step)+'_feedback_decomposition']=np.allclose(predicted,command+correction,atol=2e-5,rtol=2e-5)
    case_rows={}
    for name in cases:
        ns=np.flatnonzero(groups[ids]==name);stops=[]
        for start,sign in ((400,1),(1200,-1)):
            end=start+400
            zero=data['active'][start:end,ns].astype(bool)
            vx_input=obs[start:end,ns,0]
            # These are policy inputs, applied diagnostics and requested vx,
            # all at the same tick. Heading-hold yaw is allowed to vary.
            checks[f'{step}_{name}_{start}_zero_command_aligned']=bool(zero.all() and np.all(vx_input==0)
                and np.all(data['commands'][start:end,ns,0]==0)
                and np.all(data['raw_requested_commands'][start:end,ns,0]==0))
            checks[f'{step}_{name}_{start}_no_wheel_action_clipping']=bool(np.all(abs(actions[start:end,ns,4:])<=9))
            checks[f'{step}_{name}_{start}_low_branch_selected']=bool(np.all(mix[start:end,ns]==0))
            speed=np.linalg.norm(velocity[start:end,ns,:2],axis=-1)
            # End-state ages are .02 through8s. The existing metric uses age>1s.
            age=(np.arange(400)+1)*.02
            settles=[]
            for lane in range(len(ns)):
                suffix=np.maximum.accumulate(speed[::-1,lane])[::-1]
                eligible=np.flatnonzero(suffix<=.03)
                settles.append(float(age[eligible[0]]) if len(eligible) else None)
            first2=raw_low[start:start+100,ns]
            states=[]
            for seconds in (0.,.2,.5,1.,1.5,2.,3.):
                t=start+round(seconds/.02)
                states.append(dict(stop_age_end_s=round(seconds+.02,2),
                    root_oriented_vx_mean_m_s=float((sign*velocity[t,ns,0]).mean()),
                    planar_speed_max_m_s=float(np.linalg.norm(velocity[t,ns,:2],axis=-1).max()),
                    encoder_oriented_vx_mean_m_s=float((sign*.3*(obs[t,ns,20]-obs[t,ns,21])).mean()),
                    frozen_base_and_command_forward_target_m_s=float((sign*.3*(command[t,ns,4]-command[t,ns,5])).mean()),
                    feedback_forward_target_m_s=float((sign*.3*(correction[t,ns,4]-correction[t,ns,5])).mean()),
                    sampled_end_state_wheel_effort_oriented_mean_nm=float((sign*.5*(data['motor_effort'][t,ns,2]-data['motor_effort'][t,ns,5])).mean())))
            stops.append(dict(incoming_sign=sign,stop_start_policy_frame=start,
                actual_per_lane_settles_by_end_of_segment_s=settles,
                speed_peak_after1s_m_s=float(speed[age>1].max()),
                speed_peak_after3s_m_s=float(speed[age>3].max()),
                mean_speed_after3s_m_s=float(speed[age>3].mean()),
                first2s_low_MLP_abs_tanh_max_wheels=float(abs(first2[:,:,4:]).max()),
                first2s_low_MLP_abs_tanh_max_legs=float(abs(first2[:,:,:4]).max()),
                first2s_low_MLP_wheel_fraction_at_or_above95percent=float((abs(first2[:,:,4:])>=.95).mean()),
                first2s_low_MLP_leg_fraction_at_or_above95percent=float((abs(first2[:,:,:4])>=.95).mean()),
                first2s_tracking_gate_min=float(gate[start:start+100,ns].min()),samples=states))
        actual_peak=max(s['speed_peak_after1s_m_s'] for s in stops)
        recorded_peak=result['result']['cases'][name]['settled_stop_speed']
        checks[f'{step}_{name}_reconstruct_existing_settled_metric']=bool(np.isclose(actual_peak,recorded_peak,atol=2e-6,rtol=2e-6))
        case_rows[name]=dict(existing_settled_stop_speed_m_s=recorded_peak,stops=stops)
    rows[str(step)]=dict(checkpoint_sha256=sha(checkpoint),evaluation_sha256=sha(evaluation),traces_sha256=sha(trace),
        CPU_native_action_max_abs_error=float(abs(predicted-actions).max()),cases=case_rows)
checks['initial_selected_observations_bit_exact_between_versions']=np.array_equal(initial_inputs[1120],initial_inputs[1184])
report=dict(schema='flat-feedback1184-archived-braking-action-readout-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=all(checks.values()),checks=checks,reader_sha256=sha(Path(__file__)),versions=rows,
    data_source='Existing matched native evaluation traces, selected12 first-life episodes per checkpoint',
    interpretation_scope='Descriptive same-input branch contributions and recorded closed-loop trajectories only. '
        'No intervention, causal proof, new acceptance test, per-physics-tick saturation claim or fitted action target.',
    qualification_eligible=False,behavior_qualified=False,optimizer_updates_performed=0,native_environment_steps=0,
    hardware_motion_started=False)
if not report['passed']:
    print(json.dumps(dict(passed=False,failed_checks=[k for k,v in checks.items() if not v])))
    raise SystemExit('Archived readout did not reproduce the bound native metrics')
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),versions={
    step:{name:dict(speed_peak_after1s_m_s=c['existing_settled_stop_speed_m_s'],
        settles_s=[s['actual_per_lane_settles_by_end_of_segment_s'] for s in c['stops']],
        speed_peak_after3s_m_s=[s['speed_peak_after3s_m_s'] for s in c['stops']],
        low_MLP_max_wheel_tanh_first2s=[s['first2s_low_MLP_abs_tanh_max_wheels'] for s in c['stops']],
        low_MLP_max_leg_tanh_first2s=[s['first2s_low_MLP_abs_tanh_max_legs'] for s in c['stops']])
        for name,c in r['cases'].items()} for step,r in rows.items()})))
