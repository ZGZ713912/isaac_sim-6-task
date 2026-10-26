"""Fit an existing motion NN last layer to mild causal target damping offline."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
import time

sys.path[:0] = ['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.robust_getup import quiet_gate, tilt_gate, fast_forward_reference_gate
from wheeled_algo.paired_recovery import wrap

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
OUTPUT = R/'motion_damping_standardized_feasibility001.json'
assert not OUTPUT.exists()
source = B/'entry_phase_fit_periodic002/model_probe.pt'
mean,_ = load_mean(source)
alpha,ridge,equilibrium_weight = .8,1e-5,10.
datasets = [('source_fixed','diagnosis012_final_arithmetic_fixed/evaluation',True),
    ('source_nominal','diagnosis012_final_arithmetic_nominal/robust',True),
    ('source_RMCS','diagnosis012_final_arithmetic_rmcs/evaluation',True),
    ('new_phase_fixed','diagnosis024_entry_phase_fit_fixed/evaluation',False)]
sets = []
started = time.time()
with torch.inference_mode():
    for label,name,training in datasets:
        path = B/name/'trajectory.npz'
        z = np.load(path)
        observations = torch.tensor(z['observations'][::5].reshape(-1,35))
        active = torch.tensor((z['alive'][::5]&~z['passive'][::5]).reshape(-1).copy())
        entry = handoff_entry_gate(observations)[:,0]
        recovery = tilt_gate(observations,mean.stance_aware_recovery,mean.recovery_shape_aware)[:,0]
        quiet = quiet_gate(observations)[:,0]
        reference = fast_forward_reference_gate(observations,mean.fast_forward_reference_range,
                                                 mean.fast_forward_reference_yaw_fade)[:,0]
        arc = ((observations[:,:1].abs()-.05)/.2).clamp(0.,1.)*((observations[:,2:3].abs()-.05)/.2).clamp(0.,1.)
        moving = torch.maximum(observations[:,:1].abs()/.1,observations[:,2:3].abs()/.2).clamp(0.,1.)
        residual = (arc+mean.motion_residual_fraction*(1.-arc)*moving)[:,0]
        select = active&(entry==0.)&(recovery==0.)&(quiet==0.)&(reference==0.)&(residual>.2)
        if not select.any():
            print(label,'has no eligible moving frames; excluded from the fit',flush=True)
            continue
        obs = observations[select]
        frame = torch.cat((obs[:,:14],torch.zeros_like(obs[:,14:16]),obs[:,16:]),-1)
        features = torch.tanh(frame*mean.feature_scale)
        hidden = mean.arc[:-1](features)
        raw = mean.arc[-1](hidden)
        parent = mean.parent(frame)
        gain = residual[select,None]*mean.arc_bound
        old = parent+gain*raw.tanh()
        # Work in the unshaped normal-motion region. All labels are causal
        # functions of the existing previous action and current observations.
        step = wrap((old[:,:4]-obs[:,22:26])*.25)
        desired = old[:,:4]-(1.-alpha)*step/.25
        inverse = (desired-parent[:,:4])/gain[:,:4]
        normal = ((old[:,:4].abs()<2.95).all(-1)
            &(wrap(old[:,:4]*.25-obs[:,10:14]).abs()<.05).all(-1)
            &(step.abs()<.04).all(-1)&(inverse.abs()<.98).all(-1))
        obs,hidden,raw,parent,gain,old,step,inverse = [x[normal] for x in
            (obs,hidden,raw,parent,gain,old,step,inverse)]
        desired_raw = torch.atanh(inverse)
        X = torch.cat((hidden,torch.ones_like(hidden[:,:1])),-1).double()
        y = (desired_raw-raw[:,:4]).double()
        sample = torch.linspace(0,len(obs)-1,min(64,len(obs))).long()
        full = mean(obs[sample])
        assert float((full[:,:4]-old[sample,:4]).abs().max())<3e-6
        sets.append(dict(name=label,training=training,trace=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),obs=obs,X=X,y=y,
            raw=raw,parent=parent,gain=gain,old=old,step=step,
            candidates=int(select.sum()),normal_frames=len(obs)))
        print(label,'normal frames',len(obs),'candidate frames',int(select.sum()),flush=True)
    train = [s for s in sets if s['training']]
    X = torch.cat([s['X'] for s in train])
    y = torch.cat([s['y'] for s in train])
    step = torch.cat([s['step'] for s in train])
    weights = torch.where(step.abs().amax(-1)<.0001,equilibrium_weight,1.).double()
    center = X[:,:-1].mean(0)
    scale = X[:,:-1].std(0).clamp_min(.001)
    normalized = torch.cat(((X[:,:-1]-center)/scale,torch.ones_like(X[:,:1])),-1)
    gram = normalized.T@(weights[:,None]*normalized)/weights.sum()+ridge*torch.eye(X.shape[1],dtype=torch.float64)
    rhs = normalized.T@(weights[:,None]*y)/weights.sum()
    standardized_vector = torch.linalg.solve(gram,rhs)
    coefficients = standardized_vector[:-1]/scale[:,None]
    vector = torch.cat((coefficients,(standardized_vector[-1]-center@coefficients)[None]),0)
    rows = []
    for s in sets:
        new_raw = s['raw'][:,:4]+(s['X']@vector).float()
        predicted = s['parent'][:,:4]+s['gain'][:,:4]*new_raw.tanh()
        new_step = wrap((predicted-s['obs'][:,22:26])*.25)
        expected = alpha*s['step']
        error = wrap(new_step-expected)
        delta = wrap((predicted-s['old'][:,:4])*.25)
        before = float(s['step'].square().mean().sqrt())
        after = float(new_step.square().mean().sqrt())
        eq = s['step'].abs().amax(-1)<.0001
        rows.append(dict(name=s['name'],training=s['training'],trace=s['trace'],sha256=s['sha256'],
            candidates=s['candidates'],normal_frames=s['normal_frames'],
            normal_fraction=s['normal_frames']/s['candidates'],
            target_step_rms_before_rad=before,target_step_rms_after_rad=after,
            target_step_rms_ratio=after/before,teacher_error_rms_rad=float(error.square().mean().sqrt()),
            teacher_error_peak_rad=float(error.abs().max()),weight_edit_target_peak_delta_rad=float(delta.abs().max()),
            equilibrium_frames=int(eq.sum()),equilibrium_target_peak_delta_rad=float(delta[eq].abs().max()) if eq.any() else 0.))
checks = dict(finite=bool(torch.isfinite(vector).all()),
    teacher_error_rms=all(r['teacher_error_rms_rad']<=.0004 for r in rows),
    teacher_error_peak=all(r['teacher_error_peak_rad']<=.004 for r in rows),
    step_reduction_at_least_10pct=all(r['target_step_rms_ratio']<=.9 for r in rows),
    target_edit_bounded=all(r['weight_edit_target_peak_delta_rad']<=.02 for r in rows),
    equilibrium_preserved=all(r['equilibrium_target_peak_delta_rad']<=.0005 for r in rows))
out = dict(schema='motion-NN-target-damping-feasibility-v1',source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    parameters=dict(alpha=alpha,ridge=ridge,equilibrium_weight=equilibrium_weight,feature_whitening=True,feature_scale_floor=.001),
    method='One 65-feature by four-output existing arc NN final-layer solve. Offline target interpolation is supervision only; no runtime filter or new state is introduced.',
    independent_parameters=260,metrics=rows,checks=checks,edit_matrix=vector.tolist(),feature_center=center.tolist(),feature_scale=scale.tolist(),
    scope='Recorded-state function fit with the new phase trace excluded from the solve. It does not prove native support improvement or behavior generalization.',
    decision='eligible_for_checkpoint_preflight' if all(checks.values()) else 'reject_without_native_or_PPO',
    new_PPO=0,gradient_optimizer_updates=0,actual_experiment_consumed=2603,
    qualification=None,hardware_started=False,elapsed_seconds=time.time()-started,created_unix=time.time())
OUTPUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='edit_matrix'},indent=2),flush=True)
