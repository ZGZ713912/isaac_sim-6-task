"""Projection for the existing258 common coefficients, never a runtime controller."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import osqp
from scipy import sparse
import torch
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.getup_symmetry import reflect_observation
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits

VERSION='fixed-basis-common-projected-Adam-v1'

class CommonPhaseProjector:
    def __init__(self,native_mean,root):
        self.root=Path(root)
        R=self.root/'reports/robust_policy_repair_20261024'
        B=self.root/'runs/robust_policy_repair_20261024'
        mean=deepcopy(native_mean).cpu();prefix=mean.entry_recovery[:-1]
        bits=torch.tensor([[float(bool(mask&(1<<j))) for j in range(4)] for mask in range(16)])
        namespace=dict(mean=mean,prefix=prefix,bits=bits,torch=torch,math=math,
            reflect_observation=reflect_observation,periodic_canonical_entry_logits=periodic_canonical_entry_logits)
        nodes=[n for n in ast.parse((R/'fit_entry_phase_path001.py').read_text()).body
               if isinstance(n,ast.FunctionDef) and n.name in ('hidden','design')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'authenticated-common-design','exec'),namespace)
        design=namespace['design']
        def materialize(obs):
            matrices=[];outputs=[]
            with torch.inference_mode():
                for batch in obs.split(512):
                    l,r,old=design(batch)
                    matrices.append(torch.stack((l,r),1).reshape(-1,258).numpy())
                    outputs.append(old.reshape(-1).numpy())
            return np.concatenate(matrices),np.concatenate(outputs)
        data=np.load(R/'entry_phase_path_dataset001.npz')
        observations=torch.from_numpy(data['observations'].copy())
        desired=data['desired_common'].reshape(-1);phase=data['phase'][data['pose_ids']]
        gate=handoff_entry_gate(observations).numpy()
        far=((np.abs(phase)>=.75)&(gate>.99)).reshape(-1)
        near=((np.abs(phase)<1e-6)&(gate>.99)).reshape(-1)
        X,original=materialize(observations)
        recorded=np.load(B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz')['observations'][::5].reshape(-1,35)
        recorded=torch.from_numpy(recorded.copy());recorded=recorded[handoff_entry_gate(recorded).ravel()>0]
        Xa,_=materialize(recorded)
        self.recorded=recorded
        self.G=X.T@X/len(X)+Xa.T@Xa/len(Xa)+1e-4*np.eye(258)
        pool=np.flatnonzero(far|near);self.PX=X[pool].copy()
        is_far=far[pool];sign=np.sign(desired[pool]);self.PX[is_far]*=sign[is_far,None]
        self.lo=np.full(len(pool),-np.inf);self.hi=np.full(len(pool),np.inf)
        self.lo[is_far]=.25-sign[is_far]*original[pool[is_far]]
        self.lo[~is_far]=-.06-original[pool[~is_far]];self.hi[~is_far]=.06-original[pool[~is_far]]
        self.weight=mean.entry_recovery[-1].weight.detach().clone()
        self.bias=mean.entry_recovery[-1].bias.detach().clone()
        self.previous=np.zeros(258)
        self.initial_output=self.outputs(native_mean)
        self.previous_output=self.initial_output.copy()

    def outputs(self,mean):
        device=next(mean.parameters()).device
        with torch.inference_mode():
            return torch.cat([mean(x.to(device))[:,:4].cpu()*.25 for x in self.recorded.split(512)]).numpy()

    def vector(self,mean):
        weight=mean.entry_recovery[-1].weight.detach().cpu()-self.weight
        bias=mean.entry_recovery[-1].bias.detach().cpu()-self.bias
        return np.concatenate([torch.cat((weight[i:i+2].mean(0),bias[i:i+2].mean()[None])).double().numpy() for i in (0,2)])

    def project(self,mean):
        started=time.monotonic();proposal=self.vector(mean)
        if not np.any(proposal):
            return dict(status='neutral',rounds=[],projection_solved=True,solver_seconds=0.,
                recorded_step_target_peak_rad=0.,retained_initial_change_rms_rad=0.,proposal_coefficient_l2=0.)
        rhs=self.G@proposal;selected=set(np.linspace(0,len(self.lo)-1,min(512,len(self.lo)),dtype=int).tolist())
        rounds=[];vector=self.previous.copy();status='not_solved';violation=np.full(len(self.lo),np.inf)
        for iteration in range(8):
            ids=np.array(sorted(selected));solver=osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.triu(self.G)),q=-rhs,A=sparse.csc_matrix(self.PX[ids]),
                l=self.lo[ids],u=self.hi[ids],verbose=False,eps_abs=1e-7,eps_rel=1e-7,
                max_iter=20000,time_limit=45.,polishing=True)
            solver.warm_start(x=vector);result=solver.solve();status=result.info.status
            row=dict(iteration=iteration+1,status=status,selected_constraints=len(ids),solver_seconds=result.info.run_time);rounds.append(row)
            if status not in ('solved','solved inaccurate'):break
            vector=result.x;value=self.PX@vector;violation=np.maximum(self.lo-value,value-self.hi)
            row.update(violated_constraints=int((violation>2e-6).sum()),max_constraint_violation=float(violation.max()))
            if violation.max()<=2e-6:break
            worst=np.argsort(violation)[::-1]
            added=[int(i) for i in worst if violation[i]>2e-6 and int(i) not in selected][:256]
            if not added:break
            selected.update(added)
        solved=status in ('solved','solved inaccurate') and violation.max()<=2e-6 and np.isfinite(vector).all()
        if not solved:raise RuntimeError('Bounded phase projection did not solve: '+str(rounds))
        with torch.no_grad():
            layer=mean.entry_recovery[-1]
            for first,side in ((0,0),(2,1)):
                v=torch.tensor(vector[side*129:(side+1)*129],dtype=torch.float32)
                for row in (first,first+1):
                    layer.weight[row].copy_((self.weight[row]+v[:-1]).to(layer.weight.device))
                    layer.bias[row].copy_((self.bias[row]+v[-1]).to(layer.bias.device))
        output=self.outputs(mean)
        difference=wrap(torch.from_numpy(output-self.previous_output)).numpy()
        initial=wrap(torch.from_numpy(output-self.initial_output)).numpy()
        self.previous=vector;self.previous_output=output
        return dict(status=status,rounds=rounds,projection_solved=True,solver_seconds=time.monotonic()-started,
            recorded_step_target_peak_rad=float(np.abs(difference).max()),
            retained_initial_change_rms_rad=float(np.sqrt(np.mean(initial**2))),
            proposal_coefficient_l2=float(np.linalg.norm(proposal)),
            projected_coefficient_l2=float(np.linalg.norm(vector)),max_constraint_violation=float(violation.max()))
