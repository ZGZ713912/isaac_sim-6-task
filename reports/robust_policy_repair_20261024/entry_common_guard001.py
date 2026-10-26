"""Experimental gradient restriction and offline phase guard; no controller.

Only the two shared final common rows can receive a nonzero gradient. The
existing optimizer and likelihood remain native; runtime actor code is intact.
"""
import numpy as np
import torch

from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import periodic_canonical_entry_logits


def common_rows(gradient):
    result=torch.zeros_like(gradient)
    for start in (0,2):
        value=.5*(gradient[start]+gradient[start+1])
        result[start]=value
        result[start+1]=value
    return result


def attach_common_gradient_mask(network):
    expected={'0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'}
    params=dict(network.named_parameters())
    if set(params)!=expected or params['4.weight'].shape!=(6,128):
        raise ValueError('Common restriction requires the existing35x128x128x6 entry NN')
    if any(not p.requires_grad for p in params.values()):
        raise ValueError('Attach after the native entry-only optimizer layout is configured')
    hooks=[]
    for name,param in params.items():
        hooks.append(param.register_hook(common_rows if name.startswith('4.') else lambda g:torch.zeros_like(g)))
    return hooks


class PhaseGuard:
    def __init__(self,mean,dataset):
        self.mean=mean
        device=next(mean.parameters()).device
        z=np.load(dataset)
        self.observations=torch.from_numpy(z['observations'].copy()).to(device)
        self.desired=torch.from_numpy(np.sign(z['desired_common']).copy()).to(device)
        phase=z['phase'][z['pose_ids']]
        with torch.inference_mode():
            gate=torch.cat([handoff_entry_gate(x) for x in self.observations.split(512)])
        self.far=torch.from_numpy(np.abs(phase)>=.75).to(device)&(gate>.99)
        self.near=torch.from_numpy(np.abs(phase)<1e-6).to(device)&(gate>.99)
        self.fixed={k:v.detach().clone() for k,v in mean.entry_recovery.state_dict().items()
                    if not k.startswith('4.')}
        self.last={k:v.detach().clone() for k,v in mean.entry_recovery[-1].state_dict().items()}
        self.reference_knee=self._raw_knee()

    def _raw_knee(self):
        result=[]
        with torch.inference_mode():
            for x in self.observations.split(512):
                frame=torch.cat((x[:,:14],torch.zeros_like(x[:,14:16]),x[:,16:]),-1)
                raw=periodic_canonical_entry_logits(self.mean.entry_recovery,frame,self.mean.feature_scale)
                result.append(raw[:,[1,3]]-raw[:,[0,2]])
        return torch.cat(result)

    def measure(self):
        outputs=[]
        with torch.inference_mode():
            for x in self.observations.split(512):
                y=self.mean(x)
                lead=wrap(y[:,:4]*.25-x[:,10:14]).reshape(-1,2,2).mean(-1)
                outputs.append(lead)
        lead=torch.cat(outputs)
        nn=self.mean.entry_recovery.state_dict()
        last=self.mean.entry_recovery[-1].state_dict()
        fixed=all(torch.equal(v,nn[k]) for k,v in self.fixed.items())
        wheels=all(torch.equal(v[4:],last[k][4:]) for k,v in self.last.items())
        metrics=dict(far_signed_lead_min_rad=float((lead*self.desired)[self.far].min()),
            stopped_target_peak_rad=float(lead[self.near].abs().max()),
            knee_logit_delta_peak=float((self._raw_knee()-self.reference_knee).abs().max()))
        checks=dict(frozen_feature_basis_exact=fixed,wheel_rows_exact=wheels,
            far_margin=metrics['far_signed_lead_min_rad']>=.014,
            stopped_margin=metrics['stopped_target_peak_rad']<=.006,
            knee_roundoff=metrics['knee_logit_delta_peak']<=2e-5,
            finite=all(np.isfinite(v) for v in metrics.values()))
        return metrics,checks
