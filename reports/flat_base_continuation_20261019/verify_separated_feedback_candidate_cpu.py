"""Synthetic gradient/identity checks only; no optimizer or physical steps."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.onnx_actor_import import recover_actor
from separated_feedback_candidate import SeparatedFeedbackCandidate

torch.set_num_threads(1)
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
path=ROOT/'runs/flat_base_20261019/pilot015/model_000768.pt'
parent=mean_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True),source)
candidate=SeparatedFeedbackCandidate(parent)
teacher,_=recover_actor(source)
frames=torch.zeros(11,35);frames[:,3]=1.525;frames[:,9]=-1.;frames[:,28]=1.
frames[:3,3]=torch.tensor([1.15,1.525,2.15])
frames[3:,0]=torch.tensor([.5,-.5,0.,0.,3.,-3.,0.,0.])
frames[3:,2]=torch.tensor([0.,0.,1.,-1.,0.,0.,4.,-4.])
generator=torch.Generator().manual_seed(191029)
random=torch.randn(8192,35,generator=generator)
random[:,0]*=3;random[:,2]*=4;random[:,3]=1.525+random[:,3]*.2
with torch.no_grad():
 before=candidate(frames).clone()
 checks=dict(initial_synthetic_mean_exact=torch.equal(before,parent(frames)),
     initial_random_mean_exact=torch.equal(candidate(random),parent(random)),
     parent_state_retained_exact=all(torch.equal(v,candidate.state_dict()[n]) for n,v in parent.state_dict().items()),
     four_new_tensors_exact_clones=all(torch.equal(v,candidate.fast_feedback.state_dict()[n]) for n,v in parent.feedback.state_dict().items()),
     low_and_fast_ownership_exact=candidate.fast_mix(frames[:7]).count_nonzero().item()==0 and (candidate.fast_mix(frames[7:])==1).all().item())
scales=torch.tensor([.025]*4+[.04]*2)
loss=.02*.5*((candidate(frames[7:])-teacher(frames[7:]).detach())/scales).square().mean()
loss.backward()
checks['high_teacher_has_zero_low_branch_gradient']=all(p.grad is not None and p.grad.count_nonzero().item()==0 for p in candidate.feedback.parameters())
checks['high_teacher_has_nonzero_finite_fast_gradient']=all(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.count_nonzero().item()>0 for p in candidate.fast_feedback.parameters())
checks['original_command_and_std_unmodified']=all(p.grad is None for p in candidate.command.parameters())
with torch.no_grad():
 for p in candidate.fast_feedback.parameters():p.add_(p.grad,alpha=-3e-4)
 after=candidate(frames)
 checks['virtual_fast_step_static_and_low_outputs_exact']=torch.equal(before[:7],after[:7])
 checks['virtual_fast_step_changes_high_outputs']=not torch.equal(before[7:],after[7:])
 checks['virtual_step_stays_within_original_feedback_bounds']=bool((candidate.feedback_actions(random).abs()<=candidate.output_scale+1e-7).all())
try:candidate.export_graph(source)
except ValueError:checks['unadopted_prototype_cannot_export']=True
else:checks['unadopted_prototype_cannot_export']=False
r=dict(schema='flat-separated-feedback-prototype-cpu-v1',recorded_at=datetime.now(timezone.utc).isoformat(),checks=checks,passed=all(checks.values()),parent_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prototype_sha256=hashlib.sha256(Path(__file__).with_name('separated_feedback_candidate.py').read_bytes()).hexdigest(),synthetic_frames=8203,trainable_feedback_parameters=sum(p.numel() for p in candidate.parameters() if p.requires_grad),adopted=False,optimizer_updates_performed=0,native_environment_steps=0,checkpoint_files_written=0,behavior_qualified=False,limitations='Direct gradient separation only. Native Adam remapping, global gradient-clip interaction, ONNX export and all closed-loop behavior remain unverified.')
Path(__file__).with_name('separated_feedback_candidate_cpu001.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));assert r['passed']
