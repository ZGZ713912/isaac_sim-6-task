"""Replay-check a policy-only gate change with an exact zero-strength control."""
from pathlib import Path
import sys,json,time
sys.path[:0]=['src','scripts']
import numpy as np,torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from quiet_gyro_authority_probe001 import policy_authority,probe_gate,NATIVE_GATE
from wheeled_algo import robust_getup,quiet_output_actor
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
cp=B/'motion_damping080002/model_probe.pt';mean,_=load_mean(cp);source=B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
x=np.load(source)['observations'].reshape(-1,35)
native_methods=(robust_getup.RobustGetupActor.forward,quiet_output_actor.QuietOutputGetupActor.forward)
control_different=inactive_different=inactive_count=active_count=changed_count=0
peak=np.zeros(6)
with torch.inference_mode():
 for values in np.array_split(x,(len(x)+511)//512):
  v=torch.from_numpy(values.copy());before=mean(v)
  with policy_authority(0.):zero=mean(v)
  with policy_authority(.5):after=mean(v)
  q=NATIVE_GATE(v);p=probe_gate(v,.5);inactive=(q==p).ravel();active=~inactive
  control_different+=int((before!=zero).any(-1).sum());inactive_different+=int((before[inactive]!=after[inactive]).any(-1).sum())
  inactive_count+=int(inactive.sum());active_count+=int(active.sum());changed_count+=int((before!=after).any(-1).sum())
  peak=np.maximum(peak,(before-after).abs().amax(0).numpy())
checks=dict(zero_strength_full_function_exact=control_different==0,inactive_frames_exact=inactive_different==0,
    active_intervention_nonempty=active_count>0 and changed_count>0,
    methods_restored=native_methods==(robust_getup.RobustGetupActor.forward,quiet_output_actor.QuietOutputGetupActor.forward),
    shared_observer_distribution_gate_unchanged=robust_getup.quiet_gate is NATIVE_GATE)
out=dict(schema='quiet-authority-probe-preflight-v1',checks=checks,checkpoint=str(cp),checkpoint_sha256=digest(cp),
 trace=str(source),trace_sha256=digest(source),frames=len(x),control_different=control_different,
 inactive_count=inactive_count,inactive_different=inactive_different,active_count=active_count,changed_count=changed_count,
 normalized_action_delta_peak=peak.tolist(),probe_sha256=digest(R/'quiet_gyro_authority_probe001.py'),
 wrapper_sha256=digest(R/'evaluate_quiet_authority_probe001.py'),new_PPO=0,qualification=None,created_unix=time.time())
write_json(R/'quiet_authority_probe_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks
