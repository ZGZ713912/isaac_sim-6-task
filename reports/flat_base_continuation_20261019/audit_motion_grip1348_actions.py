"""Decompose saved actor outputs on authenticated parent replay observations."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_tasks.chassis.flat_base_continuation import digest,encode
torch.set_num_threads(1)
trial=ROOT/'runs/flat_base_20261019'
candidate=Path(__file__).resolve().parent/'motion_grip_candidate001'
out=candidate/'same_input1348_maximum001.json'
if out.exists(): raise SystemExit('Preserve existing action decomposition')
entry=json.loads((candidate/'native_1348.json').read_text())
if not entry['passed']: raise SystemExit('Require authenticated native final state')
child_path=trial/'pilot030_motion_grip1348/model_001348.pt'
if digest(child_path)!=entry['checkpoint_sha256']: raise SystemExit('Final checkpoint changed')
child_state=torch.load(child_path,map_location='cpu',weights_only=True)
parent_path=trial/'pilot029_static_grip1338/model_001338.pt'
parent_state=torch.load(parent_path,map_location='cpu',weights_only=True)
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
parent=mean_from_checkpoint(parent_state,source).eval()
child=mean_from_checkpoint(child_state,source).eval()
trace_path=trial/'ground025_static1338_maximum_envelope001/traces.npz'
rows={}
with np.load(trace_path) as trace,torch.inference_mode():
 for name in ['forward_05','backward_05','forward_1','backward_1','forward_2','backward_2','forward_3','backward_3']:
  mask=trace['active'].astype(bool)&(trace['groups'][None,:]==name)&(trace['episode_ticks']*.02>=4.)
  obs=torch.from_numpy(trace['observations'][mask].copy())
  if not len(obs): raise ValueError('Missing active parent observations: '+name)
  before,after=parent(obs),child(obs)
  portions={}
  for label,model in [('parent',parent),('child',child)]:
   parts={'original':model.command.base(obs),'command':model.command.residual_actions(obs),'feedback':model.feedback_actions(obs),'wheel_authority':model.authority_actions(obs)}
   rebuilt=parts['original']+parts['command']+parts['feedback']+parts['wheel_authority']
   assert torch.equal(rebuilt,model(obs))
   portions[label]={key:val[:,4:].mean(0).tolist() for key,val in parts.items()}
  rows[name]=dict(frames=len(obs),components_mean_raw_wheel=portions,
    raw_left_right_wheel_before=before[:,4:].mean(0).tolist(),raw_left_right_wheel_after=after[:,4:].mean(0).tolist(),
    mean_wheel_action_change=(after-before)[:,4:].mean(0).tolist(),max_abs_leg_action_change=float((after-before)[:,:4].abs().max()))
report=dict(schema='flat-motion-grip-same-input-wheel-decomposition-v1',
 parent_checkpoint_sha256=digest(parent_path),child_checkpoint_sha256=digest(child_path),trace_sha256=digest(trace_path),
 successful_updates=child_state['infos']['successful_updates_total'],motion_actor_updates=child_state['infos']['motion_grip_actor_updates'],
 cases=rows,scope='Counterfactual outputs on the saved1338 parent maximum-height replay inputs after4s. Frames include surviving active trajectories only; this is not child closed-loop motion, slip or behavior acceptance.',
 new_optimizer_updates_by_reader=0,qualified=False)
out.write_bytes(encode(report))
print(json.dumps(dict(checkpoint=report['successful_updates'],motion_actor_updates=report['motion_actor_updates'],backward3=rows['backward_3'])))
