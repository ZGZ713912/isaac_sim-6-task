"""Offline loader probe for the unadopted identity fork; no ROS/motor code."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import subprocess,sys,tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from separated_feedback_v3_candidate import SeparatedFeedbackActor,KIND

torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
checkpoint=ROOT/'runs/flat_base_20261019/pilot016/model_000832.pt'
parent=torch.load(checkpoint,map_location='cpu',weights_only=True)
mean=mean_from_checkpoint(parent,source)
actor=SeparatedFeedbackActor(mean).eval()
plan=json.loads((ROOT/'contracts/flat_base_finetune_v12.json').read_text())
yaml_path=Path('/home/noir/Documents/workspace/RMCS/rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-rl.yaml')
settings=yaml.safe_load(yaml_path.read_text())['rl_controller']['ros__parameters']
old=ROOT/'models/wheel_leg_flat_tracking640_20261019/verification.json'
old_proof=json.loads(old.read_text())
if any(sha(Path(k))!=v for k,v in old_proof['source_loader_sha256'].items()):
 raise ValueError('Existing compiled loader source no longer matches current RMCS')
binary=old.parent/'rmcs_policy_check'
graph=actor.export_graph(source)
layout=json.dumps(dict(observation=settings['observation_terms'],action=settings['action_terms']),separators=(',',':'),sort_keys=True)
metadata={v.key:v.value for v in graph.metadata_props}
metadata.update(leg_winding_version=plan['leg_target_winding'],leg_target_model_spec_sha256=plan['model_spec_sha256'],rmcs_model_spec_sha256=plan['model_spec_sha256'],rmcs_policy_hz='50',rmcs_history_length='1',rmcs_obs_frame_size='35',rmcs_obs_clip='100',rmcs_obs_layout=json.dumps(settings['observation_terms'],separators=(',',':')),rmcs_actions_layout=json.dumps(settings['action_terms'],separators=(',',':')),rmcs_layout_hash=hashlib.sha256(layout.encode()).hexdigest(),source_checkpoint_sha256=sha(checkpoint),actor_adaptation=KIND,architecture_probe='unadopted_identity_fork; no new native PPO or deployment qualification')
onnx.helper.set_model_props(graph,metadata);onnx.checker.check_model(graph,full_check=True)
trace_path=ROOT/'runs/flat_base_20261019/feedback832_basic001/traces.npz'
values=np.load(trace_path)['observations'][::10].reshape(-1,35)
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
session=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
with torch.no_grad():expected=actor(torch.from_numpy(values)).numpy();old_expected=mean(torch.from_numpy(values)).numpy()
actual=session.run(None,{session.get_inputs()[0].name:values})[0]
fixture='\n'.join(' '.join(format(float(v),'.9g') for v in row) for row in values)
with tempfile.TemporaryDirectory(prefix='flat-separated-interface-') as temp:
 path=Path(temp)/'identity_fork.onnx';onnx.save(graph,str(path));model_sha=sha(path)
 command=[str(binary),str(path),'35',plan['model_spec_sha256'],plan['leg_target_winding']]
 native=subprocess.run(command,input=fixture,text=True,capture_output=True,check=True)
 cpp=np.fromstring(native.stdout,sep=' ').reshape(-1,6)
 negative={}
 for name,index,value in [('wrong_obs_size',2,'34'),('wrong_model_spec',3,'0'*64),('wrong_winding',4,'independent_nearest_v1')]:
  bad=command.copy();bad[index]=value;r=subprocess.run(bad,input='',text=True,capture_output=True)
  negative[name]=dict(rejected=r.returncode!=0,error=r.stderr.strip())
checks=dict(identity_fork_mean_exact_on_recorded_inputs=np.array_equal(expected,old_expected),current_cpp_loader_source_matches=True,pytorch_python_ORT_within_existing_tolerance=bool(np.isclose(actual,expected,atol=2e-5,rtol=2e-5).all()),cpp_python_ORT_within_existing_tolerance=bool(np.isclose(cpp,actual,atol=2e-5,rtol=2e-5).all()),all_three_bad_contracts_rejected=all(v['rejected'] for v in negative.values()))
receipt=dict(schema='flat-separated-v3-unadopted-identity-cpp-interface-v1',recorded_at=datetime.now(timezone.utc).isoformat(),checks=checks,passed=all(checks.values()),candidate_sha256=sha(Path(__file__).with_name('separated_feedback_v3_candidate.py')),parent_checkpoint_sha256=sha(checkpoint),temporary_model_sha256=model_sha,cpp_binary_sha256=sha(binary),current_cpp_source_sha256=old_proof['source_loader_sha256'],source_rmcs_yaml_sha256=sha(yaml_path),verification_samples=len(values),pytorch_python_ORT_max_abs_error=float(np.abs(actual-expected).max()),cpp_python_ORT_max_abs_error=float(np.abs(cpp-actual).max()),negative_contracts=negative,model_file_retained=False,checkpoint_files_written=0,adopted=False,native_optimizer_updates=0,native_environment_steps=0,behavior_qualified=False,hardware_motion_started=False)
Path(__file__).with_name('separated_identity_cpp_candidate001.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(receipt,indent=2));assert receipt['passed']
