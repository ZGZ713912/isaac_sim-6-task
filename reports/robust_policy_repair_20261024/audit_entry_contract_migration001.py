"""Actual checkpoint preservation, strict rejection and default actor replay."""
import copy,hashlib,json,sys,tempfile
from pathlib import Path
sys.path[:0]=['src','scripts']
import numpy as np
import torch
from export_robust_getup import load_mean

torch.set_num_threads(1);base=Path('runs/robust_policy_repair_20261024');oldpath=base/'pilot007_entry_support0/model_final.pt';newpath=base/'pilot007_entry_contract001/model_migrated.pt';source=base/'entry_optimizer_support0_source001/model_migrated.pt'
a=torch.load(oldpath,map_location='cpu',weights_only=False);b=torch.load(newpath,map_location='cpu',weights_only=False)
def equal(x,y):
 if type(x) is not type(y):return False
 if isinstance(x,torch.Tensor):return torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
 return x==y
checks={k+'_exact':equal(a[k],b[k]) for k in a if k!='infos'}
bi=copy.deepcopy(b['infos']);bi.pop('entry_recovery_learning');bi.pop('entry_checkpoint_contract_migration');bi['source_sha256'].pop('src/wheeled_algo/entry_checkpoint.py');checks['all_original_infos_RNG_clock_and_settings_exact']=equal(a['infos'],bi)
try:load_mean(oldpath)
except ValueError as e:checks['old_unproven_checkpoint_still_rejected']='Frozen entry weights differ' in str(e)
else:checks['old_unproven_checkpoint_still_rejected']=False
new,info=load_mean(newpath);manual,_=load_mean(source);manual.load_state_dict({k.removeprefix('mlp.mean.'):v for k,v in a['actor_state_dict'].items() if k.startswith('mlp.mean.')},strict=True)
frames=np.load(base/'diagnosis018_entry_support0_native/evaluation/trajectory.npz')['observations'].reshape(-1,35);different=0
with torch.inference_mode():
 for obs in torch.tensor(frames).split(2048):different+=int((manual(obs)!=new(obs)).any(-1).sum())
checks['81600_native_frames_function_exact']=different==0 and len(frames)==81600
negative={}
with tempfile.TemporaryDirectory(prefix='entry-contract-validation-') as folder:
 for kind in ('head','frozen_std','source_hash','clock','missing_proof','contract_source','cycle'):
  bad=copy.deepcopy(b)
  if kind=='head':bad['actor_state_dict']['mlp.mean.entry_recovery.4.bias'][0]+=.01
  elif kind=='frozen_std':bad['actor_state_dict']['mlp.log_std'][0]+=.01
  elif kind=='source_hash':bad['infos']['entry_recovery_learning']['source_sha256']='0'*64
  elif kind=='clock':bad['infos']['entry_recovery_learning']['new_optimizer_updates']=9
  elif kind=='missing_proof':bad['infos'].pop('entry_recovery_learning')
  elif kind=='contract_source':bad['infos']['source_sha256']['src/wheeled_algo/entry_checkpoint.py']='0'*64
  elif kind=='cycle':bad['infos']['entry_recovery_learning']['source_checkpoint']=str(newpath.resolve())
  path=Path(folder)/(kind+'.pt');torch.save(bad,path)
  try:load_mean(path)
  except ValueError as e:negative[kind]=str(e)
  else:raise AssertionError(kind+' should be rejected')
checks['all7_negative_loads_rejected']=len(negative)==7
out=dict(schema='entry-contract-native-checkpoint-preflight-v1',checks=checks,negative_rejection=negative,old_checkpoint=str(oldpath),old_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),new_checkpoint=str(newpath),new_sha256=hashlib.sha256(newpath.read_bytes()).hexdigest(),frames=len(frames),different_frames=different,new_PPO=0,actual_experiment_consumed=2595,qualification=None,hardware_started=False)
Path('reports/robust_policy_repair_20261024/entry_contract_migration_preflight001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert all(checks.values()),checks
