"""Neutral V4 representation of best9a; no new behavior or optimizer update."""
from pathlib import Path
import sys,json,time
sys.path[:0]=['src','scripts']
import torch,numpy as np
from export_robust_getup import load_mean,digest
from train_robust_getup import write_json
from wheeled_algo.periodic_entry_actor import exact_state_equal
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024');torch.set_num_threads(1)
source=B/'motion_damping080002/model_probe.pt';cp=B/'quiet_stance000003/model_probe.pt';old,_=load_mean(source);new,info=load_mean(cp)
a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in (source,cp)];frames=different=0
traces=[]
with torch.inference_mode():
 for name in ('diagnosis035_current_source_control','diagnosis038_quiet_stance050_development'):
  p=B/name/'evaluation/trajectory.npz';x=np.load(p)['observations'].reshape(-1,35);traces.append(dict(path=str(p),sha256=digest(p)))
  for values in np.array_split(x,(len(x)+511)//512):
   obs=torch.from_numpy(values.copy());different+=int((old(obs)!=new(obs)).any(-1).sum());frames+=len(obs)
checks=dict(full_function_bit_exact=different==0,retained_authority_zero=info['settings']['quiet_stance_retained_authority']==0.,
 all_payload_weights_critic_Adam_RNG_exact=exact_state_equal({k:v for k,v in a.items() if k!='infos'},{k:v for k,v in b.items() if k!='infos'}),
 learning_clock_1501=info['completed_updates']==1501)
out=dict(schema='neutral-stance-root-preflight-v1',checks=checks,source_checkpoint=str(source),source_sha256=digest(source),checkpoint=str(cp),checkpoint_sha256=digest(cp),
 frames=frames,different_frames=different,traces=traces,new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'neutral_stance_root_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks
