"""Verify frozen branches, native Adam state and finite entry-only PPO output."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
plan=json.loads(args.plan.read_text());source=Path(plan['source_checkpoint']);run=Path(plan['training_output']);final=run/'model_final.pt'
a=torch.load(source,map_location='cpu',weights_only=False);b=torch.load(final,map_location='cpu',weights_only=False)
progress=json.loads((run/'progress.json').read_text());restore=json.loads((Path(plan['supervisor_output'])/'initial_restore.json').read_text())
old=a['actor_state_dict'];new=b['actor_state_dict'];prefix='mlp.mean.entry_recovery.'
changed=[k for k,v in old.items() if not torch.equal(v,new[k])]
entry=[k for k in old if k.startswith(prefix)]
def equal(x,y):
    if type(x) is not type(y):return False
    if isinstance(x,torch.Tensor):return torch.equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
    return x==y
def finite(x):
    if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    return True
optkey=next(k for k in a if 'optim' in k)
opt0=a[optkey];opt1=b[optkey];g0=opt0['param_groups'][0]['params'];g1=opt1['param_groups'][0]['params'];entryids=g0[-6:]
checks=dict(restore_all10=all(restore['checks'].values()) and len(restore['checks'])==10,
    completed8=progress['status']=='completed' and progress['new_optimizer_updates']==8 and progress['new_optimizer_attempts']==8,
    lineage1501to1509=a['infos']['completed_updates']==1501 and b['infos']['completed_updates']==1509,
    only6_entry_tensors_change=set(changed)==set(entry) and len(entry)==6,
    old_actor_and_std_exact=all(torch.equal(v,new[k]) for k,v in old.items() if not k.startswith(prefix)),
    old_actor_Adam_exact=all(equal(opt0['state'].get(i),opt1['state'].get(i)) for i in g0 if i not in entryids),
    entry_Adam_created=all(i not in opt0['state'] and i in opt1['state'] for i in entryids),
    actor_optimizer_IDs_exact=g0==g1,
    critic_changed=not equal(a['critic_state_dict'],b['critic_state_dict']),
    all_actor_critic_Adam_finite=all(finite(b[k]) for k in ('actor_state_dict','critic_state_dict',optkey)),
    native_freeze_check=progress.get('frozen_non_entry_and_std_unchanged') is True,
    trainable21894=progress['trainable_actor_parameters']==21894)
receipt=dict(schema='entry-only-PPO-endpoint-audit-v1',plan=str(args.plan),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint=str(final),checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),checks=checks,
    changed_actor_tensors=changed,entry_tensor_delta_l2={k:float((new[k]-old[k]).norm()) for k in entry},
    new_PPO_updates=8,qualification=None,hardware_started=False)
args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert all(checks.values()),checks
