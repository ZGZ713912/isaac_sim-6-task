"""CPU function/distribution checks before paired native sampled exposures."""
from pathlib import Path
import ast
import json
import sys
import time
sys.path[:0]=['src','scripts',str(Path(__file__).parent)]
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from entry_exploration_probe001 import alter_entry_std
from wheeled_algo.gap_entry_handoff import handoff_entry_gate
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean

torch.set_num_threads(1)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
source=B/'motion_damping080002/model_probe.pt';mean,info=load_mean(source)
state=torch.load(source,map_location='cpu',weights_only=False);s=info['settings']
model=GapRecoveryNoiseMean(mean,s['quiet_exploration_fraction'],s.get('quiet_wheel_exploration_fraction'),s['gap_recovery_leg_exploration_fraction'])
with torch.no_grad():model.log_std.copy_(state['actor_state_dict']['mlp.log_std'])
checks=dict(control_exact=True,means_exact=True,wheels_exact=True,outside_entry_exact=True,full_entry_std_quarter=True)
frames=entry0=entry1=0
with torch.inference_mode():
    for path in [B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz',B/'diagnosis015_development21_baseline/evaluation/trajectory.npz']:
        obs=np.load(path)['observations'].reshape(-1,35)
        for values in np.array_split(obs,(len(obs)+511)//512):
            x=torch.from_numpy(values.copy());a=model(x);b=alter_entry_std(a,x,.25);gate=handoff_entry_gate(x).ravel()
            zero,one=gate==0,gate==1
            checks['control_exact']&=torch.equal(a,alter_entry_std(a,x,1.))
            checks['means_exact']&=torch.equal(a[:,0],b[:,0])
            checks['wheels_exact']&=torch.equal(a[:,1,4:],b[:,1,4:])
            checks['outside_entry_exact']&=torch.equal(a[zero],b[zero])
            if one.any():checks['full_entry_std_quarter']&=bool(torch.allclose(b[one,1,:4].exp(),.25*a[one,1,:4].exp(),atol=1e-8,rtol=1e-6))
            frames+=len(x);entry0+=int(zero.sum());entry1+=int(one.sum())
checks={k:bool(v) for k,v in checks.items()}
# Execute the actual wrapper module up to its final main call, in an isolated
# CPU process, so import/patch errors cannot silently become native retries.
wrapper=R/'evaluate_entry_exploration_probe001.py'
sys.argv=[str(wrapper),'--sampled-policy','--entry-leg-std-ratio','.25']
tree=ast.parse(wrapper.read_text());tree.body=tree.body[:-1]
namespace={'__file__':str(wrapper.resolve()),'__name__':'sample_probe_import_fixture'}
exec(compile(tree,str(wrapper),'exec'),namespace);checks['wrapper_import_smoke']=True
out=dict(schema='entry-leg-exploration-probe-preflight-v1',checks=checks,source=str(source),source_sha256=digest(source),
    frames=frames,entry_zero_frames=entry0,full_entry_frames=entry1,
    module_sha256=digest(R/'entry_exploration_probe001.py'),wrapper_sha256=digest(wrapper),
    ratio=.25,source_entry_scale=.2,candidate_entry_scale=.05,new_PPO=0,
    scope='Mean and wheel/off-entry distribution exact; private common Gaussian draws will be used for native pair. No mean-policy qualification.',created_unix=time.time())
write_json(R/'entry_exploration_probe_preflight001.json',out);print(json.dumps(out,indent=2));assert all(checks.values())
