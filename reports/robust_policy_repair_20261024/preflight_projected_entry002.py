"""Check neutral projection, replay the sealed proposal and test save metadata."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path[:0]=['src','scripts',str(Path(__file__).parent)]
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest,write_json
from common_phase_projector001 import CommonPhaseProjector,VERSION
from entry_common_guard001 import PhaseGuard

torch.set_num_threads(4)
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
source=B/'function_root_entry002/model_initial.pt'
proposal=B/'pilot012_common_entry002/model_stopped.pt'
known=B/'native_common_projected001/model_probe.pt'
mean,_=load_mean(source);original=deepcopy(mean.state_dict())
projector=CommonPhaseProjector(mean,Path.cwd());neutral=projector.project(mean)
checks=dict(neutral_exact=all(torch.equal(v,mean.state_dict()[k]) for k,v in original.items()))
native,_=load_mean(proposal)
mean.load_state_dict(native.state_dict());result=projector.project(mean)
guard=PhaseGuard(load_mean(source)[0],R/'entry_phase_path_dataset001.npz');guard.mean=mean
metrics,phase=guard.measure();checks.update({'phase_'+k:v for k,v in phase.items()})
expected,_=load_mean(known)
checks['replayed_projection_weights_exact']=all(torch.equal(v,expected.state_dict()[k]) for k,v in mean.state_dict().items())
checks['bounded_recorded_step']=result['recorded_step_target_peak_rad']<=.005
checks['critic_Adam_source_unchanged']=digest(proposal)==json.loads((R/'common_entry_endpoint001.json').read_text())['checkpoint_sha256']
wrapper=R/'run_projected_entry_pilot001.py'
sys.argv=[str(wrapper),'--entry-recovery-only','--recovery-only','--output','/tmp/projected-entry-import-only','--resume',str(source)]
tree=ast.parse(wrapper.read_text());tree.body=tree.body[:-1]
namespace={'__file__':str(wrapper.resolve()),'__name__':'projected_import_fixture'}
exec(compile(tree,str(wrapper),'exec'),namespace);checks['actual_wrapper_imported']=True
captured={}
def native_save(self,path,infos=None):captured.update(infos)
namespace['native_save']=native_save;namespace['context']=SimpleNamespace(projected_rows=[dict(learning_clock=1502,checks=phase)])
namespace['save'](SimpleNamespace(),'/tmp/not-written',dict(completed_updates=1502))
record=captured['entry_projected_optimizer']
checks['metadata_explicit_projected_method']=record['schema']==VERSION and record['completed_projected_PPO_updates']==1
checks['metadata_source_bound']=record['source_sha256']==digest(source) and record['source_clock']==1501
checks['metadata_projector_bound']=record['projector_sha256']==digest(R/'common_phase_projector001.py')
out=dict(schema='projected-common-native-proposal-replay-preflight-v1',checks={k:bool(v) for k,v in checks.items()},
    source=str(source),source_sha256=digest(source),proposal=str(proposal),proposal_sha256=digest(proposal),
    known_native_reviewed_projection=str(known),known_sha256=digest(known),
    neutral=neutral,replayed_projection=result,phase_metrics=metrics,
    wrapper_sha256=digest(wrapper),projector_sha256=digest(R/'common_phase_projector001.py'),
    new_PPO=0,optimizer_steps=0,decision='Engineering preflight only, no new PPO admitted',created_unix=time.time())
write_json(R/'projected_entry_preflight002.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks
