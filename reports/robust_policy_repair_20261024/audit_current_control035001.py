"""Verify a fresh current-source control against its archived identical mean."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
import numpy as np
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
a=B/'diagnosis025_motion_damping080_fixed';b=B/'diagnosis035_current_source_control'
assert json.loads((b/'progress.json').read_text())['status']=='completed'
x,y=[np.load(p/'evaluation/trajectory.npz') for p in (a,b)]
comparison={k:bool(np.array_equal(x[k],y[k],equal_nan=True)) for k in x.files}
pa,pb=[json.loads((p/'progress.json').read_text()) for p in (a,b)]
ca,cb=[json.loads((p/'task_contract.json').read_text())['settings'] for p in (a,b)]
delta={k:dict(before=ca.get(k,'MISSING'),after=cb.get(k,'MISSING')) for k in ca.keys()|cb.keys() if ca.get(k)!=cb.get(k)}
checks=dict(all_native_arrays_exact=all(comparison.values()) and set(x.files)==set(y.files),same_checkpoint=pa['checkpoint_sha256']==pb['checkpoint_sha256'],
 same_actor_settings=pa['actor_settings']==pb['actor_settings'],expected_only_policy_task_markers=set(delta)=={'gap_entry_encoder_version','motion_leg_damping_version','motion_leg_damping_alpha'})
o=dict(schema='current-source-native-control-replay-v1',checks=checks,baseline=str(a),current_control=str(b),
 comparison=comparison,task_metadata_delta=delta,trace_sha256={str(p):digest(p/'evaluation/trajectory.npz') for p in (a,b)},
 interpretation='Tests actual closed-loop replay under current source; task policy markers were not normalized or removed in the artifacts.',
 new_PPO=0,actual_experiment_consumed=2630,qualification=None,created_unix=time.time())
write_json(R/'current_control035_replay001.json',o);print(json.dumps(o,indent=2));assert all(checks.values()),checks
