"""Require exact native replay from a saved policy without diagnostic overrides."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
import numpy as np
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
q=json.loads((R/'diagnosis_queue036.json').read_text());assert q['status']=='completed'
p=json.loads((R/'diagnosis_plan036.json').read_text());assert all(digest(n)==h for n,h in p['source_sha256'].items())
a=B/'diagnosis035_stance_authority050_fixed';b=Path(p['jobs'][0]['output'])
x,y=[np.load(t/'evaluation/trajectory.npz') for t in (a,b)]
arrays={k:bool(np.array_equal(x[k],y[k],equal_nan=True)) for k in x.files};pa,pb=[json.loads((t/'progress.json').read_text()) for t in (a,b)]
prea,preb=[np.load(t/'evaluation_precontrol_state.npz') for t in (a,b)]
initial={k:bool(np.array_equal(prea[k],preb[k],equal_nan=True)) for k in prea.files}
checks=dict(all56_native_arrays_bit_exact=len(arrays)==56 and all(arrays.values()) and set(x.files)==set(y.files),
 precontrol_exact=all(initial.values()),evaluation_overrides_empty=pb['evaluation_overrides']=={},
 explicit_checkpoint_function=pb['actor_settings']['quiet_stance_authority_version']=='quiet-stance-authority-v1' and pb['actor_settings']['quiet_stance_retained_authority']==.5,
 original_diagnostic_override_preserved=set(pa['evaluation_overrides'])=={'quiet_gyro_stance_authority_probe'},frozen_sources_exact=True,
 checkpoint_exact=pb['checkpoint_sha256']==digest(p['jobs'][0]['checkpoint']))
out=dict(schema='quiet-stance-native-implementation-parity-v1',checks=checks,arrays=arrays,initial=initial,
 checkpoint=p['jobs'][0]['checkpoint'],checkpoint_sha256=pb['checkpoint_sha256'],trace_sha256={str(t):digest(t/'evaluation/trajectory.npz') for t in (a,b)},
 frozen_inputs=len(p['source_sha256']),decision='eligible_for_expanded_zero_PPO_review_only' if all(checks.values()) else 'reject_implementation',
 new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'quiet_stance_native_implementation_parity001.json',out);print(json.dumps(out,indent=2));assert all(checks.values()),checks
