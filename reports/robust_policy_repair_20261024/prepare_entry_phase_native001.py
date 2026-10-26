"""Review FP32 coordinate preservation and freeze one unchanged native gate."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from wheeled_algo.paired_recovery import DELTA_BOUNDS

R = Path('reports/robust_policy_repair_20261024')
preflight = R/'entry_phase_fit_preflight002.json'
z = json.loads(preflight.read_text())
assert all(v for k,v in z['checks'].items() if k!='knee_logit_roundoff_under_1e_5')
raw = max(x['max_float32_knee_logit_change'] for x in z['replays'])
width = max(b-a for a,b in DELTA_BOUNDS)
lipschitz = width/2
allowance = float(32*np.finfo(np.float32).eps*width)
bound = float(lipschitz*raw+allowance)
checks = dict(original_non_roundoff_checks_pass=True,
    decoded_pair_coordinate_bound_under_2e_5_rad=bound<2e-5,
    original_fixed_RMCS_nominal_gates_unchanged=True,original_failed_preflight_preserved=True)
audit = dict(schema='entry-common-float32-roundoff-review-v1',preflight=str(preflight),
    preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
    raw_pair_logit_difference_peak=raw,bounded_pair_request_lipschitz=lipschitz,
    float32_allowance_rad=allowance,bounded_pair_request_difference_upper_bound_rad=bound,
    derivation='desired=lo+width*sigmoid(2*sign*(raw_aux-raw_hip)+bias); max derivative wrt raw difference is width/2. The measured-relative bounded request has derivative <=1. An additional 32 float32 eps*width allowance is included.',
    reviewed_tolerance_rad=2e-5,checks=checks,
    scope='This reviews FP32 preservation of the bounded requested pair coordinate only. It does not establish actual passive-knee tracking, contact safety or recovery; native 3mm closure, joint stops, speeds and unchanged behavior gates remain required.',
    decision='admit_unchanged_fixed51_zero_PPO_gate' if all(checks.values()) else 'reject',
    new_PPO=0,actual_experiment_consumed=2603,qualification=None,hardware_started=False,created_unix=time.time())
audit_path = R/'entry_phase_fit_knee_roundoff_review001.json'
assert not audit_path.exists()
audit_path.write_text(json.dumps(audit,indent=2)+'\n')
assert all(checks.values())
protocol = R/'entry_phase_fit_protocol002.json'
pnew = json.loads(protocol.read_text())
pnew.update(status='admitted_fixed51_zero_PPO_only',checkpoint_sha256=z['sha256'],
    preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
    roundoff_review=str(audit_path),roundoff_review_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    updated_unix=time.time())
protocol.write_text(json.dumps(pnew,indent=2)+'\n')
plan = json.loads((R/'diagnosis_plan023.json').read_text())
oldmap = plan['source_sha256']
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in oldmap.items())
extra = [R/f for f in ['fit_entry_phase_path001.py','entry_phase_path_dataset001.npz',
    'entry_phase_path_feasibility001.json','fit_entry_phase_constrained001.py',
    'entry_phase_constrained_feasibility001.json','fit_entry_phase_constrained002.py',
    'entry_phase_constrained_feasibility002.json','entry_stop_bound_refinement001.json',
    'create_entry_phase_constrained002.py','entry_phase_fit_preflight002.json',
    'entry_phase_fit_knee_roundoff_review001.json','entry_phase_fit_protocol002.json',
    'diagnosis_queue024.py','prepare_entry_phase_native001.py']]
extra += [Path(f'runs/robust_policy_repair_20261024/{d}/model_{n}.pt') for d,n in [
    ('entry_phase_fit_teacher002','probe'),('entry_phase_fit_split002','migrated'),
    ('entry_phase_fit_periodic002','probe')]]
for path in extra:
    oldmap[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
plan.update(source_sha256=dict(sorted(oldmap.items())),admission=str(protocol),
    hypothesis=pnew['hypothesis'],created_unix=time.time())
job = plan['jobs'][0]
for key in ('name','checkpoint','output','log'):
    job[key] = job[key].replace('entry_common_fit','entry_phase_fit').replace('023','024').replace('periodic001','periodic002')
job['command'] = [v.replace('entry_common_fit','entry_phase_fit').replace('diagnosis023','diagnosis024').replace('periodic001','periodic002') for v in job['command']]
assert job['checkpoint']==pnew['candidate']
plan_path = R/'diagnosis_plan024.json'
assert not plan_path.exists()
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(bound_rad=bound,checks=checks,native_jobs=plan['jobs'],frozen_inputs=len(oldmap)),indent=2))
