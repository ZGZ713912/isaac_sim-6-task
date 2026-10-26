"""Only a passing fixed pilot may enter the original moving-fall target assessment."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
decision = json.loads((R/'motion_round015_fixed_decision001.json').read_text())
assert decision['decision']=='admit_zero_PPO_expanded_target_review' and all(decision['checks'].values())
p = json.loads((R/'plan015_motion_output001.json').read_text())
assert json.loads((Path(p['supervisor_output'])/'progress.json').read_text())['status']=='review_ready'
assert all(digest(Path(n))==h for n,h in p['source_sha256'].items())
cp = decision['checkpoint']
assert digest(cp)==decision['checkpoint_sha256']
old = json.loads((R/'diagnosis_plan037.json').read_text())
original = next(j for j in old['jobs'] if j['name']=='quiet_stance050_nominal_pair')
job = dict(original)
old_cp, old_output = job['checkpoint'], job['output']
job.update(name='motion_output015_nominal_target', checkpoint=cp,
    output=str(B/'diagnosis045_motion_output015_nominal_pair'),
    log=str(R/'diagnosis045_motion_output015_nominal_pair.log'))
job['command'] = [cp if x==old_cp else job['output'] if x==old_output else x for x in original['command']]
assert not Path(job['output']).exists()
queue = R/'diagnosis_queue045.py'
assert not queue.exists() and not (R/'diagnosis_plan045.json').exists()
queue.write_text((R/'diagnosis_queue044.py').read_text().replace('plan044','plan045').replace('queue044','queue045'))
frozen = dict(p['source_sha256'])
extras = [R/'plan015_motion_output001.json', R/'motion_round015_fixed_decision001.json',
    R/'motion_round015_endpoint001.json', R/'audit_motion_round015_endpoint001.py',
    R/'review_motion_round015_fixed001.py', Path(cp), queue, Path(__file__)]
frozen.update({str(x):digest(x) for x in extras})
protocol = dict(schema='motion-output-actual-target-assessment-v1',
    requirement='Original35 cases,2replicas,211131,12s, nominal domains and robust/reference variants. Require actual moving-fall recovery with continuation>=1, practical compound8/inverted10/right4/left2 and ordinary motion>=42; no surrogate reward or easier push.',
    decision_scope='Zero-PPO development assessment only. RMCS expansion requires the actual target gain and family/motion nonregression; no automatic extension/adoption or formal qualification.',
    new_PPO=0, qualification=None)
plan = dict(schema='serial-native-diagnosis-v2', jobs=[job], source_sha256=frozen,
    protocol=protocol, new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R/'diagnosis_plan045.json', plan)
write_json(R/'motion_expanded_target_admission001.json', dict(plan=str(R/'diagnosis_plan045.json'),
    plan_sha256=digest(R/'diagnosis_plan045.json'), checkpoint=cp, checkpoint_sha256=digest(cp),
    new_PPO=0, no_automatic_PPO_or_adoption=True, qualification=None))
print(json.dumps(dict(plan=str(R/'diagnosis_plan045.json'), frozen_inputs=len(frozen), new_PPO=0)))
