"""Expand a nonregressing fixed pilot to original moving-fall and RMCS tests, with zero PPO."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
decision = json.loads((R / 'fallen_round014_fixed_decision001.json').read_text())
assert decision['decision'] == 'admit_zero_PPO_expanded_target_review' and all(decision['checks'].values())
p = json.loads((R / 'plan014_fallen_output001.json').read_text())
assert json.loads((Path(p['supervisor_output']) / 'progress.json').read_text())['status'] == 'review_ready'
assert all(digest(Path(n)) == h for n, h in p['source_sha256'].items())
cp = decision['checkpoint']
assert digest(cp) == decision['checkpoint_sha256']
old = json.loads((R / 'diagnosis_plan037.json').read_text())
jobs = []
for original in reversed(old['jobs']):
    j = dict(original)
    oldcp, oldout = j['checkpoint'], j['output']
    j['name'] = j['name'].replace('quiet_stance050', 'fallen_output014')
    j['checkpoint'] = cp
    j['output'] = oldout.replace('diagnosis037_quiet_stance050', 'diagnosis040_fallen_output014')
    j['log'] = j['log'].replace('diagnosis037_quiet_stance050', 'diagnosis040_fallen_output014')
    j['command'] = [cp if x == oldcp else j['output'] if x == oldout else x for x in j['command']]
    assert not Path(j['output']).exists()
    jobs.append(j)
queue = R / 'diagnosis_queue040.py'
assert not queue.exists() and not (R / 'diagnosis_plan040.json').exists()
queue.write_text((R / 'diagnosis_queue039.py').read_text().replace('plan039', 'plan040').replace('queue039', 'queue040'))
frozen = dict(p['source_sha256'])
extra = [R / 'plan014_fallen_output001.json', R / 'fallen_round014_fixed_decision001.json',
    R / 'fallen_round014_endpoint001.json', R / 'audit_fallen_round014_endpoint001.py',
    R / 'review_fallen_round014_fixed001.py', Path(cp), queue, Path(__file__)]
frozen.update({str(x): digest(x) for x in extra})
plan = dict(schema='serial-native-diagnosis-v2', jobs=jobs, source_sha256=frozen,
    protocol=str(R / 'plan014_fallen_output001.json'),
    hypothesis='The learned fallen output must improve actual moving-fall recovery and preserve RMCS/ordinary behavior, under the original conditions.',
    new_PPO=0, actual_experiment_consumed=2638, repair_optimizer_updates=258,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'diagnosis_plan040.json', plan)
write_json(R / 'fallen_expanded_admission001.json', dict(plan=str(R / 'diagnosis_plan040.json'),
    plan_sha256=digest(R / 'diagnosis_plan040.json'), checkpoint=cp, checkpoint_sha256=digest(cp),
    frozen_inputs=len(frozen), new_PPO=0, no_automatic_PPO_or_adoption=True,
    qualification=None, hardware_started=False, created_unix=time.time()))
print(json.dumps(dict(plan=str(R / 'diagnosis_plan040.json'), frozen_inputs=len(frozen),
    jobs=[x['name'] for x in jobs], new_PPO=0), indent=2))
