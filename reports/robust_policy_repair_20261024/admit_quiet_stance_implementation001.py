"""Native checkpoint implementation equivalence only; no training/adoption."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
pre=json.loads((R/'quiet_stance_implementation_preflight001.json').read_text());decision=json.loads((R/'quiet_stance050_fixed_decision001.json').read_text())
assert all(pre['checks'].values()) and all(decision['checks'].values())
assert '60 passed' in (R/'quiet_stance_implementation_tests001.log').read_text()
assert json.loads((R/'diagnosis_queue035.json').read_text())['status']=='completed'
assert json.loads((R/'diagnosis035_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
cp=pre['checkpoint'];assert digest(cp)==pre['checkpoint_sha256'] and digest('src/wheeled_algo/quiet_stance_authority.py')==pre['module_sha256']
protocol=dict(schema='quiet-stance-native-implementation-v1',checkpoint=cp,checkpoint_sha256=pre['checkpoint_sha256'],
 reference='runs/robust_policy_repair_20261024/diagnosis035_stance_authority050_fixed',
 requirement='All56 native arrays exactly reproduce the diagnostic mean, with empty evaluation overrides and authenticated checkpoint settings. No new PPO.',
 scope='Implementation parity only; expanded RMCS/nominal and fresh holdouts still required.',new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,
 qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'quiet_stance_implementation_protocol001.json',protocol)
queue=(R/'diagnosis_queue035.py').read_text().replace('plan035','plan036').replace('queue035','queue036');(R/'diagnosis_queue036.py').write_text(queue)
old=json.loads((R/'diagnosis_plan035.json').read_text());names=set(old['source_sha256'])
#035 was closed with its original hashes before the declared factory extension.
#036 records actual current sources, while its prior seal remains immutable.
names.update(str(x) for x in (Path('src/wheeled_algo/quiet_stance_authority.py'),Path('scripts/migrate_quiet_stance_authority.py'),
 Path('tests/test_quiet_stance_authority.py'),Path(cp),Path(cp).with_suffix('.json'),Path(__file__)))
for n in ('quiet_stance_implementation_preflight001.json','quiet_stance_implementation_tests001.log','preflight_quiet_stance_implementation001.py',
 'quiet_stance_implementation_protocol001.json','quiet_stance050_fixed_decision001.json','diagnosis035_closed_source_seal001.json','current_control035_replay001.json','diagnosis_queue036.py'):
 names.add(str(R/n))
frozen={n:digest(Path(n)) for n in sorted(names)}
out=str(B/'diagnosis036_quiet_stance050_implementation');assert not Path(out).exists()
jobs=[dict(name='quiet_stance050_implementation',checkpoint=cp,output=out,log=str(R/'diagnosis036_quiet_stance050_implementation.log'),
 command=['/home/noir/miniconda3/envs/isaacsim6/bin/python','scripts/evaluate_gap_robustness.py','--checkpoint',cp,'--output',out,
 '--bank',str(R/'pose_bank_rmcs003.json'),'--actuator-timing','native','--seconds','8','--seed','241201'])]
plan=dict(schema='serial-native-diagnosis-v2',jobs=jobs,source_sha256=frozen,protocol=str(R/'quiet_stance_implementation_protocol001.json'),
 new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'diagnosis_plan036.json').exists();write_json(R/'diagnosis_plan036.json',plan)
write_json(R/'quiet_stance_implementation_native_admission001.json',dict(plan=str(R/'diagnosis_plan036.json'),plan_sha256=digest(R/'diagnosis_plan036.json'),
 frozen_inputs=len(frozen),new_PPO=0,qualification=None,hardware_started=False))
print(json.dumps(dict(frozen_inputs=len(frozen),checkpoint=cp,new_PPO=0)))
