"""Expand one exact saved-mean implementation; no PPO or final qualification."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
parity=json.loads((R/'quiet_stance_native_implementation_parity001.json').read_text());assert all(parity['checks'].values())
assert json.loads((R/'diagnosis_queue036.json').read_text())['status']=='completed'
p=json.loads((R/'diagnosis_plan036.json').read_text());assert all(digest(n)==h for n,h in p['source_sha256'].items())
prior=json.loads((R/'motion_damping_expanded_decision001.json').read_text())
cp=parity['checkpoint'];assert digest(cp)==parity['checkpoint_sha256']
protocol=dict(schema='quiet-stance-expanded-zero-PPO-v1',checkpoint=cp,checkpoint_sha256=parity['checkpoint_sha256'],
 baseline_RMCS=str(B/'diagnosis026_motion_damping080_rmcs'),baseline_nominal=str(B/'diagnosis026_motion_damping080_nominal_pair'),
 gate=dict(RMCS_recovery_at_least=57,RMCS_followup_at_least=36,RMCS_tail90_at_least=56,RMCS_survival_at_least=70,
 RMCS_leg_peak_at_most_rad_s=prior['RMCS']['leg_peak'],RMCS_impacts_at_most=2,
 nominal_compound_followup_at_least=8,nominal_inverted_followup_at_least=10,nominal_right_followup_at_least=4,nominal_left_followup_at_least=2,
 nominal_motion_survival_at_least=42,closure_at_most_m=.003,passive_motor_peak_nm=0.),
 gate_origin='Nonregression to best9a actual native results, stricter than the older55-recovery gate. Fixed stance gain remains a development result.',
 remaining='Moving-fall recovery, original30s standing, fresh DR/RMCS holdouts and both formal nominal seeds remain required; no automatic PPO admission.',
 new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'quiet_stance_expanded_protocol001.json').exists();write_json(R/'quiet_stance_expanded_protocol001.json',protocol)
(R/'diagnosis_queue037.py').write_text((R/'diagnosis_queue036.py').read_text().replace('plan036','plan037').replace('queue036','queue037'))
old=json.loads((R/'diagnosis_plan026.json').read_text())
jobs=[]
for j in old['jobs']:
 j=dict(j);j['command']=list(j['command']);oldcp=j['checkpoint'];oldout=j['output'];j['name']=j['name'].replace('motion_damping080','quiet_stance050');j['checkpoint']=cp;j['output']=oldout.replace('diagnosis026_motion_damping080','diagnosis037_quiet_stance050');j['log']=j['log'].replace('diagnosis026_motion_damping080','diagnosis037_quiet_stance050');j['command']=[cp if x==oldcp else j['output'] if x==oldout else x for x in j['command']];assert not Path(j['output']).exists();jobs.append(j)
frozen=p['source_sha256']
for n in ('quiet_stance_native_implementation_parity001.json','review_quiet_stance_native_implementation001.py','quiet_stance_expanded_protocol001.json','diagnosis_queue037.py','motion_damping_expanded_decision001.json'):
 frozen[str(R/n)]=digest(R/n)
frozen[str(Path(__file__))]=digest(__file__)
plan=dict(schema='serial-native-diagnosis-v2',jobs=jobs,source_sha256=frozen,protocol=str(R/'quiet_stance_expanded_protocol001.json'),
 new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'diagnosis_plan037.json').exists();write_json(R/'diagnosis_plan037.json',plan)
write_json(R/'quiet_stance_expanded_admission001.json',dict(plan=str(R/'diagnosis_plan037.json'),plan_sha256=digest(R/'diagnosis_plan037.json'),frozen_inputs=len(frozen),
 new_PPO=0,qualification=None,hardware_started=False))
print(json.dumps(dict(plan=str(R/'diagnosis_plan037.json'),frozen_inputs=len(frozen),jobs=[x['name'] for x in jobs],new_PPO=0)))
