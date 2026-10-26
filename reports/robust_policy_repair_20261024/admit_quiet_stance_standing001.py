"""Existing development seed and unchanged formal30s standing, no PPO."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
prior=json.loads((R/'quiet_stance_expanded_decision001.json').read_text());assert all(prior['checks'].values())
assert json.loads((R/'diagnosis_queue037.json').read_text())['status']=='completed'
p=json.loads((R/'diagnosis_plan037.json').read_text());assert all(digest(n)==h for n,h in p['source_sha256'].items())
cp=str(B/'quiet_stance050003/model_probe.pt');base=Path('reports/unified_robust_20261021/acceptance_protocol004.json')
formal=json.loads(base.read_text());assert digest(base)=='05e6469f6da54b726fc2bfb5e53ff7f90f996bfbcceaf942e453f6437ee39ab3'
protocol=dict(schema='quiet-stance-standing-review-v1',checkpoint=cp,checkpoint_sha256=digest(cp),
 development_seed=241204,development_scope='Previously used development seed, not a fresh holdout.',
 formal_protocol=str(base),formal_protocol_sha256=digest(base),formal_long_standing=formal['long_standing_confirmation'],
 criteria='Call the existing assess_extended_standing without modifying any original threshold or success definition.',
 full_goal_remaining='Moving-fall success and full nominal confirmation, plus fresh fullDR/RMCS holdouts remain required even if long standing passes.',
 new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'quiet_stance_standing_protocol001.json').exists();write_json(R/'quiet_stance_standing_protocol001.json',protocol)
(R/'diagnosis_queue038.py').write_text((R/'diagnosis_queue037.py').read_text().replace('plan037','plan038').replace('queue037','queue038'))
py='/home/noir/miniconda3/envs/isaacsim6/bin/python';jobs=[]
for name,script,extra in [('quiet_stance050_development','scripts/evaluate_gap_robustness.py',['--bank',str(R/'pose_bank_rmcs003.json'),'--actuator-timing','native','--seconds','8','--seed','241204']),
 ('quiet_stance050_long_standing','scripts/evaluate_robust_getup_batch.py',['--checkpoint-task-profile','--seeds','211322','211626','--replicas','16','--variants','robust','--domain-scale','.35','--friction-min','.3','--seconds','30','--isolated-origins','--case-names','stand_low','stand','stand_high'])]:
 out=str(B/('diagnosis038_'+name));assert not Path(out).exists();jobs.append(dict(name=name,checkpoint=cp,output=out,log=str(R/('diagnosis038_'+name+'.log')),command=[py,script,'--checkpoint',cp,'--output',out]+extra))
frozen=p['source_sha256']
for n in ('quiet_stance_expanded_decision001.json','quiet_stance_standing_protocol001.json','review_quiet_stance_expanded001.py','quiet_stance_RMCS_comparison001.json','diagnosis_queue038.py'):
 frozen[str(R/n)]=digest(R/n)
frozen[str(base)]=digest(base);frozen[str(Path(__file__))]=digest(__file__)
plan=dict(schema='serial-native-diagnosis-v2',jobs=jobs,source_sha256=frozen,protocol=str(R/'quiet_stance_standing_protocol001.json'),new_PPO=0,
 actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'diagnosis_plan038.json').exists();write_json(R/'diagnosis_plan038.json',plan)
write_json(R/'quiet_stance_standing_admission001.json',dict(plan=str(R/'diagnosis_plan038.json'),plan_sha256=digest(R/'diagnosis_plan038.json'),frozen_inputs=len(frozen),
 new_PPO=0,qualification=None,hardware_started=False))
print(json.dumps(dict(plan=str(R/'diagnosis_plan038.json'),frozen_inputs=len(frozen),jobs=[x['name'] for x in jobs],new_PPO=0)))
