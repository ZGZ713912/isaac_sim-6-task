"""Current9a paired development control then exactly zero optimizer rollout."""
from pathlib import Path
import sys,json,time
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
assert json.loads((R/'diagnosis_queue038.json').read_text())['status']=='completed'
assert json.loads((R/'diagnosis038_closed_source_seal001.json').read_text())['all_frozen_sources_verified']
old=json.loads((R/'diagnosis_plan038.json').read_text());changed=[n for n,h in old['source_sha256'].items() if digest(n)!=h]
assert set(changed)=={'scripts/train_robust_getup.py','scripts/export_robust_getup.py'},changed
for n in ('neutral_stance_root_preflight001.json','fallen_output_contract_CPU_preflight001.json','fallen_collector_preflight001.json','fallen_prepare_actual_preflight001.json'):
 assert all(json.loads((R/n).read_text())['checks'].values())
assert '91 passed' in (R/'fallen_function_root_regression_tests001.log').read_text()
pre=json.loads((R/'fallen_prepare_actual_preflight001.json').read_text());args=list(pre['command']);assert args[args.index('--updates')+1]=='8'
cp=str(B/'fallen_output_root001/model_initial.pt');assert args[args.index('--resume')+1]==cp
protocol=dict(schema='fallen-output-native-preflight-v1',source=cp,source_sha256=digest(cp),source_clock=1501,
 jobs=['Same native current9a51-case development control at241204','Single actual256x512 sampled training rollout and native GAE; collector exits before any optimizer attempt'],
 gate=dict(optimizer_attempts=0,new_PPO=0,restored_actor_critic_Adam_RNG_exact=True,original_mean_bit_exact=True,
 original_conditioned_std_bit_exact=True,trainable_actor_tensors=['mlp.mean.recovery.4.weight','mlp.mean.recovery.4.bias']),
 callback_order='Installed native runner compute_returns precedes outer optimizer-attempt accounting; tested by actual source inspection and collector import.',
 no_advanced_RNG_resume_claim='model_stopped is the exact native initial checkpoint copy, not an advanced physical/RNG training resume',
 qualification=None,hardware_started=False,actual_experiment_consumed=2630,repair_optimizer_updates=250,created_unix=time.time())
write_json(R/'fallen_output_native_preflight_protocol001.json',protocol)
q=(R/'diagnosis_queue038.py').read_text().replace('plan038','plan039').replace('queue038','queue039')
q=q.replace('if completion.get("status") != "completed" or', 'if completion.get("status") != job.get("completion_status", "completed") or')
(R/'diagnosis_queue039.py').write_text(q)
py='/home/noir/miniconda3/envs/isaacsim6/bin/python';source=str(B/'motion_damping080002/model_probe.pt');out=str(B/'diagnosis039_current9a_development')
jobs=[dict(name='current9a_development',checkpoint=source,output=out,log=str(R/'diagnosis039_current9a_development.log'),
 command=[py,'scripts/evaluate_gap_robustness.py','--checkpoint',source,'--output',out,'--bank',str(R/'pose_bank_rmcs003.json'),'--actuator-timing','native','--seconds','8','--seed','241204'])]
out=str(B/'diagnosis039_fallen_output_native_zero');args[args.index('--output')+1]=out
jobs.append(dict(name='fallen_output_native_zero',checkpoint=cp,output=out,log=str(R/'diagnosis039_fallen_output_native_zero.log'),completion_status='stopped',
 command=[py,str(R/'collect_fallen_output_rollout001.py'),'--diagnostic-rollout-only']+args))
for j in jobs:assert not Path(j['output']).exists()
names=set(old['source_sha256'])
for n in ('src/wheeled_algo/fallen_output_checkpoint.py','scripts/migrate_fallen_output_function_root.py',cp,str(Path(cp).with_suffix('.json')),
 str(B/'quiet_stance000003/model_probe.pt'),str(B/'quiet_stance000003/model_probe.json')):names.add(n)
for n in ('diagnosis038_closed_source_seal001.json','quiet_stance_long_standing_decision001.json','fallen_output_contract_CPU_preflight001.json',
 'preflight_fallen_output_contract001.py','fallen_collector_preflight001.json','preflight_fallen_collector001.py','collect_fallen_output_rollout001.py',
 'fallen_prepare_actual_preflight001.json','preflight_fallen_prepare_actual001.py','fallen_function_root_regression_tests001.log',
 'neutral_stance_root_preflight001.json','preflight_neutral_stance_root001.py','fallen_output_native_preflight_protocol001.json','diagnosis_queue039.py'):
 names.add(str(R/n))
names.add(str(Path(__file__)));frozen={n:digest(n) for n in sorted(names)}
plan=dict(schema='serial-native-diagnosis-v2',jobs=jobs,source_sha256=frozen,protocol=str(R/'fallen_output_native_preflight_protocol001.json'),new_PPO=0,
 actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'diagnosis_plan039.json').exists();write_json(R/'diagnosis_plan039.json',plan)
write_json(R/'fallen_native_preflight_admission001.json',dict(plan=str(R/'diagnosis_plan039.json'),plan_sha256=digest(R/'diagnosis_plan039.json'),
 frozen_inputs=len(frozen),new_PPO=0,new_training_admitted=False,qualification=None,hardware_started=False))
print(json.dumps(dict(plan=str(R/'diagnosis_plan039.json'),frozen_inputs=len(frozen),new_PPO=0)))
