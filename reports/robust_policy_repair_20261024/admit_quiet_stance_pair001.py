"""Fresh current-source control then one recovery-scoped authority diagnostic."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
pre=json.loads((R/'quiet_stance_authority_preflight001.json').read_text());smoke=json.loads((R/'quiet_stance_wrapper_smoke001.json').read_text())
assert all(pre['checks'].values()) and all(smoke['checks'].values())
assert pre['probe_sha256']==smoke['probe_sha256']==digest(R/'quiet_gyro_stance_authority_probe001.py')
assert pre['wrapper_sha256']==smoke['wrapper_sha256']==digest(R/'evaluate_quiet_stance_authority_probe001.py')
assert json.loads((R/'diagnosis_queue034.json').read_text())['status']=='completed'
old=json.loads((R/'diagnosis_plan034.json').read_text());frozen=old['source_sha256']
assert all(digest(Path(n))==h for n,h in frozen.items())
protocol=json.loads((R/'quiet_authority_protocol001.json').read_text())
protocol.update(schema='recovery-scoped-quiet-authority-protocol-v2',
    baseline=str(B/'diagnosis035_current_source_control'),historical_reference=str(B/'diagnosis025_motion_damping080_fixed'),
    intervention='Same .5 retained quiet authority multiplied by clamp(1-max(entry,fallen)/.05,0,1); no change in the source model mean when recovery>=.05.',
    hypothesis='Preserve capture damping benefit while leaving measured recovery states untouched; the prior unscoped probe remains rejected for10.8908rad/s speed.',
    paired_task_metadata='Both new runs use the exact same current settings. The control is additionally compared with historical9a to resolve three policy-only task metadata additions.',
    gate_origin='Original034 physical and capture-improvement thresholds unchanged.',created_unix=time.time())
write_json(R/'quiet_stance_protocol001.json',protocol)
queue=(R/'diagnosis_queue031.py').read_text().replace('plan031','plan035').replace('queue031','queue035');(R/'diagnosis_queue035.py').write_text(queue)
extra=[R/n for n in ['quiet_gyro_stance_authority_probe001.py','evaluate_quiet_stance_authority_probe001.py','preflight_quiet_stance_authority001.py',
    'quiet_stance_authority_preflight001.json','smoke_quiet_stance_wrapper001.py','quiet_stance_wrapper_smoke001.json',
    'quiet_stance_protocol001.json','quiet_authority050_fixed_decision001.json','quiet_authority_speed_phase_audit002.json','diagnosis_queue035.py']]+[Path(__file__)]
frozen.update({str(x):digest(x) for x in extra})
cp=protocol['source'];python='/home/noir/miniconda3/envs/isaacsim6/bin/python';bank='reports/robust_policy_repair_20261024/pose_bank_rmcs003.json'
jobs=[]
for name,wrapper,additional in [('current_source_control','scripts/evaluate_gap_robustness.py',[]),
    ('stance_authority050_fixed',str(R/'evaluate_quiet_stance_authority_probe001.py'),['--retained-quiet-authority','.5'])]:
    output=str(B/('diagnosis035_'+name));assert not Path(output).exists()
    jobs.append(dict(name=name,checkpoint=cp,output=output,log=str(R/('diagnosis035_'+name+'.log')),
        command=[python,wrapper]+additional+['--checkpoint',cp,'--output',output,'--bank',bank,'--actuator-timing','native','--seconds','8','--seed','241201']))
plan=dict(schema='serial-native-diagnosis-v2',jobs=jobs,source_sha256=frozen,protocol=str(R/'quiet_stance_protocol001.json'),
    new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'diagnosis_plan035.json').exists();write_json(R/'diagnosis_plan035.json',plan)
write_json(R/'quiet_stance_native_admission001.json',dict(schema='paired-zero-PPO-diagnostic-admission-v1',plan=str(R/'diagnosis_plan035.json'),
    plan_sha256=digest(R/'diagnosis_plan035.json'),frozen_inputs=len(frozen),serial_jobs=2,new_PPO=0,actual_experiment_consumed=2630,
    no_training_or_implementation_adoption_admitted=True,qualification=None,hardware_started=False,created_unix=time.time()))
print(json.dumps(dict(plan=str(R/'diagnosis_plan035.json'),frozen_inputs=len(frozen),serial_jobs=2,new_PPO=0)))
