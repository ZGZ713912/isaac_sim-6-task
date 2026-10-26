"""Admit one zero-PPO policy-authority diagnostic from the unchanged best mean."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
pre=json.loads((R/'quiet_authority_probe_preflight001.json').read_text());smoke=json.loads((R/'quiet_authority_wrapper_smoke003.json').read_text())
assert all(pre['checks'].values()) and all(smoke['checks'].values())
assert pre['probe_sha256']==smoke['probe_sha256']==digest(R/'quiet_gyro_authority_probe001.py')
assert smoke['wrapper_sha256']==digest(R/'evaluate_quiet_authority_probe003.py')
old=json.loads((R/'plan013_projected_entry001.json').read_text())
frozen=old['source_sha256'];assert all(digest(Path(n))==h for n,h in frozen.items())
assert json.loads((B/'round013/progress.json').read_text())['status']=='review_ready'
assert json.loads((R/'progress.json').read_text())['actual_experiment_consumed']==2630
rows=json.loads((R/'posthandoff_authority_audit001.json').read_text())['rows'];names=['real_start_3__clean','real_start_5__pitch_plus2','real_start_6__pitch_plus2']
base=[x for x in rows if x['run']=='diagnosis025_motion_damping080_fixed' and x['name'] in names];assert len(base)==3
cp=B/'motion_damping080002/model_probe.pt';assert digest(cp)==pre['checkpoint_sha256']
protocol=dict(schema='policy-quiet-authority-falsification-v1',source=str(cp),source_sha256=digest(cp),source_clock=1501,
    intervention='Change only the policy method gyro-release factor: retain .5 of the otherwise removed quiet authority. Shared observer/standard-deviation gate is unchanged.',
    hypothesis='Releasing all quiet smoothing/correction during upright zero-command capture may sustain the measured 2rad/s oscillation.',
    baseline=str(B/'diagnosis025_motion_damping080_fixed'),cases=51,seconds=8.,seed=241201,
    gate=dict(standing_at_least=19,motion_MAE_at_most=.135,entry_recovery_at_least=8,entry_followup_at_least=8,
        fallen_recovery_at_least=4,fallen_followup_at_least=2,entry_peak_at_most_rad_s=8.911178255081177,
        fallen_peak_at_most_rad_s=24.11665153503418,impacts_at_most=2,closure_gap_at_most_m=.003,
        passive_peak_nm=0.,pairing_exact=True,
        capture_names=names,capture_angular_RMS_mean_baseline=sum(x['angular_norm_rms'] for x in base)/3.,
        capture_angular_RMS_ratio_at_most=.8,entry_followup_improvement_at_least=9),
    interpretation='Diagnostic mean-function override, not the stored checkpoint mean; no PPO, adoption or export automatically admitted.',
    original_acceptance_conditions_unchanged=True,new_PPO=0,actual_experiment_consumed=2630,
    qualification=None,hardware_started=False,created_unix=time.time())
write_json(R/'quiet_authority_protocol001.json',protocol)
queue=(R/'diagnosis_queue031.py').read_text().replace('plan031','plan034').replace('queue031','queue034')
(R/'diagnosis_queue034.py').write_text(queue)
extra=[R/n for n in ['quiet_gyro_authority_probe001.py','evaluate_quiet_authority_probe003.py','preflight_quiet_authority_probe001.py',
    'quiet_authority_probe_preflight001.json','smoke_quiet_authority_wrapper003.py','quiet_authority_wrapper_smoke003.json',
    'quiet_authority_protocol001.json','posthandoff_authority_audit001.json','projected_followup_audit001.json',
    'projected_round013_endpoint001.json','projected_round013_fixed_decision001.json','diagnosis_queue034.py']]+[Path(__file__)]
frozen.update({str(p):digest(p) for p in extra})
output=str(B/'diagnosis034_quiet_authority050_fixed')
command=['/home/noir/miniconda3/envs/isaacsim6/bin/python',str(R/'evaluate_quiet_authority_probe003.py'),
    '--retained-quiet-authority','.5','--checkpoint',str(cp),'--output',output,'--bank',old['bank'],
    '--actuator-timing','native','--seconds','8','--seed','241201']
plan=dict(schema='serial-native-diagnosis-v2',jobs=[dict(name='quiet_authority050_fixed',checkpoint=str(cp),output=output,
    log=str(R/'diagnosis034_quiet_authority050_fixed.log'),command=command)],source_sha256=frozen,
    protocol=str(R/'quiet_authority_protocol001.json'),new_PPO=0,actual_experiment_consumed=2630,repair_optimizer_updates=250,
    qualification=None,hardware_started=False,created_unix=time.time())
assert not (R/'diagnosis_plan034.json').exists() and not Path(output).exists()
write_json(R/'diagnosis_plan034.json',plan)
write_json(R/'quiet_authority_native_admission001.json',dict(schema='zero-PPO-native-admission-v1',plan=str(R/'diagnosis_plan034.json'),
    plan_sha256=digest(R/'diagnosis_plan034.json'),frozen_inputs=len(frozen),new_PPO=0,actual_experiment_consumed=2630,
    no_training_admission=True,qualification=None,hardware_started=False,created_unix=time.time()))
print(json.dumps(dict(plan=str(R/'diagnosis_plan034.json'),frozen_inputs=len(frozen),new_PPO=0)))
