"""One finite paired experiment, permitted only after native reward identity."""
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['src','scripts']
from train_robust_getup import digest,write_json

R=Path('reports/robust_policy_repair_20261024')
B=Path('runs/robust_policy_repair_20261024')
audit_path=R/'quiet_reward_native_audit001.json'
audit=json.loads(audit_path.read_text())
assert all(audit['checks'].values()) and audit['decision']=='bounded_training_pair_may_be_reviewed'
pre=json.loads((R/'quiet_training_preflight001.json').read_text())
assert all(pre['checks'].values())
assert digest(Path('scripts/train_robust_getup.py'))==pre['training_source_sha256']
frozen=json.loads((R/'diagnosis_plan030.json').read_text())['source_sha256']
assert all(digest(Path(n))==h for n,h in frozen.items())
extra=[audit_path,R/'quiet_training_preflight001.json',R/'prepare_quiet_training_pair001.py',
       R/'run_quiet_paired_pilot001.py',R/'audit_quiet_endpoint001.py',
       Path('scripts/run_bounded_gap_round.py'),Path(__file__),
       B/'low_wheel_diagonal_restoring_probe001/model_probe.pt']
extra += [Path(x['checkpoint']) for x in pre['checkpoints']]
frozen.update({str(p):digest(p) for p in extra})
plans=[]
for index,row in enumerate(pre['checkpoints'],10):
    args=pre['training_arguments'].copy()
    args[args.index('--resume')+1]=row['checkpoint']
    train=str(B/(f'pilot{index:03d}_quiet_'+row['name']))
    args[args.index('--output')+1]=train
    out=B/f'round{index:03d}'
    plan=dict(schema='bounded-quiet-pair-v1',status='admitted_once_bounded8',
        supervisor_output=str(out),training_output=train,source_checkpoint=row['checkpoint'],
        source_checkpoint_sha256=row['sha256'],bank=str(R/'pose_bank_rmcs003.json'),
        training_command=['/home/noir/miniconda3/envs/isaacsim6/bin/python',str(R/'run_quiet_paired_pilot001.py'),*args],
        evaluation_output=str(B/(f'pilot{index:03d}_quiet_fixed')),
        baseline_evaluation=str(B/'diagnosis013_low_diagonal_fixed'),
        prior_experiment_consumed=2605+(index-10)*8,source_clock=1501,new_updates=8,
        trainable_actor_parameters=24,role=row['name'],seed=241510,
        standing_sway_m=.05 if row['name']=='control' else .02,
        source_sha256=frozen,
        hypothesis='A2cm position reward supplies correction credit before5cm; test against original5cm with identical function, optimizer initialization, seed and24 trainable parameters.',
        scope='Independent quiet component. Best recovery/motion candidate remains unchanged.',
        review_gate=dict(fixed_no_previous_standing_pass_lost=True,fixed_standing_at_least=20,
            motion_MAE_at_most=.135,entry_recovery_at_least=6,entry_followup_at_least=5,
            fallen_recovery_at_least=2,fallen_followup_at_least=1,
            entry_leg_peak_at_most_rad_s=8.486846433410645,
            fallen_leg_peak_at_most_rad_s=24.11666153503418,
            standing_leg_peak_at_most_rad_s=6.073664283752441,
            impact_terminations_at_most=2,
            closure_at_most_m=.003,passive_motor_peak_nm=0.,
            improvement_before_expansion='fixed21/21 OR at least20percent reduction in failing low-case excursion without any lost prior pass',
            further_development='same21 seed241204; no prior pass lost and high_wobble improves before any composition',
            no_automatic_extension=True),
        new_PPO_admitted=8,qualification=None,hardware_started=False,created_unix=time.time())
    path=R/f'plan{index:03d}_quiet_{row["name"]}001.json'
    assert not path.exists()
    write_json(path,plan);plans.append(str(path))
write_json(R/'quiet_training_pair_admission001.json',dict(schema='quiet-pair-admission-v1',
    plans=plans,total_new_PPO_admitted=16,per_branch_source_clock=1501,per_branch_terminal_clock=1509,
    prior_experiment_consumed=2605,planned_after_pair=2621,
    sequence='control8+fixed review then tight8+fixed review; native processes serial; pair arms never merge learning clocks',
    native_reward_audit_sha256=digest(audit_path),all_frozen_inputs=len(frozen),
    no_automatic_extension=True,qualification=None,hardware_started=False,created_unix=time.time()))
print(json.dumps(dict(plans=plans,inputs=len(frozen),new_PPO_admitted=16),indent=2))
