"""Seal completed paired comparisons and the separate common-encoder probe."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
QUEUE=REPORT/'requested_v2_evaluation_queue002'
OUT=REPORT/'requested_domains_closed005.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists()
queue=read(QUEUE/'receipt.json')
assert queue['status']=='completed_unqualified' and queue['active_child'] is None
assert not Path('/proc/'+str(queue['pid'])).exists()
assert queue['new_PPO']==0 and not queue['optimizer_constructed']
assert sha(queue['checkpoint'])==queue['checkpoint_sha256']
seal=read(REPORT/'scratch_pilot010_closed_seal001.json')
assert seal['all_checks_pass'] and seal['completed_updates']==512
assert not Path('/proc/280546').exists()
rows=[]
for profile in queue['profiles']:
    assert sha(Path(profile['run'])/'receipt.json')==profile['receipt_sha256']
    assert sha(profile['review'])==profile['review_sha256']
    checks=profile['paired_nonintervened_draw_checks']
    assert checks is None or all(checks.values())
    rows.append(dict(profile=profile['profile'],middle_height=profile['middle_height'],
        all_heights=profile['all_heights'],paired_checks=checks,
        receipt_sha256=profile['receipt_sha256'],review_sha256=profile['review_sha256']))
original=read(QUEUE/'eval_requested/receipt.json')
common=read(REPORT/'eval010_shared_encoder_zero001/receipt.json')
review=read(REPORT/'eval010_shared_encoder_zero001.review.json')
assert common['status']=='completed_unqualified' and not Path('/proc/'+str(common['pid'])).exists()
assert common['checkpoint_sha256']==original['checkpoint_sha256']==queue['checkpoint_sha256']
assert common['seed']==original['seed']==261201
assert common['new_PPO']==0 and not common['optimizer_constructed']
assert common['physics_runtime']==original['physics_runtime']
a,b=original['evaluation']['actual_domain_samples'],common['evaluation']['actual_domain_samples']
assert set(a)==set(b) and len(a)==19
pair={name:a[name]==b[name] for name in a if name!='pd_encoder_bias'}
assert all(pair.values()) and a['pd_encoder_bias']!=b['pd_encoder_bias']
assert len(b['pd_encoder_bias'])==len(b['policy_bias'])==21
for pd,policy in zip(b['pd_encoder_bias'],b['policy_bias']):
    assert [pd[i] for i in (0,1,3,4)]==policy[6:10] and pd[2]==pd[5]==0.
middle=next(r for r in review['groups'] if r['height_m']==.305)
assert middle['standing_pass_v2_count']==0
result=dict(schema='requested-domain-completed-controlled-comparison-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(),completed_fresh_updates=512,
    total_closed_fresh_updates=3457,checkpoint=queue['checkpoint'],checkpoint_sha256=queue['checkpoint_sha256'],
    training_integrity_seal_sha256=sha(REPORT/'scratch_pilot010_closed_seal001.json'),
    profiles=rows,queue_sha256=sha(QUEUE/'receipt.json'),all_learners_and_native_evaluators_exited=True,
    common_encoder_probe=dict(middle_height=middle,all_heights=review['groups'],
        same_native_physics=True,non_PD_draw_checks=pair,PD_bias_equals_policy_bias_in_motor_order=True,
        receipt_sha256=sha(REPORT/'eval010_shared_encoder_zero001/receipt.json'),
        review_sha256=sha(REPORT/'eval010_shared_encoder_zero001.review.json'),
        interpretation='Immediate contact/drift changes do not restore standing. This does not measure fresh-training effects of correlated encoder zeros.'),
    conclusion='The new512-update actor fails even nominal physics. Each paired single-group removal and common-encoder probe remains0/7 at the trained height. No single test-time perturbation explains or repairs the learned balance regression.',
    next_fresh_experiment=dict(updates=128,change='joint-long-tail standing reward only',
        retained='full requestedDR widths, independent encoder zeros, seed, actor/critic architecture, optimizer, reset/termination,256x128 rollout and305mm training scope',
        from_zero=True,checkpoint_or_teacher_reads_forbidden=True,
        purpose='Test whether retaining a corrective reward outside the narrow joint peak prevents collapse of learned balance feedback.',
        admission='Must compare actual trajectories; mathematical reward sensitivity and CPU preflight are not policy success.'),
    long_training_started=False,ONNX_exported=False,RMCS_deployed=False,hardware_ready=False,qualification=False,
    producer_sha256=sha(Path(__file__)))
write(OUT,result)
(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path=REPORT/'scratch_campaign001.json'
raw=ledger_path.read_bytes()
ledger=json.loads(raw)
assert ledger['prior_closed_fresh_updates']==3457 and ledger['active_pid'] is None
ledger.update(status='ready_for_fresh_experiment_unqualified',latest_live_read=str(OUT),
    latest_controlled_comparison=str(OUT),do_not_modify_frozen_evaluator_sources=False,
    serial_evaluation_queue_pid=None,serial_evaluation_queue_start_ticks=None,
    latest_reward_signal_audit=str(REPORT/'reward_signal_audit002/result.json'),
    prepared_next_fresh_experiment=result['next_fresh_experiment'],
    latest_observation=result['conclusion'],next_action='Start the bounded128-update reward-only experiment from zero; retain full acceptance and no long-train admission.',
    observed_utc=result['observed_utc'],last_observed_utc=result['observed_utc'])
assert ledger_path.read_bytes()==raw
temp=ledger_path.with_suffix('.closed005.tmp')
write(temp,ledger)
temp.replace(ledger_path)
print(json.dumps(dict(profile_middle_passes={r['profile']:r['middle_height']['standing_pass_v2_count'] for r in rows},
    shared_middle_passes=middle['standing_pass_v2_count'],all18_non_PD_draws_identical=all(pair.values()),
    total_closed_fresh_updates=3457)))
