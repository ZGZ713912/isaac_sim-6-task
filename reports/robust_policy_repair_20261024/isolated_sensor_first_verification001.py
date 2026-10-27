"""Authenticate the independent fresh pilot; no loading inside its learner."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
RUN=BASE/'runs/scratch_standing_20261026/pilot012_isolated_sensor_requested001'
OLD=BASE/'runs/scratch_standing_20261026/pilot010_requested_domains_retry001'
OUT=REPORT/'isolated_sensor_first_verification001.json'
IDENTITY=REPORT/'scratch_pilot012_launch_identity001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists() and not IDENTITY.exists()
progress=read(RUN/'progress.json');init=read(RUN/'initialization.json');old=read(OLD/'initialization.json')
pid=progress['pid'];start=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
audit=read(RUN/'first_update_audit.json');manifest=read(RUN/'source_hashes.json')
sample=read(RUN/'randomization_study.initial_samples.json');previous=read(OLD/'randomization_study.initial_samples.json')
args=progress['args'].copy();old_args=read(OLD/'progress.json')['args'].copy()
args.pop('standing_reward_shape')
for values in (args,old_args):
    values.pop('output');values.pop('updates')
task=read(RUN/'task_contract.json')['settings'];old_task=read(OLD/'task_contract.json')['settings']
changes={k for k in set(task)|set(old_task) if task.get(k)!=old_task.get(k)}
checks=dict(exact_live_process=start==progress['start_ticks'],actual_native_PPO=progress['completed_updates']>=1,
    no_checkpoint_reads=progress['checkpoint_reads']==[],fresh_actor_critic=not init['checkpoint_loaded'] and not init['teacher_loaded'],
    empty_initial_Adam=init['optimizer_states']==0,same_initial_actor_as010=init['actor_tensor_sha256']==old['actor_tensor_sha256'],
    same_initial_critic_as010=init['critic_tensor_sha256']==old['critic_tensor_sha256'],
    native_Adam_calls=audit['optimizer_calls']==audit['committed_optimizer_calls']==12,
    actual_actor_critic_updates=audit['actor_changed_parameter_tensors']==9 and audit['critic_changed_parameter_tensors']==8,
    finite_first_update=audit['all_recorded_values_finite'],
    frozen_sources_exact=all(sha(RUN/'source_snapshot'/name)==digest for name,digest in manifest.items()),
    checkout_sources_exact=all(sha(WORK/name)==digest for name,digest in manifest.items()),
    source_matches_one_commit=progress['committed_source']['all_source_bytes_match_commit'],
    same17_actual_sensor_actuator_rigid_arrays=sample['samples']==previous['samples'] and len(sample['samples'])==17,
    same_two_actual_ground_friction_arrays=all(sample['contact_material'][key]==previous['contact_material'][key]
        for key in ['ground_static_friction_per_environment','ground_dynamic_friction_per_environment']),
    same_complete_material_and_rigid_summaries=sample['contact_material']==previous['contact_material'] and sample['rigid_summary']==previous['rigid_summary'],
    same_native_agent_hyperparameters=read(RUN/'agent_config.json')==read(OLD/'agent_config.json'),
    only_clock_metadata_and_gaussian_default_added_to_task=changes=={'version','standing_reward_shape','domain_sensor_clock_version'},
    same_CLI_except_explicit_gaussian_and_short_budget=args==old_args,
    gaussian_reward_installed=progress['precise_reward']['shape']=='gaussian',
    actual_clock_matches_task=progress['domain_sensor_clock_version']==task['domain_sensor_clock_version']=='one-sample-per-tick-selective-reset-v1')
sys.path.insert(0,str(WORK/'src'))
import torch
from wheeled_algo.scratch_standing import tensor_hash
from wheeled_tasks.chassis.randomization_study import scalar_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
initial=torch.load(RUN/'model_initial.pt',map_location='cpu',weights_only=False)
checks['actual_initial_Adam_empty']=not initial['optimizer_state_dict']['state']
checks['actual_initial_tensor_hashes_match']=all(tensor_hash(initial[role+'_state_dict'])==init[role+'_tensor_sha256'] for role in ['actor','critic'])
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
checks['actual_GPU_process_present']=str(pid) in gpu.splitlines()
resolved=read(RUN/'randomization_study.resolved.json')
expected={'DomainStudyConfig/'+k:v for k,v in scalar_metrics(resolved).items()}
expected.update({'DomainSensorsConfig/selective_reset_clock_enabled':1.,'PrecisionStandingConfig/joint_long_tail_enabled':0.})
events=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
checks['all_RSL_config_scalars_match']=all(any(point.step==0 and math.isclose(point.value,value,rel_tol=2e-6,abs_tol=2e-8)
    for point in events.Scalars(tag)) for tag,value in expected.items())
assert all(checks.values()),checks
result=dict(schema='isolated-sensor-reset-first-fresh-PPO-verification-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
    run=str(RUN),pid=pid,start_ticks=start,completed_updates=progress['completed_updates'],requested_updates=128,
    source_count=len(manifest),source_commit=progress['committed_source']['commit'],
    archive_sha256=progress['committed_source']['archive_sha256'],checks=checks,all_checks_pass=True,
    actual_initial_domain_arrays_compared=19,RSL_config_scalars_verified=len(expected),
    task_metadata_changes=sorted(changes),actual_randomized_lanes=sum(sample['samples']['enabled']),
    comparison='Only the reset-isolation fix and explicit metadata versus original010; full requested widths, initial weights and all19 domain arrays match. No gradual admission or long-tail reward.',
    new_PPO_by_reader=0,native_behavior_qualified=False,hardware_ready=False,
    producer_sha256=sha(__file__),evidence_sha256={name:sha(RUN/name) for name in ['initialization.json','first_update_audit.json',
        'randomization_study.initial_samples.json','agent_config.json','source_revision.json']})
write(OUT,result);(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
identity=dict(schema='scratch-from-zero-launch-identity-v2',observed_utc=result['observed_utc'],pid=pid,start_ticks=start,
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0'),source_checkout=str(WORK),
    run=str(RUN),commit=result['source_commit'],source_archive_sha256=result['archive_sha256'],source_count=len(manifest),
    requested_updates=128,prior_closed_fresh_updates=3585,exec_session_id=79576,initialization=init,
    first_verification=str(OUT),scope='Sensor-clock repair controlled from-zero middle-height foundation only',hardware_ready=False)
write(IDENTITY,identity)
ledger_path=REPORT/'scratch_campaign001.json';raw=ledger_path.read_bytes();ledger=json.loads(raw)
assert ledger['active_pid'] is None and ledger['prior_closed_fresh_updates']==3585
ledger.update(status='training_unqualified',active_run=str(RUN),active_pid=pid,active_start_ticks=start,
    active_launch_identity=str(IDENTITY),latest_training_launch=str(IDENTITY),requested_updates=128,
    completed_updates_observed=progress['completed_updates'],active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed=3585+progress['completed_updates'],total_from_zero_completed_observed=3585+progress['completed_updates'],
    latest_first_verification=str(OUT),do_not_modify_live_src=True,do_not_modify_training_entrypoint=True,
    active_evaluation=None,active_evaluation_pid=None,active_evaluation_start_ticks=None,
    do_not_modify_frozen_evaluator_sources=False,
    latest_observation='Sensor-reset isolation passed actual native neighbor-reset proof; new fresh pilot passed19 actual draw, initial weight and native Adam checks. No behavior improvement claimed.',
    next_action='Compare common100 RSL updates and seal128, then cold native nominal/requested tests; retain complete permanent capabilities and no long-train qualification.',
    observed_utc=result['observed_utc'],last_observed_utc=result['observed_utc'])
assert ledger_path.read_bytes()==raw
temporary=ledger_path.with_suffix('.isolatedfirst001.tmp');write(temporary,ledger);temporary.replace(ledger_path)
print(json.dumps(dict(checks=len(checks),all_pass=True,completed_updates=result['completed_updates'],
    actual_initial_domain_arrays_compared=19,RSL_config_scalars_verified=len(expected))))
