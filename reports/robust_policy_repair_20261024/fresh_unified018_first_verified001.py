"""Read the actual fresh mixed-task first update without touching its learner."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/fresh_unified_20261027/pilot018_velocity_quartic_physical_speed_mixed1536'
FAILED = BASE/'runs/fresh_unified_20261027/pilot017_dense_pose_physical_speed_mixed1536'
OUT = BASE/REL/'fresh_unified018_first_verified001.json'
IDENTITY = BASE/REL/'fresh_unified018_launch_identity001.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exec-session-id',type=int,required=True)
args=parser.parse_args()

assert not OUT.exists() and not IDENTITY.exists()
progress, init, audit, sources, samples = [read(RUN/name) for name in
    ('progress.json', 'initialization.json', 'first_update_audit.json', 'source_hashes.json',
     'randomization_study.initial_samples.json')]
pid = progress['pid']
stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
task = read(RUN/'task_contract.json')
old_init, old_audit, old_samples = [read(FAILED/name) for name in
    ('initialization.json', 'first_update_audit.json', 'randomization_study.initial_samples.json')]
nominal = len(samples['samples']['enabled'])-sum(samples['samples']['enabled'])
sys.path.insert(0, str(ROOT/'src'))
import torch
from wheeled_algo.scratch_standing import tensor_hash
from wheeled_tasks.chassis.randomization_study import scalar_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

torch.set_num_threads(2)
initial = torch.load(RUN/'model_initial.pt', map_location='cpu', weights_only=False)
first = torch.load(RUN/'model_0.pt', map_location='cpu', weights_only=False)
events = EventAccumulator(str(RUN), size_guidance={'scalars':0}).Reload()
expected = {'DomainStudyConfig/'+key:value for key,value in
    scalar_metrics(read(RUN/'randomization_study.resolved.json')).items()}
expected.update({'DomainAdmissionConfig/'+key:value for key,value in
    dict(warmup_updates=192, ramp_updates=128, total_updates=1536, full_population_at_completed_update=320).items()})
expected.update({'FreshTasksConfig/'+key:value for key,value in
    dict(foundation_updates=192, full_tasks_at=768, independent_training_seed=271701).items()})
groups = {tag:events.Scalars(tag)[0].value for tag in events.Tags()['scalars'] if tag.startswith('DomainAdmission/')}
first_extrema = {tag.split('/',1)[1]:events.Scalars(tag)[0].value
    for tag in events.Tags()['scalars'] if tag.startswith('ScratchStandingExtrema/')}
counts = {role:int(first[role+'_state_dict']['obs_normalizer.count']) for role in ('actor','critic')}
witness = {tag.split('/',1)[1]:events.Scalars(tag)[0].value for tag in events.Tags()['scalars'] if tag.startswith('PhysicalPeakWitness/')}
assert len(witness)==11 and all(math.isfinite(v) for v in witness.values())

checks = dict(
    exact_live_training_process=stat[0] != 'Z' and int(stat[19]) == progress['start_ticks'],
    actual_GPU_process=str(pid) in subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).splitlines(),
    actual_native_updates=progress['status']=='training' and progress['completed_updates']>=2,
    no_checkpoint_or_teacher=progress['checkpoint_reads']==[] and not init['checkpoint_loaded'] and not init['teacher_loaded'],
    actual_initial_empty_Adam=not initial['optimizer_state_dict']['state'],
    zero_initial_clocks=init['actor_updates']==init['critic_updates']==init['optimizer_states']==initial['iter']==0,
    actual_initial_tensor_hashes=all(tensor_hash(initial[role+'_state_dict'])==init[role+'_tensor_sha256'] for role in ('actor','critic')),
    deterministic_fresh_initialization_matches017=all(init[key]==old_init[key] for key in ('actor_tensor_sha256','critic_tensor_sha256')),
    same_actual17_domain_arrays=samples['samples']==old_samples['samples'] and len(samples['samples'])==17,
    same_actual_ground_and_rigid_summary=samples['contact_material']==old_samples['contact_material'] and samples['rigid_summary']==old_samples['rigid_summary'],
    source_count274=len(sources)==274,
    live_and_archived_source_bytes=all(sha(ROOT/name)==value==sha(RUN/'source_snapshot'/name) for name,value in sources.items()),
    committed_source=progress['committed_source']['all_source_bytes_match_commit'],
    actual_first12_Adam_calls=audit['committed_optimizer_calls']==audit['optimizer_calls']==12,
    actual_actor10_critic8_changed=audit['actor_changed_parameter_tensors']==10 and audit['critic_changed_parameter_tensors']==8,
    finite_first_update=audit['all_recorded_values_finite'],
    first_update_checkpoint_clock=first['iter']==first['infos']['completed_updates']==1,
    only_admitted_samples_normalized=all(value==nominal*128 for value in counts.values()),
    nominal_only_admission=groups['DomainAdmission/admitted_nominal_lanes']==nominal
        and groups['DomainAdmission/admitted_randomized_lanes']==0.
        and groups['DomainAdmission/completed_actor_updates']==0.,
    separate_group_diagnostics=len(groups)==19 and all(math.isfinite(value) for value in groups.values()),
    native_compacted_minibatch=progress['valid_rollout']['valid_transitions']==nominal*128
        and progress['valid_rollout']['minimum_mini_batch_population']==nominal*128//4,
    actual_RSL_config_values=all(any(point.step==0 and math.isclose(point.value,value,rel_tol=2e-6,abs_tol=2e-8)
        for point in events.Scalars(tag)) for tag,value in expected.items()),
    actual_full_rollout_extrema_logged=first_extrema['policy_decisions']==128.
        and first_extrema['observed_leg_motor_samples']==128*5*256*4
        and math.isfinite(first_extrema['peak_leg_speed_rad_s']),
    repeated_extrema_flush_succeeded=len(events.Scalars('ScratchStandingExtrema/peak_leg_speed_rad_s'))>=2,
    tasks_initially_standing=events.Scalars('FreshTasks/standing/active_lanes')[0].value==256.,
    actual1536_plan=progress['requested_updates']==1536 and task['curriculum']['foundation_updates']==192
        and task['curriculum']['full_tasks_at']==768,
    declared_case_scoped_failure=task['settings']['fresh_standing_failure_training']==progress['fresh_standing_failure_training']
        ==initial['infos']['settings']['fresh_standing_failure_training']==first['infos']['settings']['fresh_standing_failure_training']
        and task['standing_failure_resets'] is True and task['standing_failure_scope']=='pure-standing-upright-births-only',
    actual_failure_RSL_tags=all(len(events.Scalars('FreshStandingFailure/'+metric+'_events'))>=2
        and all(math.isfinite(row.value) for row in events.Scalars('FreshStandingFailure/'+metric+'_events'))
        for metric in ('episode','tilt','height')),
    actual_new_objective_declared=task['settings']['fresh_training_objective']==progress['fresh_training_objective']
        ==initial['infos']['settings']['fresh_training_objective']==first['infos']['settings']['fresh_training_objective'],
    actual_new_objective_RSL_values=all(len(events.Scalars('FreshObjective/'+tag))>=2 and all(math.isfinite(row.value) for row in events.Scalars('FreshObjective/'+tag))
        for tag in ('pure_standing_lanes','active_lanes','attitude_paid_cost_per_policy_step','velocity_paid_cost_per_policy_step','active_speed_paid_cost_per_policy_step','attitude_density_per_second','velocity_density_per_second','active_speed_density_per_second')),
    actual_first_rollout_same_sample_witness=(witness['speed_rad_s']==first_extrema['peak_leg_speed_rad_s']
        and abs(witness['signed_speed_rad_s'])==witness['speed_rad_s']
        and math.isclose(witness['motor_power_w'],witness['signed_speed_rad_s']*witness['applied_torque_nm'],rel_tol=2e-6,abs_tol=2e-6)
        and witness['phase_code'] in (0.,1.,2.,3.)
        and all(len(events.Scalars('PhysicalPeakWitness/'+tag))>=2 for tag in witness)),
    native006_proof=progress['native_runtime_verification']['sha256']==sha(BASE/REL/'fresh_unified_native006/receipt.json'),
)
assert all(checks.values()), checks
stamp = datetime.now(timezone.utc).isoformat()
result = dict(schema='fresh-unified-first-PPO-verified-v1', observed_utc=stamp, run=str(RUN), pid=pid,
    start_ticks=progress['start_ticks'], checks=checks, all_checks_pass=True,
    observed_updates=progress['completed_updates'], requested_updates=1536, source_count=274,
    source_commit=progress['committed_source']['commit'], RSL_config_values_read_back=len(expected),
    nominal_lanes=nominal, randomized_lanes=256-nominal, first_valid_transitions=nominal*128,
    first_minibatch_population=nominal*128//4, first_checkpoint_normalizer_counts=counts,
    first_rollout_extrema=first_extrema, first_rollout_peak_witness=witness, initial_tensor_hashes={role:init[role+'_tensor_sha256'] for role in ('actor','critic')},
    previous_fully_completed_training_updates=4225, previous_failed_consumed_updates=1, previous_stopped_consumed_updates=842,
    previous_actual_consumed_updates=5068, no_prior_weight_assignment=True,
    interpretation='First-update/implementation evidence only; full13 behavior, robustness and hardware remain unqualified.',
    new_PPO_by_reader=0, qualified=False, hardware_ready=False, producer_sha256=sha(Path(__file__)))
identity = dict(schema='fresh-unified-launch-identity-v1', observed_utc=stamp, pid=pid,
    start_ticks=progress['start_ticks'], command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0'),
    source_checkout=str(ROOT), run=str(RUN), commit=result['source_commit'],
    source_archive_sha256=progress['committed_source']['archive_sha256'], source_count=274,
    requested_updates=1536, previous_actual_consumed_updates=5068, exec_session_id=args.exec_session_id,
    initialization=init, first_verification=str(OUT), formal_long_training=False, hardware_ready=False)
write(OUT,result)
write(IDENTITY,identity)
for p in (OUT,IDENTITY):
    (ROOT/REL/p.name).write_bytes(p.read_bytes())
(BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path = BASE/REL/'scratch_campaign001.json'
raw = ledger_path.read_bytes()
ledger = json.loads(raw)
assert ledger['active_pid'] is None and ledger['total_actual_consumed_training_updates']==5068
ledger.update(status='training_unqualified', active_run=str(RUN), active_pid=pid,
    active_start_ticks=progress['start_ticks'], active_launch_identity=str(IDENTITY),
    latest_training_launch=str(IDENTITY), latest_first_verification=str(OUT), requested_updates=1536,
    prior_closed_fresh_updates=4225, prior_consumed_training_updates=5068,
    completed_updates_observed=progress['completed_updates'], active_completed_updates_observed=progress['completed_updates'],
    total_actual_consumed_training_updates=5068+progress['completed_updates'],
    total_from_zero_completed_observed=4225+progress['completed_updates'],
    do_not_modify_live_src=True, do_not_modify_training_entrypoint=True, do_not_modify_domain_admission_helper=True,
    source_snapshot_count=274, active_evaluation=None, active_evaluation_pid=None,
    active_evaluation_start_ticks=None, latest_observed_utc=stamp,
    latest_observation='Fresh018 first real PPO, source, normalization and repeated extrema flush verified. No behavior qualification.',
    next_action='Observe immutable learner and case-scoped failure frequency;  after exact exit and seal run cold same-PT full13 nominal/requested/holdout suite.')
assert ledger_path.read_bytes()==raw
tmp=ledger_path.with_suffix('.fresh018.tmp');write(tmp,ledger);tmp.replace(ledger_path)
print(json.dumps(dict(checks=len(checks),all_pass=True,actual_updates=progress['completed_updates'],
    nominal_lanes=nominal,first_minibatch=nominal*128//4,first_extrema=first_extrema)))
