"""Read the fresh curriculum's first real update; never load in its learner."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
RUN = BASE / 'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'
OLD = BASE / 'runs/scratch_standing_20261026/pilot012_isolated_sensor_requested001'
OUT = REPORT / 'domain_admission_first_verification001.json'
IDENTITY = REPORT / 'scratch_pilot013_launch_identity001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


assert not OUT.exists() and not IDENTITY.exists()
progress = read(RUN / 'progress.json')
pid = progress['pid']
start = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
init, old_init = read(RUN / 'initialization.json'), read(OLD / 'initialization.json')
audit, manifest = read(RUN / 'first_update_audit.json'), read(RUN / 'source_hashes.json')
sample, old_sample = read(RUN / 'randomization_study.initial_samples.json'), read(OLD / 'randomization_study.initial_samples.json')
task, old_task = read(RUN / 'task_contract.json')['settings'], read(OLD / 'task_contract.json')['settings']
args, old_args = progress['args'].copy(), read(OLD / 'progress.json')['args'].copy()
for values in (args, old_args):
    values.pop('output')
    values.pop('updates')
for key in ('domain_admission_warmup_updates', 'domain_admission_ramp_updates', 'domain_admission_seed'):
    args.pop(key)
checks = dict(
    exact_live_process=start == progress['start_ticks'],
    actual_native_update=progress['completed_updates'] >= 1,
    no_checkpoint_or_teacher=progress['checkpoint_reads'] == [] and init['checkpoint_loaded'] is False and init['teacher_loaded'] is False,
    initialization_clocks_zero=init['actor_updates'] == init['critic_updates'] == init['optimizer_states'] == 0,
    same_initial_actor_critic_as012=all(init[key] == old_init[key] for key in ('actor_tensor_sha256', 'critic_tensor_sha256')),
    source_count_266=len(manifest) == 266,
    frozen_and_live_sources_match=all(sha(ROOT / name) == value and sha(RUN / 'source_snapshot' / name) == value for name, value in manifest.items()),
    committed_source=progress['committed_source']['all_source_bytes_match_commit'],
    identical17_sensor_actuator_rigid_arrays=sample['samples'] == old_sample['samples'] and len(sample['samples']) == 17,
    identical_two_actual_ground_arrays=all(sample['contact_material'][key] == old_sample['contact_material'][key]
        for key in ('ground_static_friction_per_environment', 'ground_dynamic_friction_per_environment')),
    complete_material_and_rigid_summaries_match=sample['contact_material'] == old_sample['contact_material'] and sample['rigid_summary'] == old_sample['rigid_summary'],
    same_native_agent=read(RUN / 'agent_config.json') == read(OLD / 'agent_config.json'),
    only_admission_task_change=sorted(key for key in set(task) | set(old_task) if task.get(key) != old_task.get(key)) == ['domain_learning_admission', 'version'],
    same_CLI_except_schedule_and_budget=args == old_args,
    native_first12_Adam_calls=audit['committed_optimizer_calls'] == audit['optimizer_calls'] == 12,
    actual_actor_critic_changes=audit['actor_changed_parameter_tensors'] == 9 and audit['critic_changed_parameter_tensors'] == 8,
    finite_first_update=audit['all_recorded_values_finite'],
    corrected_sensor_clock_retained=progress['domain_sensor_clock_version'] == old_task['domain_sensor_clock_version'],
    Gaussian_reward_retained=progress['precise_reward']['shape'] == 'gaussian',
    requested512_updates=progress['requested_updates'] == 512,
    actual_schedule=all(progress['domain_learning_admission'][key] == value for key, value in
        dict(warmup_updates=192, ramp_updates=128, priority_seed=261501, full_population_at_completed_update=320, total_updates=512).items()),
    actual_full_width_physical_domains=progress['domain_learning_admission']['widths_changed'] is False
        and task['randomization_study'] == old_task['randomization_study'],
)
sys.path.insert(0, str(ROOT / 'src'))
import torch
from wheeled_algo.scratch_standing import tensor_hash
from wheeled_tasks.chassis.randomization_study import scalar_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

torch.set_num_threads(2)
initial = torch.load(RUN / 'model_initial.pt', map_location='cpu', weights_only=False)
checks['actual_initial_Adam_empty'] = not initial['optimizer_state_dict']['state']
checks['actual_initial_tensor_hashes_match'] = all(tensor_hash(initial[role + '_state_dict']) == init[role + '_tensor_sha256'] for role in ('actor', 'critic'))
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
checks['actual_GPU_process_present'] = str(pid) in gpu.splitlines()
events = EventAccumulator(str(RUN), size_guidance={'scalars': 0}).Reload()
expected = {'DomainStudyConfig/' + key: value for key, value in scalar_metrics(read(RUN / 'randomization_study.resolved.json')).items()}
expected.update({'DomainSensorsConfig/selective_reset_clock_enabled': 1., 'PrecisionStandingConfig/joint_long_tail_enabled': 0.})
expected.update({'DomainAdmissionConfig/' + key: value for key, value in
    dict(warmup_updates=192, ramp_updates=128, total_updates=512, full_population_at_completed_update=320).items()})
checks['RSL_config_scalars_match'] = all(any(point.step == 0 and math.isclose(point.value, value, rel_tol=2e-6, abs_tol=2e-8)
    for point in events.Scalars(tag)) for tag, value in expected.items())
first_groups = {tag: events.Scalars(tag)[0].value for tag in events.Tags()['scalars'] if tag.startswith('DomainAdmission/')}
nominal = len(sample['samples']['enabled']) - sum(sample['samples']['enabled'])
first_checkpoint = torch.load(RUN / 'model_0.pt', map_location='cpu', weights_only=False)
normalizer_counts = {role: int(first_checkpoint[role + '_state_dict']['obs_normalizer.count']) for role in ('actor', 'critic')}
checks['actual_first_checkpoint_only_normalizes_admitted_samples'] = (
    first_checkpoint['iter'] == first_checkpoint['infos']['completed_updates'] == 1
    and all(value == nominal * 128 for value in normalizer_counts.values()))
checks['actual_first_rollout_admission'] = (
    first_groups['DomainAdmission/admitted_nominal_lanes'] == nominal
    and first_groups['DomainAdmission/admitted_randomized_lanes'] == 0.
    and first_groups['DomainAdmission/completed_actor_updates'] == 0.
    and math.isclose(first_groups['DomainAdmission/population_fraction'], nominal / 256))
checks['group_diagnostics_separate_and_finite'] = len(first_groups) == 19 and all(math.isfinite(value) for value in first_groups.values())
checks['actual_compacted_warmup_population'] = (
    progress['valid_rollout']['valid_transitions'] == nominal * 128
    and progress['valid_rollout']['minimum_mini_batch_population'] == nominal * 128 // 4)
assert all(checks.values()), checks
stamp = datetime.now(timezone.utc).isoformat()
result = dict(schema='domain-admission-first-fresh-native-PPO-verification-v1', observed_utc=stamp,
    run=str(RUN), pid=pid, start_ticks=start, completed_updates=progress['completed_updates'],
    requested_updates=512, checks=checks, all_checks_pass=True, source_count=266,
    source_commit=progress['committed_source']['commit'], actual_domain_arrays_compared=19,
    RSL_config_scalars_verified=len(expected), RSL_first_group_scalars=first_groups,
    actual_first_valid_population=nominal * 128, actual_warmup_minibatch_population=nominal * 128 // 4,
    actual_first_checkpoint_normalizer_counts=normalizer_counts,
    legacy_rollout_metric_interpretation='script_or_blend_transitions is the existing generic excluded-sample counter; in this warmup it counts randomized lanes excluded by admission, not a scripted controller.',
    actual_warmup_excluded_randomized_transitions=(256 - nominal) * 128,
    old_cumulative_closed_fresh_updates=3713, native_behavior_verified=False,
    long_training_started=False, ONNX_exported=False, qualified=False, hardware_ready=False,
    comparison='Same fresh initialization and all19 requested startup arrays as012; optional admission and512 budget are explicit. No old-policy training reads.',
    producer_sha256=sha(__file__))
write(OUT, result)
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
identity = dict(schema='scratch-from-zero-launch-identity-v2', observed_utc=stamp, pid=pid, start_ticks=start,
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0'), source_checkout=str(ROOT),
    run=str(RUN), commit=result['source_commit'], source_archive_sha256=progress['committed_source']['archive_sha256'],
    source_count=266, requested_updates=512, prior_closed_fresh_updates=3713,
    exec_session_id=1230,
    initialization=init, first_verification=str(OUT), scope='Gradual full-width domain admission, fresh middle-height foundation only', hardware_ready=False)
write(IDENTITY, identity)
ledger_path = REPORT / 'scratch_campaign001.json'
raw = ledger_path.read_bytes()
ledger = json.loads(raw)
assert ledger['active_pid'] is None and ledger['prior_closed_fresh_updates'] == 3713
ledger.update(status='training_unqualified', active_run=str(RUN), active_pid=pid, active_start_ticks=start,
    active_launch_identity=str(IDENTITY), latest_training_launch=str(IDENTITY), requested_updates=512,
    completed_updates_observed=progress['completed_updates'], active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed=3713 + progress['completed_updates'], total_from_zero_completed_observed=3713 + progress['completed_updates'],
    latest_first_verification=str(OUT), do_not_modify_live_src=True, do_not_modify_training_entrypoint=True,
    source_snapshot_count=266, do_not_modify_domain_admission_helper=True,
    active_evaluation=None, active_evaluation_pid=None, active_evaluation_start_ticks=None,
    do_not_modify_frozen_evaluator_sources=False,
    latest_observation='Actual fresh curriculum first native PPO and full-width draw checks passed; no policy behavior or robustness improvement is yet established.',
    next_action='Observe nominal/randomized RSL groups through warmup, ramp and full population; complete512 and cold held-out native tests.',
    observed_utc=stamp, last_observed_utc=stamp)
assert ledger_path.read_bytes() == raw
temporary = ledger_path.with_suffix('.admissionfirst001.tmp')
write(temporary, ledger)
temporary.replace(ledger_path)
print(json.dumps(dict(checks=len(checks), all_pass=True, completed_updates=progress['completed_updates'],
    actual_domain_arrays_compared=19, RSL_config_scalars_verified=len(expected),
    valid_rollout=progress['valid_rollout'], group_scalar_count=len(first_groups))))
