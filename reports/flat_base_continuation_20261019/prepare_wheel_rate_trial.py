"""Prepare one32-update actor-rate trial after the previous full trial is sealed."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.flat_optimizer_rates import apply_flat_optimizer_rates
from wheeled_tasks.chassis.flat_base_continuation import read_plan

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
write = lambda p, d: p.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
directory = HERE / 'wheel_rate_candidate001'
if directory.exists():
    raise SystemExit('Preserve the existing rate trial candidate')
queue_path = RUN / 'wheel_from1280_queue001.json'
seal_path = HERE / 'wheel_from1280_completion001.json'
clock_path = HERE / 'release_clock_adoption024.json'
queue, seal, clock = (read(p) for p in (queue_path, seal_path, clock_path))
parent_path = RUN / 'pilot027_from1280/model_001296.pt'
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
info = parent['infos']
checks = dict(
    previous_trial_owners_exited=queue['status'] == 'complete'
        and all(not Path('/proc/' + str(queue[k])).exists() for k in ('pid', 'worker_pid')),
    previous_trial_sealed=seal['passed'] and all(seal['checks'].values())
        and seal['queue_sha256'] == sha(queue_path) and seal['checkpoint_sha256'] == sha(parent_path),
    no_new_regression_from_original1280=all(
        cohort['comparisons']['parent1280']['no_lost_checks']
        and not cohort['comparisons']['parent1280']['relative_regressions']
        for cohort in seal['cohorts'].values()),
    actual_clock_reader_adopted=clock['passed'] and all(clock['checks'].values())
        and clock['new_export_sha256'] == sha(ROOT / 'scripts/export_flat_base.py'),
    existing_training_sources_exact=all(sha(ROOT / n) == value for n, value in info['source_sha256'].items()),
    only_existing994_wheel_parameters_train=info['actor_trainability']['trainable_parameters'] == 994
        and all(n.startswith('mlp.wheel_authority.') for n in info['actor_trainability']['trainable_names']),
    previous_branch_completed=info['successful_updates_total'] == 1296 and info['critic_warmup_remaining'] == 0,
)
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
old_path = ROOT / info['plan_file']
old = read_plan(ROOT, info['plan_file'])
if sha(old_path) != info['plan_sha256']:
    raise SystemExit('The parent plan is not exact')
plan = deepcopy(old)
plan['learning_rate'] = .001
changed = sorted(k for k in set(old) | set(plan) if old.get(k) != plan.get(k))
opt = SimpleNamespace(param_groups=deepcopy(parent['optimizer_state_dict']['param_groups']))
expected = deepcopy(opt.param_groups)
expected[0]['lr'] = .001
rate = apply_flat_optimizer_rates(opt, plan['learning_rate'], plan['critic_learning_rate'],
    parent_rates=(old['learning_rate'], old['critic_learning_rate']))
checks.update(only_learning_rate_plan_field_changes=changed == ['learning_rate'],
    only_actor_group_rate_changes=opt.param_groups == expected,
    exact_declared_rates=rate['before'] == dict(actor=.0003, critic=.0005)
        and rate['after'] == dict(actor=.001, critic=.0005),
    clocks_population_and_reward_retained=plan['policy_dt'] == .02 and plan['physics_dt'] == .001
        and plan['default_num_envs'] == 4096 and plan['additional_critic_warmup_updates'] == 0,
)
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
directory.mkdir()
plan_path = directory / 'plan.json'
write(plan_path, plan)
if read_plan(ROOT, str(plan_path.relative_to(ROOT))) != plan:
    raise SystemExit('New plan failed the production reader')
revision = dict(schema='flat-training-plan-revision-v1',
    parent_checkpoint_sha256=sha(parent_path), old_plan_file=info['plan_file'],
    old_plan_sha256=sha(old_path), new_plan_sha256=sha(plan_path),
    old_source_sha256=info['source_sha256'], new_source_sha256=info['source_sha256'], changed_plan_fields=changed)
revision_path = directory / 'revision.json'
write(revision_path, revision)
write(directory / 'cpu001.json', dict(schema='flat-wheel-rate-candidate-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    parent_checkpoint=str(parent_path.relative_to(ROOT)), parent_checkpoint_sha256=sha(parent_path),
    previous_trial_seal_sha256=sha(seal_path), release_clock_adoption_sha256=sha(clock_path),
    plan_file=str(plan_path.relative_to(ROOT)), plan_sha256=sha(plan_path), revision_sha256=sha(revision_path),
    source_sha256=info['source_sha256'], rate_contract=rate, preparer_sha256=sha(Path(__file__)),
    planned_optimizer_updates=32, planned_actor_updates=32, additional_critic_warmup=0,
    expected_final_successful_updates=1328, expected_final_lineage_consumption=1329,
    origin1280_and_both_trials_plus_this_trial_consumption_at_exit=1345,
    required_replays=['basic', '23cm full22', '42cm full22'], heldout_seed_used=False,
    new_optimizer_updates=0, new_native_steps=0, subsequent_training_queued=False,
    behavior_qualified=False, hardware_motion_started=False,
    scope='CPU plan and isolated rate application only; the native first-step audit must prove full restoration and learning ownership. No behavior benefit follows from this preparation.'))
print(json.dumps(dict(passed=True, checks=len(checks), candidate=str(directory.relative_to(ROOT))), indent=2))
