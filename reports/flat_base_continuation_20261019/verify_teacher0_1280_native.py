"""Verify the bounded teacher-zero continuation, without optimizer or env steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--first-only', action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
trial = RUN / 'pilot025'
required = (1265,) if args.first_only else (1265, 1280)
output = HERE / ('teacher0_native1265.json' if args.first_only else 'teacher0_native1280.json')
if output.exists():
    raise SystemExit('Preserve the completed native continuation proof')
if not all(p.is_file() for p in [trial / 'initial_state.pt', *(trial / f'model_{n:06d}.pt' for n in required)]):
    raise SystemExit('Wait for the requested actual checkpoints')
parent_path = RUN / 'pilot024/model_001264.pt'
parent = load(parent_path)
initial = load(trial / 'initial_state.pt')
before, info = parent['infos'], initial['infos']
revision_path = HERE / 'teacher0_plan_revision020.json'
revision = read(revision_path)


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


checks = {f'initial_{k}_exact_parent': exact(initial[k], parent[k]) for k in
          ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'torch_rng_state', 'cuda_rng_state')}
checks['all_six_native_restore_checks'] = (len(info['full_restore_verification']) == 6
                                         and all(info['full_restore_verification'].values()))
checks['exact1264_parent'] = (sha(parent_path) == info['parent_checkpoint_sha256']
    == revision['parent_checkpoint_sha256'] == 'a2da6311fb1d5bfcc75892f4fa1ac43ec8f8866edafc0f68ef28436ec1606b48')
changed_sources = sorted(n for n in set(before['source_sha256']) | set(info['source_sha256'])
                         if before['source_sha256'].get(n) != info['source_sha256'].get(n))
entry = 'scripts/train_flat_base.py'
checks['only_reviewed_cause_logger_entry_changed'] = (changed_sources == [entry]
    and before['source_sha256'] == revision['old_source_sha256']
    and info['source_sha256'] == revision['new_source_sha256']
    and info['source_sha256'][entry] == 'fb09bf77cb0bba0df968695aa456304cc84694866a7fde158da83607c6331e3c'
    and info.get('plan_revision_sha256') == sha(revision_path))
old_plan, new_plan = (read(ROOT / revision[k]) for k in ('old_plan_file', 'new_plan_file'))
changed_plan = sorted(k for k in old_plan.keys() | new_plan.keys() if old_plan.get(k) != new_plan.get(k))
old_anchor, new_anchor = old_plan['teacher_anchor'], new_plan['teacher_anchor']
checks['only_teacher_coefficient_and_zero_extra_warmup_changed'] = (
    changed_plan == revision['changed_plan_fields'] == info['changed_plan_fields']
    == ['additional_critic_warmup_updates', 'teacher_anchor']
    and {k: v for k, v in old_anchor.items() if k != 'coefficient'}
        == {k: v for k, v in new_anchor.items() if k != 'coefficient'}
    and old_anchor['coefficient'] == .02 and new_anchor['coefficient'] == 0
    and new_plan['additional_critic_warmup_updates'] == 0
    and sha(ROOT / revision['new_plan_file']) == info['plan_sha256'] == revision['new_plan_sha256']
    and sha(ROOT / revision['old_plan_file']) == before['plan_sha256'] == revision['old_plan_sha256'])
runtime = info['teacher_anchor_runtime']
checks['actual_teacher0_all_other_settings_exact'] = (runtime['coefficient'] == 0
    and {k: v for k, v in runtime.items() if k != 'coefficient'}
        == {k: v for k, v in before['teacher_anchor_runtime'].items() if k != 'coefficient'})
checks['same4096_population_and_all_clocks'] = (info['num_envs'] == before['num_envs'] == 4096
    and info['control_math_sha256'] == before['control_math_sha256']
    and info['training_command_reference'] == before['training_command_reference']
    and all(info['startup']['runtime_preflight']['observed'][n] == value for n, value in
            dict(physics_dt=.001, policy_dt=.02, pc_control_dt=.001, decimation=20).items()))
checks['no_warmup_or_Adam_rate_change'] = (info['critic_warmup_remaining'] == 0
    and info['optimizer_after_declared_rate_exact'] and not info['optimizer_rate_contract']['declared_actor_rate_change'])
checks['same_height_metrics_and_new_native_causes'] = (
    info['height_command_metrics_contract'] == before['height_command_metrics_contract']
    and info['termination_metrics_contract']['schema'] == 'flat-native-termination-counts-v1'
    and info['termination_metrics_contract']['windows'] == ['whole', 'first1s', 'after1s'])
checks['all1264_history_and_exposure_retained'] = all(info[k] == before[k] for k in
    ('successful_updates_total', 'actor_updates_total', 'active_branch_actor_updates',
     'feedback_branch_actor_updates', 'post_split_actor_updates', 'inherited_shared_feedback_updates',
     'consumed_optimizer_updates_total', 'training_transitions_total'))
model = mean_from_checkpoint(initial, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx')
names = ['distribution.std_param'] + ['mlp.' + n for n, _ in model.named_parameters()]
ids = dict(zip(names, initial['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
prefixes = ('mlp.feedback.', 'mlp.fast_feedback.')
learned = [n for n in names if n.startswith(prefixes)]
frozen = [n for n in names if n not in learned]
base_slots = initial['optimizer_state_dict']['state']
milestones = {}
for count in required:
    path = trial / f'model_{count:06d}.pt'
    current = load(path)
    row, slots, delta = current['infos'], current['optimizer_state_dict']['state'], count - 1264
    proof = dict(frozen_actor_std_buffers_exact=all(torch.equal(v, current['actor_state_dict'][n])
        for n, v in initial['actor_state_dict'].items() if not n.startswith(prefixes)),
        all_frozen_Adam_exact=all(exact(base_slots.get(ids[n]), slots.get(ids[n])) for n in frozen),
        all_eight_feedback_tensors_learned=len(learned) == 8 and all(
            not torch.equal(initial['actor_state_dict'][n], current['actor_state_dict'][n]) for n in learned),
        exact2252_scoped_parameters=row['actor_trainability']['trainable_parameters'] == 2252
            and set(row['actor_trainability']['trainable_names']) == set(learned),
        critic_learned=any(not torch.equal(v, current['critic_state_dict'][n]) for n, v in initial['critic_state_dict'].items()),
        twelve_Adam_steps_per_update=all(float(slots[i]['step']) == float(base_slots[i]['step']) + 12 * delta
            for i in [ids[n] for n in learned] + initial['optimizer_state_dict']['param_groups'][1]['params']),
        all_lifetime_counters_retained=(row['successful_updates_total'], row['actor_updates_total'],
            row['active_branch_actor_updates'], row['feedback_branch_actor_updates'], row['post_split_actor_updates'],
            row['inherited_shared_feedback_updates'], row['consumed_optimizer_updates_total'], row['critic_warmup_remaining'])
            == (count, 1084 + delta, 44, 706 + delta, 402 + delta, 304, count + 1, 0),
        exact_cumulative_frame_exposure=row['training_transitions_total'] == 67633152 + delta * 4096 * 96,
        unchanged_actor_critic_rates=[g['lr'] for g in current['optimizer_state_dict']['param_groups']] == [3e-4, 5e-4],
        all_native_losses_finite=all(math.isfinite(v) for v in row['last_losses'].values()),
        teacher0_and_cause_contract_unchanged=row['teacher_anchor_runtime'] == runtime
            and row['termination_metrics_contract'] == info['termination_metrics_contract'])
    checks.update({str(count) + '_' + k: v for k, v in proof.items()})
    milestones[str(count)] = dict(checkpoint_sha256=sha(path), checks=proof)

cause_rows = 0
if not args.first_only:
    events = EventAccumulator(str(trial), size_guidance={'scalars': 0})
    events.Reload()
    tags = set(events.Tags()['scalars'])
    cache = {}
    steps = list(range(1265, 1281))
    def scalar(tag):
        if tag not in cache:
            cache[tag] = {p.step: p.value for p in events.Scalars(tag)} if tag in tags else {}
        return cache[tag]
    cause_checks = []
    unique = {n: 0 for n in steps}
    bands = info['termination_metrics_contract']['height_bands']
    for group, population in info['training_command_reference']['population'].items():
        views = [group] + ([group + '/rmcs_direct'] if population['direct'] else [])
        views += [group + '/legacy_ramp'] if population['ramped'] else []
        for view in views:
            for band in bands:
                for flag in info['termination_metrics_contract']['flags']:
                    values = [scalar(f'FlatTermination/{w}/{view}/{band}/{flag}_events')
                              for w in ('whole', 'first1s', 'after1s')]
                    cause_checks.append(all(sorted(v) == steps for v in values))
                    for step in steps:
                        total, birth, later = (v.get(step, float('nan')) for v in values)
                        cause_checks.append(all(math.isfinite(v) and v >= 0 and v.is_integer()
                                                for v in (total, birth, later)) and total == birth + later)
                        if view == group and flag == 'terminated':
                            unique[step] += total
                        if flag == 'terminated':
                            for namespace, count in (('FlatBirth', birth), ('FlatBehavior', later)):
                                prefix = f'{namespace}/{view}/{band}'
                                frames = scalar(prefix + '/sample_frames').get(step, float('nan'))
                                fraction = scalar(prefix + '/terminated_frame_fraction').get(step, 0.)
                                cause_checks.append(math.isclose(frames * fraction, count, abs_tol=1e-5))
                        cause_rows += 1
    failure = scalar('/reward/failure_event')
    checks['all_native_cause_counts_integer_and_window_partition_exact'] = bool(cause_checks) and all(cause_checks)
    checks['unique_terminated_matches_native_failure_payment'] = (
        sorted(failure) == list(range(1264, 1280)) and all(
            math.isclose(unique[n], -failure[n - 1] * 4096 * 96 / 200, abs_tol=1e-5) for n in steps))
    checks['all16_native_teacher_diagnostics_logged'] = sorted(scalar('Loss/teacher')) == list(range(1264, 1280))

receipt = dict(schema='flat-teacher0-1280-native-verification-v1', passed=all(checks.values()),
    checks=checks, milestones=milestones, first_only=args.first_only,
    native_cause_rows_checked=cause_rows, behavior_qualified=False, new_optimizer_updates=0)
if not receipt['passed']:
    print(json.dumps(checks, indent=2))
    raise SystemExit('Native teacher-zero continuation proof failed; preserve the negative output')
output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), milestones=list(milestones), cause_rows=cause_rows)))
