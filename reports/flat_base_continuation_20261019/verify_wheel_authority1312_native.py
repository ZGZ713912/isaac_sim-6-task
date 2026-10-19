"""Audit the actual bounded zero wheel-head continuation without changing its learning state."""
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
from wheeled_algo.wheel_authority import KIND

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--first-only', action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
trial = RUN / 'pilot027'
required = (1297,) if args.first_only else (1297, 1312)
out = HERE / ('wheel_authority_native1297.json' if args.first_only else 'wheel_authority_native1312.json')
if out.exists():
    raise SystemExit('Preserve completed audit')
paths = [trial / 'initial_state.pt', *(trial / f'model_{n:06d}.pt' for n in required)]
if not all(p.is_file() for p in paths):
    raise SystemExit('Wait for actual checkpoints')
parent_path = RUN / 'pilot026/model_001296.pt'
parent, initial = load(parent_path), load(paths[0])
before, info = parent['infos'], initial['infos']
revision_path = HERE / 'wheel_authority_plan_revision022.json'
revision = read(revision_path)
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
old_mean, mean = mean_from_checkpoint(parent, source), mean_from_checkpoint(initial, source)


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


names = ['distribution.std_param'] + ['mlp.' + n for n, _ in mean.named_parameters()]
old_names = ['distribution.std_param'] + ['mlp.' + n for n, _ in old_mean.named_parameters()]
ids = dict(zip(names, initial['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
old_ids = dict(zip(old_names, parent['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
base_slots = initial['optimizer_state_dict']['state']
old_slots = parent['optimizer_state_dict']['state']
prefixes = ('mlp.wheel_authority.',)
learned = [n for n in names if n.startswith(prefixes)]
frozen = [n for n in names if n not in learned]
static_names = [n for n in names if n.startswith('mlp.wheel_authority.')]
old_plan, plan = (read(ROOT / revision[k]) for k in ('old_plan_file', 'new_plan_file'))
changed = sorted(k for k in old_plan.keys() | plan.keys() if old_plan.get(k) != plan.get(k))
anchor = dict(info['teacher_anchor_runtime'])
anchor['coefficient'] = before['teacher_anchor_runtime']['coefficient']
counters = ('successful_updates_total', 'actor_updates_total', 'active_branch_actor_updates',
            'feedback_branch_actor_updates', 'post_split_actor_updates', 'inherited_shared_feedback_updates',
            'consumed_optimizer_updates_total', 'training_transitions_total')
checks = dict(
    exact_parent=sha(parent_path) == info['parent_checkpoint_sha256'] == revision['parent_checkpoint_sha256'],
    existing_actor_weights_and_buffers_exact=all(torch.equal(v, initial['actor_state_dict'][n])
        for n, v in parent['actor_state_dict'].items()),
    only_four_new_head_tensors_and_bound_added=set(initial['actor_state_dict']) - set(parent['actor_state_dict']) == set(static_names) | {'mlp.authority_bound'}
        and len(static_names) == 4,
    frozen_reference_exact=sha(ROOT / plan['static_feedback_recall']['checkpoint']) == plan['static_feedback_recall']['sha256']
        and info['static_feedback_recall'] == plan['static_feedback_recall'],
    existing_actor_Adam_exact=all(exact(old_slots.get(old_ids[n]), base_slots.get(ids[n])) for n in old_names),
    critic_Adam_remapped_exact=all(exact(old_slots.get(a), base_slots.get(b)) for a, b in zip(
        parent['optimizer_state_dict']['param_groups'][1]['params'],
        initial['optimizer_state_dict']['param_groups'][1]['params'], strict=True)),
    static_has_no_Adam=all(ids[n] not in base_slots for n in static_names),
    critic_and_saved_sampling_RNG_exact=all(exact(parent[k], initial[k])
        for k in ('critic_state_dict', 'torch_rng_state', 'cuda_rng_state')),
    all_six_native_restore_checks=len(info['full_restore_verification']) == 6
        and all(info['full_restore_verification'].values()),
    old_lifetime_counters_exact=all(info[k] == before[k] for k in counters),
    exact_declared_plan_and_sources=info['plan_revision_sha256'] == sha(revision_path)
        and before['source_sha256'] == revision['old_source_sha256']
        and info['source_sha256'] == revision['new_source_sha256']
        and info['plan_sha256'] == sha(ROOT / revision['new_plan_file']) == revision['new_plan_sha256'],
    only_actor_kind_declared_plan_change=changed == revision['changed_plan_fields'] == info['changed_plan_fields']
        == ['actor_adaptation'],
    teacher_stays_zero=info['teacher_anchor_runtime'] == before['teacher_anchor_runtime']
        and info['teacher_anchor_runtime']['coefficient'] == 0.,
    explicit_zero_identity_extension=info['actor_adaptation'] == KIND
        and info['wheel_authority_extension']['initial_mean_is_identity'] is True,
    same_population_command_reference_and_clocks=info['num_envs'] == before['num_envs'] == 4096
        and info['control_math_sha256'] == before['control_math_sha256']
        and info['training_command_reference'] == before['training_command_reference']
        and all(info['startup']['runtime_preflight']['observed'][n] == v for n, v in
                dict(physics_dt=.001, policy_dt=.02, pc_control_dt=.001, decimation=20).items()),
    no_extra_warmup=info['critic_warmup_remaining'] == 0 and plan['additional_critic_warmup_updates'] == 0,
    unchanged_height_and_termination_contracts=all(info[k] == before[k]
        for k in ('height_command_metrics_contract', 'termination_metrics_contract')))
milestones = {}
for count in required:
    path = trial / f'model_{count:06d}.pt'
    state = load(path)
    row, slots, delta = state['infos'], state['optimizer_state_dict']['state'], count - 1296
    proof = dict(
        frozen_original_command_static_std_and_buffers_exact=all(torch.equal(v, state['actor_state_dict'][n])
            for n, v in initial['actor_state_dict'].items() if not n.startswith(prefixes)),
        frozen_Adam_exact=all(exact(base_slots.get(ids[n]), slots.get(ids[n])) for n in frozen),
        all_four_new_head_tensors_learned=len(learned) == 4
            and all(not torch.equal(initial['actor_state_dict'][n], state['actor_state_dict'][n]) for n in learned),
        exactly994_trainable_parameters=row['actor_trainability']['trainable_parameters'] == 994
            and set(row['actor_trainability']['trainable_names']) == set(learned),
        critic_learned=any(not torch.equal(v, state['critic_state_dict'][n]) for n, v in initial['critic_state_dict'].items()),
        twelve_Adam_steps_per_update=all(float(slots[i]['step']) == float(base_slots.get(i, {'step':0})['step']) + 12 * delta
            for i in [ids[n] for n in learned] + initial['optimizer_state_dict']['param_groups'][1]['params']),
        cumulative_counters_exact=all(row[k] == before[k] + (delta if k in (
            'successful_updates_total', 'actor_updates_total', 'feedback_branch_actor_updates',
            'post_split_actor_updates', 'consumed_optimizer_updates_total') else
            delta * 4096 * 96 if k == 'training_transitions_total' else 0) for k in counters),
        unchanged_rates_and_finite_losses=[g['lr'] for g in state['optimizer_state_dict']['param_groups']] == [3e-4, 5e-4]
            and all(math.isfinite(v) for v in row['last_losses'].values()),
        unchanged_trial_contracts=row['teacher_anchor_runtime'] == info['teacher_anchor_runtime']
            and row['static_feedback_recall'] == info['static_feedback_recall']
            and row['termination_metrics_contract'] == info['termination_metrics_contract'])
    checks.update({str(count) + '_' + k: v for k, v in proof.items()})
    milestones[str(count)] = dict(checkpoint_sha256=sha(path), checks=proof)
if not args.first_only:
    events = EventAccumulator(str(trial), size_guidance={'scalars': 0})
    events.Reload()
    expected = list(range(1297, 1313))
    checks['sixteen_wheel_head_updates_logged_in_RSL_events'] = all(
        [p.step for p in events.Scalars(tag)] == expected
        for tag in ('FlatTransfer/actor_updates_total', 'FlatTransfer/feedback_branch_actor_updates',
                    'FlatTransfer/training_transitions_total'))
if not all(checks.values()):
    print(json.dumps(checks, indent=2))
    raise SystemExit('Native state restoration or learning audit failed')
out.write_text(json.dumps(dict(schema='flat-wheel-authority-native-audit-v1', passed=True,
    checks=checks, parent_checkpoint_sha256=sha(parent_path), initial_state_sha256=sha(paths[0]),
    revision_sha256=sha(revision_path), milestones=milestones, new_optimizer_updates=0,
    scope='Audit only; behavior remains subject to fixed full-horizon native replays.'), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), milestones=milestones), indent=2))
