"""Read actual restored state and first/final learning for the isolated rate trial."""
import argparse
from copy import deepcopy
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
trial = RUN / 'pilot028_wheel_rate1328'
out = HERE / ('wheel_rate_native1297.json' if args.first_only else 'wheel_rate_native1328.json')
if out.exists():
    raise SystemExit('Preserve the completed actual-state audit')
candidate_path = HERE / 'wheel_rate_candidate001/cpu001.json'
candidate = read(candidate_path)
parent_path = ROOT / candidate['parent_checkpoint']
parent, initial = load(parent_path), load(trial / 'initial_state.pt')
before, info = parent['infos'], initial['infos']
plan_path = ROOT / candidate['plan_file']
plan = read(plan_path)
revision_path = HERE / 'wheel_rate_candidate001/revision.json'


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


mean = mean_from_checkpoint(initial, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx')
names = ['distribution.std_param'] + ['mlp.' + n for n, _ in mean.named_parameters()]
ids = dict(zip(names, initial['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
learned = [n for n in names if n.startswith('mlp.wheel_authority.')]
frozen = [n for n in names if n not in learned]
slots0 = initial['optimizer_state_dict']['state']
expected_opt = deepcopy(parent['optimizer_state_dict'])
expected_opt['param_groups'][0]['lr'] = .001
counters = ('successful_updates_total', 'actor_updates_total', 'active_branch_actor_updates',
    'feedback_branch_actor_updates', 'post_split_actor_updates', 'inherited_shared_feedback_updates',
    'consumed_optimizer_updates_total', 'training_transitions_total')
checks = dict(
    exact_declared1296_parent=sha(parent_path) == candidate['parent_checkpoint_sha256'] == info['parent_checkpoint_sha256'],
    actor_critic_sampling_rng_exact_at_entry=all(exact(v, initial[k]) for k, v in parent.items()
        if k not in ('infos', 'env_generator_state', 'optimizer_state_dict')),
    only_actor_rate_changes_in_initial_Adam=exact(initial['optimizer_state_dict'], expected_opt),
    all_six_native_full_restore_checks=len(info['full_restore_verification']) == 6
        and all(info['full_restore_verification'].values()),
    actual_rate_application_verified=info['optimizer_after_declared_rate_exact']
        and info['optimizer_rate_application_order'] == 'full_parent_restore_verified_then_declared_actor_rate'
        and info['optimizer_rate_contract']['before'] == dict(actor=.0003, critic=.0005)
        and info['optimizer_rate_contract']['after'] == dict(actor=.001, critic=.0005),
    exact_isolated_plan_revision=info['changed_plan_fields'] == ['learning_rate']
        and info['plan_revision_sha256'] == sha(revision_path) == candidate['revision_sha256']
        and info['plan_sha256'] == candidate['plan_sha256'] == sha(plan_path),
    all_training_sources_exact=info['source_sha256'] == before['source_sha256'] == candidate['source_sha256']
        and all(sha(ROOT / n) == value for n, value in info['source_sha256'].items()),
    existing994_wheel_parameters_only=len(learned) == 4
        and set(info['actor_trainability']['trainable_names']) == set(learned)
        and info['actor_trainability']['trainable_parameters'] == 994,
    wheel_Adam_steps_inherited=all(float(slots0[ids[n]]['step']) == 192. for n in learned),
    no_critic_warmup=before['critic_warmup_remaining'] == info['critic_warmup_remaining']
        == plan['additional_critic_warmup_updates'] == 0,
    counters_exact_at_entry=all(info[k] == before[k] for k in counters),
    population_and_clocks_preserved=info['num_envs'] == before['num_envs'] == 4096
        and all(info['startup']['runtime_preflight']['observed'][k] == v for k, v in
            dict(physics_dt=.001, policy_dt=.02, pc_control_dt=.001, decimation=20).items()),
    other_learning_contracts_preserved=all(info[k] == before[k] for k in (
        'teacher_anchor_runtime', 'static_feedback_recall', 'training_command_reference',
        'height_command_metrics_contract', 'termination_metrics_contract', 'control_math_sha256')),
)
milestones = {}
for count in ((1297,) if args.first_only else (1297, 1328)):
    path = trial / f'model_{count:06d}.pt'
    state = load(path)
    row, slots, delta = state['infos'], state['optimizer_state_dict']['state'], count - 1296
    proof = dict(
        all_frozen_actor_state_exact=all(torch.equal(v, state['actor_state_dict'][n])
            for n, v in initial['actor_state_dict'].items() if not n.startswith('mlp.wheel_authority.')),
        frozen_actor_Adam_exact=all(exact(slots0.get(ids[n]), slots.get(ids[n])) for n in frozen),
        four_wheel_tensors_learned=all(not torch.equal(initial['actor_state_dict'][n], state['actor_state_dict'][n])
            for n in learned),
        exactly994_trainable_parameters=row['actor_trainability']['trainable_parameters'] == 994
            and set(row['actor_trainability']['trainable_names']) == set(learned),
        critic_learned=any(not torch.equal(v, state['critic_state_dict'][n])
            for n, v in initial['critic_state_dict'].items()),
        twelve_Adam_steps_per_update=all(float(slots[i]['step']) == float(slots0[i]['step']) + 12 * delta
            for i in [ids[n] for n in learned] + initial['optimizer_state_dict']['param_groups'][1]['params']),
        counters_exact=all(row[k] == before[k] + (delta if k in (
            'successful_updates_total', 'actor_updates_total', 'feedback_branch_actor_updates',
            'post_split_actor_updates', 'consumed_optimizer_updates_total') else delta * 4096 * 96
            if k == 'training_transitions_total' else 0) for k in counters),
        rate_and_finite_losses=[g['lr'] for g in state['optimizer_state_dict']['param_groups']] == [.001, .0005]
            and all(math.isfinite(v) for v in row['last_losses'].values()),
        revision_preserved=row['plan_revision_sha256'] == info['plan_revision_sha256']
            and row['plan_sha256'] == info['plan_sha256'],
    )
    checks.update({str(count) + '_' + k: v for k, v in proof.items()})
    milestones[str(count)] = dict(checkpoint_sha256=sha(path), checks=proof)
if not args.first_only:
    events = EventAccumulator(str(trial), size_guidance={'scalars': 0})
    events.Reload()
    checks['thirty_two_updates_in_original_RSL_events'] = all(
        [p.step for p in events.Scalars(tag)] == list(range(1297, 1329))
        for tag in ('FlatTransfer/actor_updates_total', 'FlatTransfer/feedback_branch_actor_updates',
            'FlatTransfer/training_transitions_total'))
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-rate-native-state-audit-v1', passed=True, checks=checks,
    parent_checkpoint_sha256=sha(parent_path), initial_state_sha256=sha(trial / 'initial_state.pt'),
    candidate_sha256=sha(candidate_path), revision_sha256=sha(revision_path), milestones=milestones,
    reader_sha256=sha(Path(__file__)), new_optimizer_updates=0,
    scope='Read-only exact state restoration and learning ownership; native fixed replays decide behavior.'),
    indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), milestones=milestones), indent=2))
