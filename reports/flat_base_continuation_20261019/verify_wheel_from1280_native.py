"""Read-only first/final state audit of the16-update v6 trial from composed1280."""
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
trial = RUN / 'pilot027_from1280'
out = HERE / ('wheel_from1280_native_first001.json' if args.first_only else 'wheel_from1280_native_final001.json')
if out.exists():
    raise SystemExit('Preserve completed audit')
parent_path = RUN / 'pilot025/model_001280_wheel_authority_composed002.pt'
adoption_path = HERE / 'wheel_authority_from1280_adoption023.json'
parent, initial = load(parent_path), load(trial / 'initial_state.pt')
before, info = parent['infos'], initial['infos']
adoption = read(adoption_path)
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
mean = mean_from_checkpoint(initial, source)

def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b

names = ['distribution.std_param'] + ['mlp.' + n for n, _ in mean.named_parameters()]
ids = dict(zip(names, initial['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
learned = [n for n in names if n.startswith('mlp.wheel_authority.')]
frozen = [n for n in names if n not in learned]
slots0 = initial['optimizer_state_dict']['state']
plan = read(ROOT / info['plan_file'])
counters = ('successful_updates_total', 'actor_updates_total', 'active_branch_actor_updates',
    'feedback_branch_actor_updates', 'post_split_actor_updates', 'inherited_shared_feedback_updates',
    'consumed_optimizer_updates_total', 'training_transitions_total')
checks = dict(
    exact_composed1280_parent=sha(parent_path) == info['parent_checkpoint_sha256'] == adoption['parent_checkpoint_sha256'],
    complete_actor_critic_Adam_and_sampling_rng_exact=all(exact(v, initial[k]) for k, v in parent.items()
        if k not in ('infos', 'env_generator_state')),
    all_six_native_restore_checks=len(info['full_restore_verification']) == 6 and all(info['full_restore_verification'].values()),
    lifetime_counters_exact_at_entry=all(info[k] == before[k] for k in counters),
    own_v6_kind=info['actor_adaptation'] == before['actor_adaptation'] == KIND,
    exact_declared_execution_sources=info['source_sha256'] == before['source_sha256'] == adoption['new_source_sha256'],
    exact_plan=info['plan_sha256'] == before['plan_sha256'] == adoption['plan_sha256'] == sha(ROOT / info['plan_file']),
    teacher_stays_zero=info['teacher_anchor_runtime'] == before['teacher_anchor_runtime']
        and info['teacher_anchor_runtime']['coefficient'] == 0.,
    only_new_wheel_head_has_learning_scope=len(learned) == 4
        and info['actor_trainability']['trainable_parameters'] == 994
        and set(info['actor_trainability']['trainable_names']) == set(learned),
    new_wheel_head_has_no_inherited_Adam=all(ids[n] not in slots0 for n in learned),
    static_head_has_no_Adam=all(ids[n] not in slots0 for n in names if n.startswith('mlp.static_feedback.')),
    zero_wheel_final_layer=all(torch.count_nonzero(initial['actor_state_dict'][n]).item() == 0
        for n in learned if '.2.' in n),
    static_reference_exact=info['static_feedback_recall'] == plan['static_feedback_recall']
        and sha(ROOT / plan['static_feedback_recall']['checkpoint']) == plan['static_feedback_recall']['sha256'],
    unchanged_population_clocks_control_reference=info['num_envs'] == before['num_envs'] == 4096
        and info['control_math_sha256'] == before['control_math_sha256']
        and info['training_command_reference'] == before['training_command_reference']
        and all(info['startup']['runtime_preflight']['observed'][k] == v for k,v in
            dict(physics_dt=.001, policy_dt=.02, pc_control_dt=.001, decimation=20).items()),
    no_new_critic_warmup=info['critic_warmup_remaining'] == plan['additional_critic_warmup_updates'] == 0,
    unchanged_behavior_logging_contracts=all(info[k] == before[k]
        for k in ('height_command_metrics_contract', 'termination_metrics_contract')))
milestones = {}
for count in ((1281,) if args.first_only else (1281, 1296)):
    path = trial / f'model_{count:06d}.pt'
    state = load(path)
    row, slots, delta = state['infos'], state['optimizer_state_dict']['state'], count - 1280
    proof = dict(
        frozen_actor_state_and_buffers_exact=all(torch.equal(v, state['actor_state_dict'][n])
            for n,v in initial['actor_state_dict'].items() if not n.startswith('mlp.wheel_authority.')),
        frozen_Adam_exact=all(exact(slots0.get(ids[n]), slots.get(ids[n])) for n in frozen),
        all_four_wheel_tensors_learned=all(not torch.equal(initial['actor_state_dict'][n], state['actor_state_dict'][n]) for n in learned),
        exactly994_trainable_parameters=row['actor_trainability']['trainable_parameters'] == 994
            and set(row['actor_trainability']['trainable_names']) == set(learned),
        critic_learned=any(not torch.equal(v, state['critic_state_dict'][n]) for n,v in initial['critic_state_dict'].items()),
        twelve_Adam_steps_per_update=all(float(slots[i]['step']) == float(slots0.get(i, {'step':0})['step']) + 12*delta
            for i in [ids[n] for n in learned] + initial['optimizer_state_dict']['param_groups'][1]['params']),
        cumulative_counters_exact=all(row[k] == before[k] + (delta if k in (
            'successful_updates_total','actor_updates_total','feedback_branch_actor_updates',
            'post_split_actor_updates','consumed_optimizer_updates_total') else delta*4096*96
            if k == 'training_transitions_total' else 0) for k in counters),
        unchanged_rates_and_finite_losses=[g['lr'] for g in state['optimizer_state_dict']['param_groups']] == [3e-4,5e-4]
            and all(math.isfinite(v) for v in row['last_losses'].values()),
        unchanged_trial_contracts=row['teacher_anchor_runtime'] == info['teacher_anchor_runtime']
            and row['static_feedback_recall'] == info['static_feedback_recall'])
    checks.update({str(count)+'_'+k:v for k,v in proof.items()})
    milestones[str(count)] = dict(checkpoint_sha256=sha(path), checks=proof)
if not args.first_only:
    events = EventAccumulator(str(trial), size_guidance={'scalars':0})
    events.Reload()
    checks['sixteen_updates_logged_in_RSL_events'] = all([p.step for p in events.Scalars(tag)] == list(range(1281,1297))
        for tag in ('FlatTransfer/actor_updates_total','FlatTransfer/feedback_branch_actor_updates','FlatTransfer/training_transitions_total'))
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-from1280-native-audit-v1', passed=True,
    checks=checks, parent_checkpoint_sha256=sha(parent_path), initial_state_sha256=sha(trial/'initial_state.pt'),
    adoption_sha256=sha(adoption_path), milestones=milestones, new_optimizer_updates=0,
    scope='Read-only state/learning audit; full-horizon fixed native replays decide behavior.'),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True, checks=len(checks), milestones=milestones),indent=2))
