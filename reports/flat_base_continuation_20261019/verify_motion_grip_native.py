"""Authenticate native motion learning and preservation of the adapted static head."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--count', type=int, choices=(1339, 1341, 1348), required=True)
args = parser.parse_args()
torch.set_num_threads(1)
candidate = HERE / 'motion_grip_candidate001'
trial = ROOT / 'runs/flat_base_20261019/pilot030_motion_grip1348'
load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
parent = load(ROOT / 'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt')
initial = load(trial / 'initial_state.pt')
state = load(trial / f'model_{args.count:06d}.pt')
before, row = initial['infos'], state['infos']
prep = json.loads((candidate / 'preparation001.json').read_text())


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
mean = mean_from_checkpoint(initial, source)
current = mean_from_checkpoint(state, source)
names = ['distribution.std_param'] + ['mlp.' + name for name, _ in mean.named_parameters()]
ids = dict(zip(names, initial['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
prefixes = ('mlp.feedback.', 'mlp.fast_feedback.', 'mlp.wheel_authority.')
motion_names = [name for name in names if name.startswith(prefixes)]
old_slots = initial['optimizer_state_dict']['state']
new_slots = state['optimizer_state_dict']['state']
delta = args.count - 1338
actor_delta = max(0, delta - 2)
checks = dict(
    exact_parent=before['parent_checkpoint_sha256'] == prep['parent_checkpoint_sha256'],
    exact_actor_critic_and_sampling_rng_at_entry=all(exact(parent[key], initial[key]) for key in
        ('actor_state_dict', 'critic_state_dict', 'torch_rng_state', 'cuda_rng_state')),
    entire_initial_Adam_exact=exact(parent['optimizer_state_dict'], initial['optimizer_state_dict']),
    all_native_restore_checks=len(before['full_restore_verification']) == 6
        and all(before['full_restore_verification'].values()) and before['optimizer_after_declared_rate_exact'],
    explicit_v7_to_v8_identity_revision=before['motion_grip_revision']['initial_mean_is_identity']
        and before['motion_grip_revision']['trainable_actor_branches'] == ['feedback', 'fast_feedback', 'wheel_authority'],
    frozen_adapted_static_parent=before['motion_grip_parent'] == row['motion_grip_parent']
        and row['motion_grip_parent']['sha256'] == prep['parent_checkpoint_sha256'],
    exact_reviewed_sources=before['source_sha256'] == row['source_sha256'] == prep['source_sha256']
        and all(digest(ROOT / name) == sha for name, sha in prep['source_sha256'].items()),
    exact_plan_and_revision=before['plan_sha256'] == prep['plan_sha256']
        and before['plan_revision_sha256'] == prep['revision_sha256']
        and before['changed_plan_fields'] == prep['changed_plan_fields'],
    unchanged_counters_at_entry=all(before[key] == parent['infos'][key] for key in
        ('successful_updates_total', 'actor_updates_total', 'consumed_optimizer_updates_total', 'training_transitions_total')),
    explicit_two_critic_updates=before['critic_warmup_remaining'] == 2
        and row['critic_warmup_remaining'] == max(0, 2 - delta),
    only_motion_actor_state_can_change=all(torch.equal(value, state['actor_state_dict'][name])
        for name, value in initial['actor_state_dict'].items() if not name.startswith(prefixes)),
    all_other_actor_Adam_exact=all(exact(old_slots.get(ids[name]), new_slots.get(ids[name]))
        for name in names if name not in motion_names),
    static_counter_retained=before['static_grip_actor_updates'] == row['static_grip_actor_updates'] == 8,
    critic_learned=any(not torch.equal(value, state['critic_state_dict'][name])
        for name, value in initial['critic_state_dict'].items()),
    critic_Adam_steps=all(float(new_slots[i]['step']) == float(old_slots[i]['step']) + 12 * delta
        for i in initial['optimizer_state_dict']['param_groups'][1]['params']),
    counters_exact=row['successful_updates_total'] == 1338 + delta
        and row['actor_updates_total'] == before['actor_updates_total'] + actor_delta
        and row['motion_grip_actor_updates'] == actor_delta
        and row['consumed_optimizer_updates_total'] == before['consumed_optimizer_updates_total'] + delta
        and row['training_transitions_total'] == before['training_transitions_total'] + delta * 4096 * 96,
    population_and_clocks=row['num_envs'] == 4096 and all(
        row['startup']['runtime_preflight']['observed'][key] == value for key, value in
        dict(physics_dt=.001, policy_dt=.02, pc_control_dt=.001, decimation=20).items()),
    current_material=row['startup']['contact_material']['wheel_static_friction'] == .1
        and row['startup']['contact_material']['ground_static_range'] == [.2, .5]
        and row['startup']['contact_material']['ground_dynamic_range'] == [.2, .5]
        and row['startup']['contact_material']['native_readback']['usd_collision_bindings_verified']
        and row['startup']['contact_material']['native_readback']['physx_robot_shape_coefficients_verified'],
    finite_models_and_losses=all(torch.isfinite(value).all().item() for key in
        ('actor_state_dict', 'critic_state_dict') for value in state[key].values())
        and all(torch.isfinite(torch.tensor(value)).all().item() for value in row['last_losses'].values()),
)
if actor_delta:
    checks.update(twelve_motion_tensors_learned=all(not torch.equal(initial['actor_state_dict'][name],
        state['actor_state_dict'][name]) for name in motion_names),
        only3246_motion_parameters_train=row['actor_trainability']['trainable_parameters'] == 3246
            and set(row['actor_trainability']['trainable_names']) == set(motion_names),
        motion_Adam_steps=all(float(new_slots[ids[name]]['step']) ==
            (float(old_slots[ids[name]]['step']) if ids[name] in old_slots else 0.) + 12 * actor_delta
            for name in motion_names))
else:
    checks['all_actor_and_Adam_frozen_during_critic_warmup'] = (
        exact(initial['actor_state_dict'], state['actor_state_dict'])
        and all(exact(old_slots.get(ids[name]), new_slots.get(ids[name])) for name in names))
if args.count == 1348:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(trial), size_guidance={'scalars': 0})
    events.Reload()
    values = events.Scalars('FlatTransfer/motion_grip_actor_updates')
    checks['all_ten_updates_in_RSL_event'] = [v.step for v in values] == list(range(1339, 1349))
    checks['motion_counter_in_RSL_event'] = [v.value for v in values] == [0., 0., *map(float, range(1, 9))]
target = candidate / f'native_{args.count}.json'
if target.exists():
    raise SystemExit('Preserve existing native motion audit')
target.write_bytes(encode(dict(schema='flat-motion-grip-native-ownership-v1', passed=all(checks.values()),
    checks=checks, checkpoint_sha256=digest(trial / f'model_{args.count:06d}.pt'),
    initial_state_sha256=digest(trial / 'initial_state.pt'), reader_sha256=digest(Path(__file__)),
    successful_updates=args.count, motion_actor_updates=actor_delta, behavior_qualified=False)))
print(json.dumps(dict(passed=all(checks.values()), checks=checks)))
if not all(checks.values()):
    raise SystemExit(1)
