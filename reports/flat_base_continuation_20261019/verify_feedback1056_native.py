"""Verify exact1024 restoration, clock migration, critic-only adaptation and v15 learning."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.separated_feedback import KIND

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--through', type=int, choices=(1025, 1034, 1035, 1056), default=1056)
args = parser.parse_args()
torch.set_num_threads(1)
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parent = load(RUN / 'pilot019/model_001024.pt')
trial = RUN / 'pilot020'
initial = load(trial / 'initial_state.pt')
info = initial['infos']
revision = json.loads((HERE / 'training_plan_revision017.json').read_text())


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(v, b[k]) for k,v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x,y in zip(a,b))
    return a == b


checks = {f'initial_{key}_exact_parent': exact(initial[key], parent[key]) for key in
    ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'torch_rng_state', 'cuda_rng_state')}
checks.update(native_six_restore_checks=all(info['full_restore_verification'].values()),
    exact_declared_migration=info['plan_revision_sha256'] == sha(HERE / 'training_plan_revision017.json')
        and info['source_sha256'] == revision['new_source_sha256']
        and info['changed_plan_fields'] == revision['changed_plan_fields']
        and info['plan_sha256'] == revision['new_plan_sha256'],
    no_rate_change=info['optimizer_after_declared_rate_exact'] and not info['optimizer_rate_contract']['declared_actor_rate_change'],
    same_actor_kind_no_fork=info['actor_adaptation'] == KIND and info['restore_verification_reference'] == 'parent',
    ten_critic_updates_declared=info['critic_warmup_remaining'] == 10,
    parent_counters=(info['successful_updates_total'], info['actor_updates_total'], info['active_branch_actor_updates'],
        info['feedback_branch_actor_updates'], info['post_split_actor_updates'], info['inherited_shared_feedback_updates'],
        info['consumed_optimizer_updates_total']) == (1024,864,44,486,182,304,1025),
    teacher_unchanged=info['teacher_anchor_runtime'] == parent['infos']['teacher_anchor_runtime'])
config = json.loads((trial / 'config/contract.json').read_text())
old_config = json.loads((RUN / 'pilot019/config/contract.json').read_text())
checks['only_declared_clock_sampling_and_rewards_retained'] = (
    config['physics_dt'] == .001 and old_config['physics_dt'] == .005 and
    config['policy_dt'] == old_config['policy_dt'] == .02 and
    config['scene_groups'] == old_config['scene_groups'] and
    config['stationary_tracking'] == old_config['stationary_tracking'])
timing = info['startup']['runtime_preflight']['observed']
checks['actual_native50_policy1000_physics_and_PD'] = all(timing[k] == v for k,v in dict(
    physics_dt=.001,policy_dt=.02,pc_control_dt=.001,decimation=20).items())
transition = info['feedback_clock_transition']
checks['authenticated_clock_transition_and_same_gains'] = (transition['old']['physics_dt'] == .005 and
    transition['new']['physics_dt'] == .001 and transition['policy_clock_unchanged'] and
    transition['gamma_gae_and_rollout_unchanged'] and transition['gains_geometry_actuators_unchanged'])
checks['actual_command_reference_clock'] = info['training_command_reference']['physics_hz'] == 1000
mean = mean_from_checkpoint(initial, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx')
names = ['distribution.std_param'] + ['mlp.' + n for n,_ in mean.named_parameters()]
ids = dict(zip(names, initial['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
prefixes = ('mlp.feedback.', 'mlp.fast_feedback.')
learned = [n for n in names if n.startswith(prefixes)]
frozen = [n for n in names if not n.startswith(prefixes)]
slots0 = initial['optimizer_state_dict']['state']
required = [n for n in (1025,1034,1035,1056) if n <= args.through]
milestones = {}
for step in required:
    path = trial / f'model_{step:06d}.pt'
    if not path.exists():
        continue
    state = load(path)
    row, slots = state['infos'], state['optimizer_state_dict']['state']
    delta, actor_delta = step - 1024, max(0, step - 1034)
    proof = dict(
        frozen_actor_std_buffers_exact=all(torch.equal(v, state['actor_state_dict'][n])
            for n,v in initial['actor_state_dict'].items() if not n.startswith(prefixes)),
        frozen_Adam_slots_exact=all(exact(slots0.get(ids[n]), slots.get(ids[n])) for n in frozen),
        feedback_critic_Adam_steps_exact=(all(float(slots[ids[n]]['step']) == float(slots0[ids[n]]['step']) + 12*actor_delta for n in learned)
            and all(float(slots[i]['step']) == float(slots0[i]['step']) + 12*delta for i in initial['optimizer_state_dict']['param_groups'][1]['params'])),
        critic_learned=any(not torch.equal(v,state['critic_state_dict'][n]) for n,v in initial['critic_state_dict'].items()),
        all_counters_without_refund=(row['successful_updates_total'], row['actor_updates_total'], row['active_branch_actor_updates'],
            row['feedback_branch_actor_updates'], row['post_split_actor_updates'], row['inherited_shared_feedback_updates'],
            row['consumed_optimizer_updates_total'], row['critic_warmup_remaining']) ==
            (step,864+actor_delta,44,486+actor_delta,182+actor_delta,304,step+1,max(0,10-delta)),
        exact_group_rates=[g['lr'] for g in state['optimizer_state_dict']['param_groups']] == [3e-4,5e-4],
        finite_losses=all(math.isfinite(v) for v in row['last_losses'].values()))
    if actor_delta:
        proof['eight_feedback_tensors_learned'] = len(learned) == 8 and all(
            not torch.equal(initial['actor_state_dict'][n],state['actor_state_dict'][n]) for n in learned)
        proof['exactly2252_feedback_parameters_trainable'] = row['actor_trainability']['trainable_parameters'] == 2252 and set(row['actor_trainability']['trainable_names']) == set(learned)
    else:
        proof['entire_actor_and_Adam_exact_during_critic_adaptation'] = (
            exact(initial['actor_state_dict'],state['actor_state_dict']) and
            all(exact(slots0.get(ids[n]),slots.get(ids[n])) for n in names))
    checks.update({str(step) + '_' + k:v for k,v in proof.items()})
    milestones[str(step)] = dict(checkpoint_sha256=sha(path), checks=proof, last_losses=row['last_losses'])
receipt = dict(schema='flat-feedback1056-native-full-resume-milestones-v1', passed=all(checks.values()), checks=checks,
    all_required_milestones_present=len(milestones) == len(required), milestones=milestones,
    through=args.through, optimizer_updates_performed_by_verifier=0, behavior_qualified=False)
path = HERE / ('feedback1056_native_milestones.json' if args.through == 1056 else f'feedback1056_native{args.through}.json')
path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=receipt['passed'], checks=len(checks), milestones=list(milestones),
    all_required_milestones_present=receipt['all_required_milestones_present'], failures=[k for k,v in checks.items() if not v])))
if not receipt['passed'] or not receipt['all_required_milestones_present']:
    raise SystemExit(1)
