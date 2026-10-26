"""Audit actual bounded entry PPO state without assuming a particular clock."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
source = Path(plan['source_checkpoint'])
before = torch.load(source, map_location='cpu', weights_only=False)
after = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
progress = json.loads((Path(plan['training_output']) / 'progress.json').read_text())
restore = json.loads((Path(plan['supervisor_output']) / 'initial_restore.json').read_text())
count = progress['new_optimizer_updates']
prefix = 'mlp.mean.entry_recovery.'
changed = [key for key, value in before['actor_state_dict'].items()
           if not torch.equal(value, after['actor_state_dict'][key])]
actor_ids = before['optimizer_state_dict']['param_groups'][0]['params']
entry_ids = actor_ids[-6:]
steps = {}
for key in entry_ids:
    old = before['optimizer_state_dict']['state'].get(key, {}).get('step', torch.tensor(0.))
    new = after['optimizer_state_dict']['state'][key]['step']
    steps[str(key)] = dict(before=float(old), after=float(new), increment=float(new - old))
checks = dict(initial_restore_all=all(restore['checks'].values()),
    sealed_completed_or_stopped=progress['status'] in ('completed', 'stopped'),
    attempts_match_completed=progress['new_optimizer_attempts'] == count,
    clock=after['infos']['completed_updates'] == before['infos']['completed_updates'] + count,
    only_six_entry_tensors_changed=set(changed) == {k for k in before['actor_state_dict'] if k.startswith(prefix)},
    frozen_actor_std_exact=all(torch.equal(value, after['actor_state_dict'][key])
        for key, value in before['actor_state_dict'].items() if not key.startswith(prefix)),
    frozen_actor_Adam_exact=all(exact_state_equal(before['optimizer_state_dict']['state'].get(key),
        after['optimizer_state_dict']['state'].get(key)) for key in actor_ids if key not in entry_ids),
    entry_Adam_exact_step_increment=all(x['increment'] == 12 * count for x in steps.values()),
    critic_changed=not exact_state_equal(before['critic_state_dict'], after['critic_state_dict']))
mean, info = load_mean(args.checkpoint)
checks['strict_V3_decode'] = info['schema'] == 'robust-getup-checkpoint-v3'
out = dict(schema='bounded-entry-native-endpoint-audit-v2', checks=checks,
    plan=str(args.plan), source=str(source), source_sha256=digest(source),
    checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
    actor_class=type(mean).__name__, changed_actor_tensors=changed, entry_Adam_steps=steps,
    actual_completed_PPO=count, learning_clock=info['completed_updates'],
    actual_experiment_consumed=plan['prior_experiment_consumed'] + progress['new_optimizer_attempts'],
    scope='Sealed tensors, clocks, real optimizer step counters, source identity and strict decoding; behavior reviewed separately',
    qualification=None, hardware_started=False, created_unix=time.time())
assert not args.output.exists()
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
assert all(checks.values()), checks
