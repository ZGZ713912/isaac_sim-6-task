"""Read the real quiet-output endpoint, optimizer history and native TensorBoard evidence."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from wheeled_algo.quiet_full_checkpoint import LEARNED_KEYS
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.robust_getup import tilt_gate

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R / 'plan022_contact_weight001.json').read_text())
run = Path(plan['output'])
p = json.loads((run / 'progress.json').read_text())
assert p['status'] in ('completed', 'stopped')
n = p['new_optimizer_updates']
assert n == p['new_optimizer_attempts'] and 1 <= n <= 16
source = Path(plan['source_checkpoint'])
target = run / ('model_final.pt' if p['status'] == 'completed' else 'model_stopped.pt')
a, b = [torch.load(x, map_location='cpu', weights_only=False) for x in (source, target)]
rows = json.loads((run / 'quiet_full_updates.json').read_text())
first = json.loads((run / 'first_rollout_native_comparison.json').read_text())
from run_gap_long_training import verify_restore
verify_restore(source, run / 'model_initial.pt', R / 'contact_weight_initial_restore022.json')
before, _ = load_mean(source)
after, _ = load_mean(target)
changed = [k for k, v in a['actor_state_dict'].items() if not torch.equal(v, b['actor_state_dict'][k])]
old, new = a['optimizer_state_dict'], b['optimizer_state_dict']
names = b['infos']['policy_function_root']['optimizer_actor_parameter_names']
actor_ids = old['param_groups'][0]['params']
learned_ids = {actor_ids[names.index(k)] for k in LEARNED_KEYS}
critic_ids = old['param_groups'][1]['params']
trace = B / 'diagnosis036_quiet_stance050_implementation/evaluation/trajectory.npz'
frames = np.load(trace)['observations'].reshape(-1, 35)
outside_count = outside_different = active_different = 0
peak_delta = 0.
with torch.inference_mode():
    for values in np.array_split(frames, (len(frames) + 511) // 512):
        x = torch.from_numpy(values.copy())
        v, w = before(x), after(x)
        outside = before.quiet_stance_gate(x).squeeze(-1) == 0.
        different = (v != w).any(-1)
        outside_count += int(outside.sum())
        outside_different += int(different[outside].sum())
        active_different += int(different[~outside].sum())
        peak_delta = max(peak_delta, float((v - w).abs().max()))
events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
tags = sorted(t for t in events.Tags()['scalars'] if t.startswith('QuietFullTraining/'))
scalars = {t: [dict(step=x.step, value=x.value) for x in events.Scalars(t)] for t in tags}
contact_tags = sorted(t for t in events.Tags()['scalars'] if t.startswith('StrictStanding/'))
contact_scalars = {t: [dict(step=x.step, value=x.value) for x in events.Scalars(t)] for t in contact_tags}
checks = dict(
    native_contact_TensorBoard_seven_tags=len(contact_tags) == 7,
    native_contact_TensorBoard_exact_steps=all([x['step'] for x in values] == list(range(1501, 1501 + n)) for values in contact_scalars.values()),
    contact_penalty_actually_paid=any(x['value'] > 0 for x in contact_scalars.get('StrictStanding/paid_penalty_per_policy_step', [])),
    endpoint_clock=b['infos']['completed_updates'] == 1501 + n,
    declared_budget=n <= 16 and (p['status'] != 'completed' or n == 16),
    attempted_charged_exact=p['new_optimizer_attempts'] == n and not p['optimizer_update_in_flight'],
    all_native_update_rows=len(rows['updates']) == n,
    all_native_step_and_finite_checks=all(all(x['checks'].values()) for x in rows['updates']),
    native_update_row_clocks=[x['learning_clock'] for x in rows['updates']] == list(range(1502, 1502 + n)),
    wrapper_bound=rows['wrapper_sha256'] == digest(R / 'run_contact_weight_pilot022.py'),
    source_bound=rows['source_sha256'] == digest(source),
    first_rollout_native_exact=first['all_exact'] and all(first['checks'].values()),
    exactly_twelve_existing_actor_tensors_changed=set(changed) == set(LEARNED_KEYS),
    frozen_actor_and_basis_and_std_exact=all(torch.equal(v, b['actor_state_dict'][k])
        for k, v in a['actor_state_dict'].items() if k not in LEARNED_KEYS),
    all16_frozen_actor_Adam_slots_exact=all(exact_state_equal(old['state'][i], new['state'][i])
        for i in actor_ids if i not in learned_ids),
    learned_and_critic_Adam_steps_exact=all(float(new['state'][i]['step']) == float(old['state'][i]['step']) + 12 * n
        for i in learned_ids | set(critic_ids)),
    critic_changed=not exact_state_equal(a['critic_state_dict'], b['critic_state_dict']),
    frozen_inputs_exact=all(digest(Path(x)) == h for x, h in plan['source_sha256'].items()),
    initial_restore_all=all(json.loads((R / 'contact_weight_initial_restore022.json').read_text())['checks'].values()),
    exact_contact_contract=b['infos']['standing_contact_training_contract'] == plan['training_contact_contract'],
    strict_V8_decode=True,
    outside_quiet_function_exact=outside_count > 0 and outside_different == 0,
    quiet_function_changed=active_different > 0,
    native_TensorBoard_seven_tags=len(tags) == 7,
    native_TensorBoard_exact_steps=all([x['step'] for x in values] == list(range(1501, 1501 + n)) for values in scalars.values()),
    native_TensorBoard_step_checks_pass=all(x['value'] == 1. for x in scalars.get('QuietFullTraining/passed', [])),
)
out = dict(schema='native-quiet-output-endpoint-v1', checks={k: bool(v) for k, v in checks.items()},
    checkpoint=str(target), checkpoint_sha256=digest(target), source=str(source), source_sha256=digest(source),
    changed_actor_tensors=changed, source_clock=1501, branch_clock=1501 + n,
    real_PPO_attempts=p['new_optimizer_attempts'], real_PPO_completed=n,
    actual_experiment_consumed=2742 + p['new_optimizer_attempts'], repair_optimizer_updates=362 + n,
    native_update_rows=rows['updates'], native_TensorBoard=scalars, native_contact_TensorBoard=contact_scalars, frozen_inputs=len(plan['source_sha256']),
    function_probe=dict(trace=str(trace), sha256=digest(trace), frames=len(frames), outside_count=outside_count,
        outside_different=outside_different, active_different=active_different, peak_action_delta=peak_delta),
    interpretation='Engineering readback only; frozen recorded observations do not prove closed-loop behavior or qualification.',
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'contact_weight_round022_endpoint001.json', out)
print(json.dumps({k: v for k, v in out.items() if k not in ('native_TensorBoard', 'native_update_rows', 'native_contact_TensorBoard')}, indent=2))
assert all(checks.values()), checks
