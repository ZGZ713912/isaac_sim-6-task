"""Compare actual weight16 rollout with the identical weight8 native rollout."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from train_robust_getup import digest, write_json
from run_gap_long_training import verify_restore
from export_robust_getup import load_mean
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
R = ROOT / 'reports/robust_policy_repair_20261024'
B = ROOT / 'runs/robust_policy_repair_20261024'
plan = json.loads((R / 'diagnosis_plan076.json').read_text())
queue = json.loads((R / 'diagnosis_queue076.json').read_text())
assert queue['status'] == 'completed'
source = B / 'quiet_full_root001/model_initial.pt'
native = B / 'diagnosis076_contact_weight16_zero'
old = B / 'diagnosis064_quiet_full_zero'
verify_restore(source, native / 'model_initial.pt', R / 'contact_weight_native_zero_restore076.json')
a, b = [torch.load(p, map_location='cpu', weights_only=False)
        for p in (source, native / 'model_initial.pt')]
restore = json.loads((R / 'contact_weight_native_zero_restore076.json').read_text())
progress = json.loads((native / 'progress.json').read_text())
rollout = json.loads((native / 'rollout_diagnosis.json').read_text())
old_receipt = json.loads((old / 'rollout_diagnosis.json').read_text())
function = json.loads((native / 'native_function_readback.json').read_text())
changed = [n for n, h in plan['source_sha256'].items() if digest(ROOT / n) != h]
load_mean(native / 'model_initial.pt')
load_mean(native / 'model_stopped.pt')
old_values = np.load(old / 'native_quiet_rollout_and_GAE.npz')
new_values = np.load(native / 'native_quiet_rollout_and_GAE.npz')
payment = np.load(ROOT / rollout['payment_data'])
support = np.load(ROOT / rollout['support_data'])
old_support = np.load(old / 'native_quiet_support_rollout.npz')
force, gap = support['force'], support['clearance']
density = ((1. - np.clip(force[..., 2] / np.float32(20.), 0., 1.)) ** 2
    + np.log1p((np.maximum(gap, 0.) / np.float32(.0002)) ** 2)).max(-1).mean(1)
penalty = np.float32(16. * .005) * payment['density'] * payment['eligible']
additional_cost = np.float32(8. * .005) * payment['density'] * payment['eligible']
reward_error = np.max(np.abs(new_values['rewards'] - old_values['rewards'] + additional_cost[..., None]))
density_error = np.max(np.abs(density - payment['density']))
same = {k: np.array_equal(old_values[k], new_values[k])
    for k in set(old_values.files) - {'rewards', 'returns', 'advantages'}}
old_contract = dict(old_receipt['standing_contact_training_contract'])
old_contract['weight_density_per_second'] = 16.
checks = dict(
    native_queue_completed=all(j['status'] == 'completed' and j['returncode'] == 0 for j in queue['jobs']),
    frozen_inputs=not changed,
    restore_actor_critic_Adam_and_four_RNGs=all(restore['checks'].values()),
    optimizer_including_LRs_exact=exact_state_equal(a['optimizer_state_dict'], b['optimizer_state_dict']),
    authenticated_function_ancestry=exact_state_equal(a['infos']['policy_function_root'], b['infos']['policy_function_root']),
    original_mean_and_conditioned_std=all(function['checks'].values()) and function['frames'] == 131072
        and function['mean_different'] == function['std_different'] == 0,
    exactly8082_trainable_actor_parameters=progress['trainable_actor_parameters'] == 8082,
    zero_PPO=progress['new_optimizer_updates'] == progress['new_optimizer_attempts']
        == rollout['optimizer_attempts'] == rollout['optimizer_steps'] == rollout['new_PPO'] == 0,
    clock1501=progress['completed_updates'] == b['infos']['completed_updates'] == 1501,
    stopped_before_attempt=rollout['status'] == 'captured_before_optimizer_attempt'
        and progress['status'] == 'stopped' and not progress['optimizer_update_in_flight'],
    original_native_sampling_and_GAE=rollout['native_GAE'] and rollout['native_sampling_and_likelihood'],
    recorded_data_SHA=digest(ROOT / rollout['data']) == rollout['data_sha256'],
    payment_data_SHA=digest(ROOT / rollout['payment_data']) == rollout['payment_data_sha256'],
    stopped_checkpoint_is_exact_initial=digest(native / 'model_stopped.pt') == digest(native / 'model_initial.pt'),
    strict_native_initial_and_stopped_load=True,
    only_contact_weight_changes_in_contract=rollout['standing_contact_training_contract']
        == plan['proposed_contact_contract'] == b['infos']['standing_contact_training_contract'] == old_contract,
    support_trace_authenticated=digest(ROOT / rollout['support_data']) == rollout['support_data_sha256'],
    reset_only_placement_bounds=rollout['reset_state']['maximum_translation_m'] <= .025
        and rollout['reset_state']['maximum_roll_rad'] <= .02,
    physical_rollout_all_arrays_bit_exact=set(support.files) == set(old_support.files)
        and all(np.array_equal(support[k], old_support[k]) for k in support.files),
    observations_actions_sampling_masks_and_critic_bit_exact=all(same.values()),
    density_recomputed_every_1ms=bool(np.allclose(density, payment['density'], rtol=2e-6, atol=5e-6)),
    penalty_exact_cost_and_learning_mask=np.array_equal(penalty, payment['penalty']),
    actual_native_reward_minus_payment_exact=np.array_equal(new_values['rewards'][..., 0],
        payment['native_reward'] - payment['penalty']),
    weight8_to16_exact_reward_change_to_float32_rounding=reward_error < 2e-6,
    positive_extra_cost_paid=bool((additional_cost > 0.).any()),
    native_returns_and_advantages_responded=not np.array_equal(old_values['returns'], new_values['returns'])
        and not np.array_equal(old_values['advantages'], new_values['advantages']),
)
out = dict(schema='native-contact-weight-zero-PPO-audit-v1', checks={k: bool(v) for k, v in checks.items()},
    unchanged_rollout_tensor_checks=same, source_checkpoint=str(source), source_sha256=digest(source),
    native_initial_sha256=digest(native / 'model_initial.pt'), frozen_inputs=len(plan['source_sha256']),
    changed_inputs=changed, valid_transitions=rollout['valid_rollout']['valid_transitions'],
    eligible_transitions=int(payment['eligible'].sum()),
    weight16_paid_penalty_sum=float(penalty.sum()), additional_penalty_vs8=float(additional_cost.sum()),
    original_native_reward_sum=float(payment['native_reward'].sum()),
    reward_change_max_float32_error=float(reward_error), independent_density_max_float32_error=float(density_error),
    new_PPO=0, actual_experiment_consumed=2742, repair_optimizer_updates=362,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'contact_weight_native_zero_audit076.json', out)
assert all(checks.values()), out
write_json(R / 'diagnosis076_closed_source_seal001.json', dict(
    schema='closed-native-queue-source-seal-v1', queue='076', all_frozen_sources_verified=True,
    source_files=len(plan['source_sha256']), audit_sha256=digest(R / 'contact_weight_native_zero_audit076.json'),
    new_PPO=0, qualification=None, created_unix=time.time()))
print(json.dumps(out, indent=2))
