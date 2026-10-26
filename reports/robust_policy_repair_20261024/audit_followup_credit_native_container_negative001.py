"""Seal four native zero-update windows and the actual training state readback."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from run_gap_long_training import verify_restore
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_tasks.chassis.followup_continuity_reward import validate_contract

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R / 'diagnosis_plan046.json').read_text())
queue = json.loads((R / 'diagnosis_queue046.json').read_text())
assert queue['status'] == 'completed'
native = B / 'diagnosis046_followup_credit'
progress = json.loads((native / 'progress.json').read_text())
rollout = json.loads((native / 'rollout_diagnosis.json').read_text())
comparison = json.loads((native / 'first_window_original_reward_comparison.json').read_text())
source = B / 'motion_output_root001/model_initial.pt'
verify_restore(source, native / 'model_initial.pt', R / 'followup_credit_initial_restore001.json')
restore = json.loads((R / 'followup_credit_initial_restore001.json').read_text())
a, initial, endpoint = [torch.load(path, map_location='cpu', weights_only=False)
    for path in (source, native / 'model_initial.pt', Path(rollout['native_state_readback']))]
expected_optimizer = deepcopy(a['optimizer_state_dict'])
for before, actual in zip(expected_optimizer['param_groups'], endpoint['optimizer_state_dict']['param_groups'], strict=True):
    before['lr'] = actual['lr']
unchanged_windows = []
credit_rows = []
for row in rollout['windows']:
    path = Path(row['data'])
    assert digest(path) == row['data_sha256']
    old_path = B / 'diagnosis045_followup_credit' / path.name
    with np.load(path) as values, np.load(old_path) as old:
        unchanged_windows.append(dict(window=row['window'], original045_common_array_exact={
            key: bool(np.array_equal(values[key], old[key])) for key in old.files}))
        moving = values['moving_fall'] & values['learning_mask']
        bonus_positive = moving & (values['bonus'] > 0.)
        delta = (values['returns'] - values['original_returns'])[..., 0]
        adv = values['advantages'][..., 0]
        old_adv = values['original_advantages'][..., 0]
        credit_rows.append(dict(window=row['window'], moving_history_frames=int(moving.sum()),
            moving_bonus_frames=int(bonus_positive.sum()), moving_bonus_sum=float(values['bonus'][moving].sum()),
            native_return_delta_sum_on_moving_history=float(delta[moving].sum()),
            native_normalized_advantage_delta_sum_on_moving_history=float((adv - old_adv)[moving].sum()),
            moving_bonus_positive_mean_advantage=float(adv[bonus_positive].mean()) if bonus_positive.any() else None,
            moving_bonus_positive_original_mean_advantage=float(old_adv[bonus_positive].mean()) if bonus_positive.any() else None))
validate_contract(initial['infos']['training_reward_contract'], initial['infos']['source_sha256'])
load_mean(native / 'model_initial.pt')
load_mean(native / 'model_stopped.pt')
checks = dict(native_queue_complete=True, frozen_inputs=all(digest(p) == sha for p, sha in plan['source_sha256'].items()),
    initial_actor_critic_Adam_and_four_RNGs=all(restore['checks'].values()),
    final_actor_exact=exact_state_equal(a['actor_state_dict'], endpoint['actor_state_dict']),
    final_critic_exact=exact_state_equal(a['critic_state_dict'], endpoint['critic_state_dict']),
    final_Adam_slots_and_original_parameter_groups_exact=exact_state_equal(expected_optimizer, endpoint['optimizer_state_dict']),
    final_LRs_exactly_declared=[x['lr'] for x in endpoint['optimizer_state_dict']['param_groups']] == [.0001, .001],
    stopped_CP_exact_initial=digest(native / 'model_stopped.pt') == digest(native / 'model_initial.pt'),
    first_original044_all16_rollout_GAE_tensors_exact=comparison['all_exact'],
    four_windows_and_original045_all_arrays_exact=len(unchanged_windows) == 4 and all(
        all(x['original045_common_array_exact'].values()) for x in unchanged_windows),
    zero_PPO=progress['new_optimizer_attempts'] == progress['new_optimizer_updates'] == rollout['optimizer_attempts'] == rollout['optimizer_steps'] == 0,
    source_clock1501=progress['completed_updates'] == initial['infos']['completed_updates'] == 1501,
    source390_parameter_scope=progress['trainable_actor_parameters'] == 390,
    native_credit_exposure=all(rollout['checks'].values()),
    native_state_readback_SHA=digest(rollout['native_state_readback']) == rollout['native_state_readback_sha256'],
    strict_decoder_and_reward_contract=True)
out = dict(schema='native-followup-reward-credit-audit-v1', checks={k: bool(v) for k, v in checks.items()},
    credit_rows=credit_rows, preserved045_arrays=unchanged_windows, frozen_inputs=len(plan['source_sha256']),
    diagnosis='A positive bonus reaches actual moving-fall history and native GAE; normalized advantages are not uniformly increased. This is an exposure/credit observation, not a learned-behavior gain or PPO admission.',
    first_actual_rollout_tensors=comparison['actual_reward_tensors'],
    source_checkpoint=str(source), source_sha256=digest(source), native_initial_sha256=digest(native / 'model_initial.pt'),
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'followup_credit_native_audit001.json', out)
assert all(checks.values()), checks
write_json(R / 'diagnosis046_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1',
    queue='046', all_frozen_sources_verified=True, source_files=len(plan['source_sha256']),
    audit_sha256=digest(R / 'followup_credit_native_audit001.json'), new_PPO=0,
    qualification=None, created_unix=time.time()))
print(json.dumps(dict(checks=checks, credit_rows=credit_rows), indent=2))
