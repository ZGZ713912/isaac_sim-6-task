"""Read back the completed native zero-PPO preflight before any new optimizer work."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import torch
from train_robust_getup import digest, write_json
from run_gap_long_training import verify_restore
from export_robust_getup import load_mean
from wheeled_algo.periodic_entry_actor import exact_state_equal

R = ROOT / 'reports/robust_policy_repair_20261024'
B = ROOT / 'runs/robust_policy_repair_20261024'
plan = json.loads((R / 'diagnosis_plan039.json').read_text())
queue = json.loads((R / 'diagnosis_queue039.json').read_text())
assert queue['status'] == 'completed'
assert all(j['status'] == 'completed' and j['returncode'] == 0 for j in queue['jobs'])
source = B / 'fallen_output_root001/model_initial.pt'
native = B / 'diagnosis039_fallen_output_native_zero'
verify_restore(source, native / 'model_initial.pt', R / 'fallen_native_zero_restore001.json')
a, b = [torch.load(p, map_location='cpu', weights_only=False)
        for p in (source, native / 'model_initial.pt')]
restore = json.loads((R / 'fallen_native_zero_restore001.json').read_text())
progress = json.loads((native / 'progress.json').read_text())
rollout = json.loads((native / 'rollout_diagnosis.json').read_text())
function = json.loads((native / 'native_function_readback.json').read_text())
changed = [n for n, h in plan['source_sha256'].items() if digest(ROOT / n) != h]
mean, state = load_mean(native / 'model_initial.pt')
load_mean(native / 'model_stopped.pt')
checks = dict(
    native_queue_completed=True,
    frozen_inputs=not changed,
    restore_actor_critic_Adam_and_four_RNGs=all(restore['checks'].values()),
    optimizer_including_LRs_exact=exact_state_equal(a['optimizer_state_dict'], b['optimizer_state_dict']),
    authenticated_function_ancestry=exact_state_equal(a['infos']['policy_function_root'], b['infos']['policy_function_root']),
    original_mean_and_conditioned_std=all(function['checks'].values()) and function['frames'] == 131072
        and function['mean_different'] == function['std_different'] == 0,
    exactly774_trainable_actor_parameters=progress['trainable_actor_parameters'] == 774,
    zero_PPO=progress['new_optimizer_updates'] == progress['new_optimizer_attempts']
        == rollout['optimizer_attempts'] == rollout['optimizer_steps'] == rollout['new_PPO'] == 0,
    clock1501=progress['completed_updates'] == b['infos']['completed_updates'] == 1501,
    stopped_before_attempt=rollout['status'] == 'captured_before_optimizer_attempt'
        and progress['status'] == 'stopped' and not progress['optimizer_update_in_flight'],
    original_native_sampling_and_GAE=rollout['native_GAE'] and rollout['native_sampling_and_likelihood'],
    recorded_data_SHA=digest(ROOT / rollout['data']) == rollout['data_sha256'],
    stopped_checkpoint_is_exact_initial=digest(native / 'model_stopped.pt') == digest(native / 'model_initial.pt'),
    strict_native_initial_and_stopped_load=True,
)
out = dict(schema='native-fallen-output-zero-PPO-audit-v1', checks=checks,
    source_checkpoint=str(source), source_sha256=digest(source),
    native_initial_sha256=digest(native / 'model_initial.pt'),
    frozen_inputs=len(plan['source_sha256']), changed_inputs=changed,
    valid_transitions=rollout['valid_rollout']['valid_transitions'],
    passive_placement_masked_transitions=rollout['valid_rollout']['script_or_blend_transitions'],
    phase='Real rollout and native GAE completed before any optimizer attempt; no physical state continuation claimed.',
    new_PPO=0, actual_experiment_consumed=2630, repair_optimizer_updates=250,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'fallen_native_zero_audit001.json', out)
assert all(checks.values()), out
write_json(R / 'diagnosis039_closed_source_seal001.json', dict(
    schema='closed-native-queue-source-seal-v1', queue='039', all_frozen_sources_verified=True,
    source_files=len(plan['source_sha256']), audit_sha256=digest(R / 'fallen_native_zero_audit001.json'),
    new_PPO=0, qualification=None, created_unix=time.time()))
print(json.dumps(out, indent=2))
