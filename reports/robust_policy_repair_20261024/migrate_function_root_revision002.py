"""Repair only V3 metadata; retain both real PPO updates and their full state."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import torch
from export_robust_getup import load_mean
from wheeled_algo.function_root_checkpoint import MODULE, ENVIRONMENT_MODULE, VERSION, digest, make_progress, environment_revision
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'pilot009_periodic_motion_entry/model_stopped.pt'
old_initial_path = B / 'function_root_entry001/model_initial.pt'
new_initial_path = B / 'function_root_entry002/model_initial.pt'
original = torch.load(source, map_location='cpu', weights_only=False)
old_initial = torch.load(old_initial_path, map_location='cpu', weights_only=False)
new_initial = torch.load(new_initial_path, map_location='cpu', weights_only=False)
root = torch.load(original['infos']['training_function_root']['checkpoint'], map_location='cpu', weights_only=False)
archive = R / 'function_root_failure_sources001' / MODULE
assert digest(archive) == original['infos']['source_sha256'][MODULE]
spec = importlib.util.spec_from_file_location('wheeled_algo._original_function_root_contract', archive)
old_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_contract)
normalized_for_audit = deepcopy(original)
normalized_for_audit['infos']['source_sha256'][ENVIRONMENT_MODULE] = root['infos']['source_sha256'][ENVIRONMENT_MODULE]
old_contract.validate_state(normalized_for_audit, root, old_initial)
assert exact_state_equal({k: v for k, v in old_initial.items() if k != 'infos'},
                         {k: v for k, v in new_initial.items() if k != 'infos'})
assert all(exact_state_equal(old_initial['infos'][key], new_initial['infos'][key])
           for key in ('environment_rng', 'domain_rng', 'torch_rng', 'cuda_rng', 'completed_updates'))
result = deepcopy(original)
info = result['infos']
info['source_sha256'][MODULE] = digest(Path(MODULE))
info['training_function_root']['schema'] = VERSION
count = original['infos']['function_root_learning']['new_optimizer_updates']
assert count == 2 and info['completed_updates'] == 1503
info['function_root_learning'] = make_progress(new_initial_path, new_initial, result['actor_state_dict'], count)
info['function_root_environment_revision'] = environment_revision(root['infos']['source_sha256'], info['source_sha256'], info['settings'])
info['function_root_contract_revision'] = dict(schema='entry-function-root-metadata-revision-v1',
    source_checkpoint=str(source.resolve()), source_sha256=digest(source),
    previous_contract_source=str(archive.resolve()), previous_contract_sha256=digest(archive),
    actual_training_source=str(old_initial_path.resolve()), actual_training_source_sha256=digest(old_initial_path),
    equivalent_source=str(new_initial_path.resolve()), equivalent_source_sha256=digest(new_initial_path),
    source_policy_critic_Adam_RNG_clock_exact=True, all_saved_non_metadata_state_exact=True,
    retained_completed_PPO_updates=count, new_PPO=0)
output = B / 'pilot009_contract002/model_migrated.pt'
output.parent.mkdir(parents=True, exist_ok=False)
torch.save(result, output)
load_mean(output)
assert exact_state_equal({k: v for k, v in result.items() if k != 'infos'},
                         {k: v for k, v in original.items() if k != 'infos'})
receipt = dict(schema='entry-function-root-metadata-revision-receipt-v1',
    checkpoint=str(output), sha256=digest(output), source=str(source), source_sha256=digest(source),
    checks=dict(original_contract_accepts_everything_except_known_reward_source=True,
                old_and_new_initial_actor_critic_Adam_RNG_clock_exact=True,
                stopped_actor_critic_Adam_RNG_clock_unchanged=True,
                revised_decoder_authenticates=True),
    retained_PPO_updates=2, new_PPO=0, actual_experiment_consumed=2605,
    learning_clock=1503, qualification=None, hardware_started=False, created_unix=time.time())
(R / 'function_root_contract_revision002.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
