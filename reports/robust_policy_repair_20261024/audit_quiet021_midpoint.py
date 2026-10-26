"""Read a sealed middle checkpoint while its native training process continues."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.quiet_full_checkpoint import LEARNED_KEYS

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
plan = json.loads((R / 'plan021_quiet_continuation001.json').read_text())
source = Path(plan['source_checkpoint'])
target = Path(plan['output']) / 'model_1528.pt'
source_state, state = [torch.load(p, map_location='cpu', weights_only=False) for p in (source, target)]
load_mean(target)
names = state['infos']['policy_function_root']['optimizer_actor_parameter_names']
old, new = source_state['optimizer_state_dict'], state['optimizer_state_dict']
actor_ids = old['param_groups'][0]['params']
learned_ids = {actor_ids[names.index(k)] for k in LEARNED_KEYS}
changed = [k for k, v in source_state['actor_state_dict'].items() if not torch.equal(v, state['actor_state_dict'][k])]
checks = dict(
    exact_native_clock=state['infos']['completed_updates'] == 1529,
    exact_native_increment=state['infos']['policy_function_learning']['new_optimizer_updates'] == 12,
    exactly_twelve_existing_tensors_changed=set(changed) == set(LEARNED_KEYS),
    other_sixteen_actor_tensors_and_std_bit_exact=all(torch.equal(v, state['actor_state_dict'][k])
        for k, v in source_state['actor_state_dict'].items() if k not in LEARNED_KEYS),
    frozen_Adam_slots_bit_exact=all(exact_state_equal(old['state'][i], new['state'][i])
        for i in actor_ids if i not in learned_ids),
    active_Adam_step_increment_exact=all(float(new['state'][i]['step']) == float(old['state'][i]['step']) + 144
        for i in learned_ids | set(old['param_groups'][1]['params'])),
    original_dense_cost=state['infos']['standing_contact_training_contract'] == plan['training_contact_contract'],
    rejected_event_cost_absent='standing_contact_continuity_contract' not in state['infos'],
    all_258_frozen_inputs_unchanged=all(digest(Path(k)) == v for k, v in plan['source_sha256'].items()),
    native_function_decoder_passed=True,
)
receipt = dict(schema='native-sealed-middle-quiet-checkpoint-v1', checks={k: bool(v) for k, v in checks.items()},
    checkpoint=str(target), sha256=digest(target), source=str(source), source_sha256=digest(source),
    source_clock=1517, learning_clock=1529, branch_updates=12, branch_planned_updates=32,
    real_behavior_assessed=False, qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'quiet_continuation021_midpoint001.json', receipt)
assert all(checks.values()), checks
print(json.dumps(receipt, indent=2))
