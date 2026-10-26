"""A reward-only deadband contrast; no PPO, policy or optimizer modification."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_tasks.chassis.quiet_standing import standing_position_cost

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = B / 'low_wheel_diagonal_restoring_probe001/model_probe.pt'
original = torch.load(source, map_location='cpu', weights_only=False)
assert original['infos']['settings']['quiet_sway_m'] == .05
assert original['infos']['settings']['standing_early_position_weight'] == 6.
assert original['infos']['settings']['quiet_position_weight'] == 0.
old, _ = load_mean(source)
rows = []
for name, sway in [('control', .05), ('tight', .02)]:
    state = deepcopy(original)
    state['infos']['settings']['quiet_sway_m'] = sway
    state['infos']['quiet_position_reward_revision'] = dict(
        schema='quiet-deadband-reward-only-v1', source=str(source.resolve()),
        source_sha256=digest(source), old_sway_m=.05, new_sway_m=sway,
        actor_critic_Adam_RNG_clock='exact', new_PPO=0,
        scope='training position penalty only; standing acceptance and runtime inputs unchanged')
    state['infos']['hardware_ready'] = False
    target = B / ('quiet_reward_' + name + '001') / 'model_probe.pt'
    target.parent.mkdir(parents=True, exist_ok=False)
    torch.save(state, target)
    new, _ = load_mean(target)
    assert exact_state_equal(old.state_dict(), new.state_dict())
    assert exact_state_equal({k:v for k,v in original.items() if k != 'infos'},
                            {k:v for k,v in state.items() if k != 'infos'})
    rows.append(dict(name=name, checkpoint=str(target), sha256=digest(target), sway_m=sway))
radius = torch.tensor([0., .01, .02, .03, .04, .05, .055, .10, .20])
costs = {str(s): (6. * standing_position_cost(radius, s)).tolist() for s in (.05, .02)}
write_json(R / 'quiet_reward_preflight001.json', dict(
    schema='quiet-reward-deadband-preflight-v1', source=str(source), source_sha256=digest(source),
    checkpoints=rows, checks=dict(all_non_metadata_state_exact=True, actor_function_exact=True,
        no_new_inputs_or_actions=True, original_acceptance_unchanged=True),
    position_radius_m=radius.tolist(), position_cost_density=costs,
    hypothesis='A smaller reward deadband supplies displacement credit before the unchanged5cm acceptance limit.',
    next_gate='Matched native21 standing trajectories, sensors, physics and actions exact; reward difference only at standing position cost, before any PPO admission.',
    new_PPO=0, actual_experiment_consumed=2605, qualification=None,
    hardware_started=False, created_unix=time.time()))
print(json.dumps(dict(checkpoints=rows, position_cost_density=costs), indent=2))
