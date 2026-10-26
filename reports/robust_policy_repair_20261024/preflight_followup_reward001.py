"""Check reward bounds, native timer ownership, autoreset and checkpoint provenance."""
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
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.motion_output_checkpoint import make_progress
from wheeled_tasks.chassis.followup_continuity_reward import (
    MODULE, bonus_from_hold, contract, install, validate_contract)

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
outdir = R / 'followup_reward_CPU001'
outdir.mkdir(exist_ok=False)
checks = {}
hold = torch.tensor([0., .5, 1., 2., 9., 2.])
active = torch.tensor([True, True, True, True, True, False])
checks['bounded_formula_and_passive_mask'] = torch.equal(
    bonus_from_hold(hold, active, weight=5.), torch.tensor([0., .0015625, .00625, .025, .025, 0.]))
negative = 0
for weight in (-1., 5.1, float('nan'), float('inf'), True):
    try:
        bonus_from_hold(hold, active, weight=weight)
    except ValueError:
        negative += 1
checks['five_invalid_weight_negative_controls'] = negative == 5
try:
    bonus_from_hold(torch.tensor([-1.]), torch.tensor([True]), weight=5.)
except ValueError:
    checks['negative_native_timer_rejected'] = True


class NativeEnv:
    fixed_cases = None
    policy_dt = .005
    getup_cfg = {'followup_seconds': 2.}

    def __init__(self):
        self.followup_hold = torch.zeros(3)
        self.in_motion_fall = torch.zeros(3, dtype=torch.bool)
        self.reset_count = 0

    def reset(self, ids):
        self.followup_hold[ids] = 0.
        self.in_motion_fall[ids] = False
        self.reset_count += 1

    def step(self, actions):
        self.followup_hold.copy_(torch.tensor([1., 2., 2.5]))
        self.in_motion_fall.copy_(torch.tensor([False, True, True]))
        reward = torch.rand(3)
        done = torch.tensor([False, True, True])
        obs = {'policy': torch.rand(3, 35)}
        extras = {'policy_learning_mask': torch.tensor([True, True, False]), 'log': {}}
        self.last_native = (obs, reward, done, extras)
        self.reset(done.nonzero().flatten())
        return self.last_native


for weight in (0., 5.):
    env = NativeEnv()
    spec = install(env, weight)
    torch.manual_seed(8925)
    actual = env.step(torch.zeros(3, 6))
    rng = torch.get_rng_state()
    native = NativeEnv()
    torch.manual_seed(8925)
    expected = native.step(torch.zeros(3, 6))
    checks[f'weight{weight}_native_observation_done_state_RNG_exact'] = (
        exact_state_equal(actual[0], expected[0]) and torch.equal(actual[2], expected[2])
        and torch.equal(env.followup_hold, native.followup_hold)
        and torch.equal(env.in_motion_fall, native.in_motion_fall)
        and env.reset_count == native.reset_count and torch.equal(rng, torch.get_rng_state()))
    checks[f'weight{weight}_autoreset_uses_pre_reset_timer'] = torch.equal(
        env.training_reward_last['native_followup_hold'], torch.tensor([1., 2., 2.5]))
    wanted = expected[1] + torch.tensor([weight * .00125, weight * .005, 0.])
    checks[f'weight{weight}_exact_reward_only_difference'] = torch.equal(actual[1], wanted)
    if weight == 0.:
        checks['zero_weight_reward_object_identity'] = actual[1] is env.last_native[1]
    validate_contract(spec, {MODULE: digest(Path(MODULE))})
    try:
        install(env, weight)
    except ValueError:
        checks[f'weight{weight}_duplicate_install_rejected'] = True

source = B / 'motion_output_root001/model_initial.pt'
original = torch.load(source, map_location='cpu', weights_only=False)
state = deepcopy(original)
state['infos']['training_reward_contract'] = contract(5.)
state['infos']['source_sha256'][MODULE] = digest(Path(MODULE))
state['infos']['policy_function_stage'] = 'native-training'
state['infos']['policy_function_learning'] = make_progress(source, original, state['actor_state_dict'], 0)
cp = outdir / 'reward_metadata_decode.pt'
torch.save(state, cp)
before, _ = load_mean(source)
after, _ = load_mean(cp)
trace = B / 'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz'
frames = np.load(trace)['observations'].reshape(-1, 35)
with torch.inference_mode():
    checks['metadata_only_decode_function_bit_exact'] = all(
        torch.equal(before(torch.from_numpy(a.copy())), after(torch.from_numpy(a.copy())))
        for a in np.array_split(frames[::20], 8))
bad_count = 0
for key, value in [('weight_density_per_second', 6.), ('followup_seconds', 1.),
                   ('policy_dt', .01), ('module_sha256', '0' * 64), ('formula', 'different')]:
    bad = deepcopy(state)
    bad['infos']['training_reward_contract'][key] = value
    path = outdir / ('bad_' + key + '.pt')
    torch.save(bad, path)
    try:
        load_mean(path)
    except ValueError:
        bad_count += 1
checks['five_actual_decoder_tamper_controls_rejected'] = bad_count == 5
checks['original_saved_training_payload_exact'] = all(exact_state_equal(original[k], state[k])
    for k in original if k != 'infos')
for path in ('pilot015_motion_output/model_final.pt', 'quiet_stance050003/model_probe.pt',
             'pilot014_fallen_output/model_final.pt'):
    load_mean(B / path)
checks['legacy_V4_V5_V6_decoders_pass'] = True
result = dict(schema='followup-continuity-reward-CPU-v1', checks={k: bool(v) for k, v in checks.items()},
    contract=contract(5.), files_sha256={str(p): digest(p) for p in (
        Path(MODULE), Path(__file__), Path('scripts/train_robust_getup.py'), Path('scripts/export_robust_getup.py'))},
    source_checkpoint=str(source), source_sha256=digest(source),
    interpretation='CPU timer/autoreset and real decoder checks; no native exposure or learned behavior claim.',
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'followup_reward_preflight001.json', result)
print(json.dumps(result, indent=2))
assert all(checks.values()), checks
