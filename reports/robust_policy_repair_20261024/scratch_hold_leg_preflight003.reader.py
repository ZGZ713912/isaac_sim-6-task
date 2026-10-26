"""CPU action-intervention check on recorded frozen007 observations."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
torch.set_num_threads(2)
sys.path.insert(0, str(ROOT / 'scripts'))
from hold_initial_leg_target_probe import HoldInitialLegTargetProbe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


baseline = ROOT / 'runs/scratch_standing_20261026/eval007_final512_contact_state001'
receipt = json.loads((baseline / 'receipt.json').read_text())
assert receipt['status'] == 'completed_unqualified' and receipt['new_PPO'] == 0
checkpoint = Path(receipt['checkpoint'])
assert sha(checkpoint) == receipt['checkpoint_sha256']
state = torch.load(checkpoint, map_location='cpu', weights_only=False)
infos = state['infos']
assert infos['training_origin'] == 'scratch' and infos['continuation_forbidden']
run = checkpoint.parent
archive = run / 'source_snapshot'
assert all(sha(archive / name) == value for name, value in infos['source_hashes'].items())
agent = json.loads((run / 'agent_config.json').read_text())
assert sha(run / 'agent_config.json') == infos['agent_config_sha256']
assert agent['actor']['class_name'] == 'wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel'
model_file = archive / 'src/wheeled_algo/wheel_velocity_increment.py'
spec = importlib.util.spec_from_file_location('frozen007_preflight_actor', model_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = dict(agent['actor']); config.pop('class_name')
actor = module.WheelVelocityIncrementModel(TensorDict(dict(policy=torch.zeros(1, 35)), batch_size=[1]),
    agent['obs_groups'], 'actor', 6, **config)
actor.load_state_dict(state['actor_state_dict'], strict=True)
actor.eval()
initial_state = {name: value.clone() for name, value in actor.state_dict().items()}
trace = baseline / 'evaluation/feedback_trace.npz'
assert sha(trace) == receipt['evaluation']['feedback_trace']['sha256']
with np.load(trace) as recorded:
    frames, actions = recorded['observation_policy'], recorded['action_raw_policy']
assert frames.shape == (1600, 21, 35) and actions.shape == (1600, 21, 6)
probe = HoldInitialLegTargetProbe(actor)
checks = dict(recorded_native_actions_reproduce_on_CPU=True, first_action_unchanged=True,
    later_selected_leg_targets_held=True, native_wheels_unchanged=True,
    other_height_outputs_unchanged=True, later_intervention_nonzero=False)
cpu_error, maximum_intervention = 0., 0.
with torch.inference_mode():
    for step, frame in enumerate(frames):
        observation = TensorDict(dict(policy=torch.from_numpy(frame.copy())), batch_size=[21])
        native = actor(observation)
        output = probe(observation)
        cpu_error = max(cpu_error, float((native - torch.from_numpy(actions[step])).abs().max()))
        checks['native_wheels_unchanged'] &= torch.equal(output[:, 4:], native[:, 4:])
        checks['other_height_outputs_unchanged'] &= torch.equal(output[~probe.selected], native[~probe.selected])
        checks['later_selected_leg_targets_held'] &= torch.equal(output[probe.selected, :4], probe.initial_legs[probe.selected])
        if step == 0:
            checks['first_action_unchanged'] = torch.equal(output, native)
        maximum_intervention = max(maximum_intervention, float((output - native).abs().max()))
checks.update(recorded_native_actions_reproduce_on_CPU=cpu_error < 2e-6,
    all1600_calls_accounted=probe.calls == 1600 and len(probe.native_actions) == 1600,
    selected_original_cases=probe.selected.nonzero().flatten().tolist() == list(range(7, 14)),
    later_intervention_nonzero=maximum_intervention > 0.,
    actor_parameters_and_buffers_unchanged=all(torch.equal(value, actor.state_dict()[name])
        for name, value in initial_state.items()))
assert all(checks.values()), checks
sources = ('scripts/evaluate_scratch_standing.py', 'scripts/hold_initial_leg_target_probe.py',
           'tests/test_scratch_hold_initial_leg_target_probe.py')
previous = json.loads((BASE / 'scratch_hold_leg_preflight002.json').read_text())
assert all(sha(ROOT / name) == previous['source_sha256'][name] for name in sources)
frozen = BASE / 'scratch_hold_leg_preflight003'; frozen.mkdir()
for name in sources:
    (frozen / Path(name).name).write_bytes((ROOT / name).read_bytes())
report = dict(schema='scratch-hold-initial-leg-target-preflight-v1', status='prepared_not_started',
    observed_utc=datetime.now(timezone.utc).isoformat(), checks=checks, all_checks_pass=True,
    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
    baseline=str(baseline), recorded_feedback_sha256=sha(trace),
    source_sha256={name: sha(ROOT / name) for name in sources}, reader_sha256=sha(__file__),
    original_frozen_sources_verified=len(infos['source_hashes']),
    CPU_to_original_GPU_maximum_action_error=cpu_error,
    maximum_leg_intervention_raw=maximum_intervention,
    new_PPO=0, native_steps=0, optimizer_constructed=False, adoption_by_training=False,
    behavior_qualification=False, hardware_ready=False,
    prerequisite='Verify the separately launched fresh-process original-policy baseline '
                 'matches the contact observer, and verify its actual process/GPU exit before leg-hold.',
    scope='CPU replay on original recorded observations validates the action intervention only. '
          'It does not predict the altered closed-loop trajectory or certify standing.')
output = BASE / 'scratch_hold_leg_preflight003.json'; assert not output.exists()
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(checks=checks, CPU_action_error=cpu_error, maximum_leg_intervention_raw=maximum_intervention)))
