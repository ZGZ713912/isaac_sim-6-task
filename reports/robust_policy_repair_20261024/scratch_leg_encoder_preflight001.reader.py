"""CPU proof of a bounded diagnostic using all original007 recorded inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from leg_encoder_response_probe import LegEncoderResponseProbe, responses, validate_scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.set_num_threads(2)
HERE = Path(__file__).parent
OUTPUT = HERE / 'scratch_leg_encoder_preflight001.json'
ARTIFACTS = HERE / 'scratch_leg_encoder_preflight001'
assert not OUTPUT.exists() and not ARTIFACTS.exists()
ARTIFACTS.mkdir()
run = ROOT / 'runs/scratch_standing_20261026/pilot007_contact_cost_zero'
evaluation = ROOT / 'runs/scratch_standing_20261026/eval007_final512_feedback002'
receipt = json.loads((evaluation / 'receipt.json').read_text())
checkpoint_path = run / 'model_final.pt'
assert sha(checkpoint_path) == receipt['checkpoint_sha256']
assert receipt['new_PPO'] == 0 and not receipt['optimizer_constructed']
feedback_path = evaluation / 'evaluation/feedback_trace.npz'
assert sha(feedback_path) == receipt['evaluation']['feedback_trace']['sha256']
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
infos = checkpoint['infos']
assert infos['training_origin'] == 'scratch' and infos['continuation_forbidden']
agent_path = run / 'agent_config.json'
assert sha(agent_path) == infos['agent_config_sha256']
agent = json.loads(agent_path.read_text())
actor_config = dict(agent['actor'])
assert actor_config.pop('class_name') == 'wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel'
name = 'src/wheeled_algo/wheel_velocity_increment.py'
model_path = run / 'source_snapshot' / name
assert sha(model_path) == infos['source_hashes'][name]
spec = importlib.util.spec_from_file_location('frozen_leg_encoder_preflight_actor', model_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
actor = module.WheelVelocityIncrementModel(
    TensorDict(dict(policy=torch.zeros(21, 35)), batch_size=[21]), agent['obs_groups'], 'actor', 6, **actor_config)
actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
actor.eval()
original_state = deepcopy(actor.state_dict())
rng_state = torch.get_rng_state().clone()
checks = {}
with np.load(feedback_path) as archive:
    frames = archive['observation_policy'].copy()
    recorded_actions = archive['action_raw_policy'].copy()
assert frames.shape == (1600, 21, 35)
probe = LegEncoderResponseProbe(actor, 0.)
max_native_error = 0.
changed_frames = 0
with torch.inference_mode():
    for index in range(1600):
        frame = torch.tensor(frames[index])
        observation = TensorDict(dict(policy=frame), batch_size=[21])
        original_frame = frame.clone()
        output = probe(observation)
        native = probe.native_actions[-1]
        max_native_error = max(max_native_error, float((native - torch.tensor(recorded_actions[index])).abs().max()))
        assert torch.equal(frame, original_frame)
        assert torch.equal(output[:, 4:], native[:, 4:])
        assert torch.equal(output[~probe.selected], native[~probe.selected])
        changed_frames += int(not torch.equal(output[probe.selected, :4], native[probe.selected, :4]))
assert max_native_error < 2e-6 and changed_frames > 1000
checks.update(all1600_native_mean_parity=True, input_immutable=True, all_native_wheels_exact=True,
              all_other_heights_exact=True, first_action_native=True, leg_response_actually_changed=True)

# Keep every other sensor fixed and vary only the four leg-speed channels.
# The zero-response diagnostic must remove that path without freezing posture feedback.
base = torch.tensor(frames[600, 7:8])
observation = TensorDict(dict(policy=base), batch_size=[1])
with torch.inference_mode():
    output, native, anchor, _ = responses(actor, observation, 0.)
    for axis in range(4):
        changed = observation.clone()
        changed['policy'][0, 16 + axis] += .005
        actual, full, _, _ = responses(actor, changed, 0.)
        assert torch.equal(actual[:, :4], output[:, :4])
        assert torch.equal(actual[:, 4:], full[:, 4:])
    for axis in (5, 10):
        changed = observation.clone()
        changed['policy'][0, axis] += .001
        actual, _, _, _ = responses(actor, changed, 0.)
        assert not torch.equal(actual[:, :4], output[:, :4])
    identity, full, _, _ = responses(actor, observation, 1.)
    assert torch.equal(identity, full)
    half, full, neutral, _ = responses(actor, observation, .5)
    assert torch.allclose(half[:, :4], .5 * (full[:, :4] + neutral[:, :4]), atol=1e-7, rtol=0.)
checks.update(direct_leg_velocity_path_removed=True, gyro_and_position_feedback_remains=True,
              scale1_is_exact_identity=True, scale_half_interpolation_correct=True)
assert torch.equal(rng_state, torch.get_rng_state())
assert all(torch.equal(value, actor.state_dict()[key]) for key, value in original_state.items())
checks['actor_parameters_normalizer_and_RNG_unchanged'] = True
for scale in (True, False, -.1, .2, 1.1, float('nan'), float('inf')):
    try:
        validate_scale(scale)
    except ValueError:
        pass
    else:
        raise AssertionError(scale)
checks['invalid_scales_rejected'] = True

logger = Logger(log_dir=str(ARTIFACTS / 'events'), cfg=agent,
    env_cfg=json.loads((run / 'config/contract.json').read_text()), num_envs=21,
    is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device='cpu')
logger.init_logging_writer()
try:
    result = probe.finish(ARTIFACTS, logger.writer, 512, sha)
finally:
    logger.writer.close()
event, = (ARTIFACTS / 'events').glob('events.out.tfevents.*')
accumulator = EventAccumulator(str(event), size_guidance={'scalars': 0}).Reload()
tags = accumulator.Tags()['scalars']
assert len(tags) == 5 and all(len(accumulator.Scalars(tag)) == 1 for tag in tags)
assert accumulator.Scalars('LegEncoderProbe/actor_calls')[0].value == 1600.
assert accumulator.Scalars('LegEncoderProbe/selected_original_cases')[0].value == 7.
checks['full_sidecar_and_actual_RSL_tags'] = True

invalid_cli = [[], ['--wheel-response-scale', '.5'], ['--hold-initial-leg-targets'],
               ['--velocity-iterations', '4'], ['--duration-seconds', '30'],
               ['--external-force-mode', 'every-iteration'], ['--reset-observation-mode', 'refresh-final-pose']]
with tempfile.TemporaryDirectory() as temporary:
    for index, extra in enumerate(invalid_cli):
        output_path = Path(temporary) / str(index)
        command = [sys.executable, str(ROOT / 'scripts/evaluate_scratch_standing.py'),
                   '--checkpoint', str(Path(temporary) / 'nonexistent.pt'), '--output', str(output_path),
                   '--leg-speed-response-scale', '0']
        if index:
            command += ['--trace-feedback']
        command += extra
        rejected = subprocess.run(command, capture_output=True, text=True)
        assert rejected.returncode != 0 and not output_path.exists()
        assert ('requires trace-feedback' if index == 0 else 'one declared counterfactual') in rejected.stderr
checks['all7_bad_CLI_combinations_rejected_before_output_and_App'] = True
sources = {}
for name in ('scripts/leg_encoder_response_probe.py', 'scripts/evaluate_scratch_standing.py'):
    source = ROOT / name
    (ARTIFACTS / source.name).write_bytes(source.read_bytes())
    sources[name] = sha(source)
assert all(checks.values())
report = dict(
    schema='scratch-leg-encoder-response-CPU-preflight-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    checks=checks, all_checks_pass=True, checkpoint_sha256=sha(checkpoint_path),
    feedback_trace_sha256=sha(feedback_path), maximum_CPU_native_action_error=max_native_error,
    diagnostic_changed_frames=changed_frames, actual_counterfactual_implementation=result,
    actual_event_sha256=sha(event), source_sha256=sources, reader_sha256=sha(__file__),
    new_PPO=0, native_steps=0, optimizer_constructed=False, native_validation=False, hardware_ready=False,
    next_native_condition='Only after original008 PID709399/start31429889 completes its own evaluation, exits GPU/process, '
                          'and is sealed. Need native baseline and identical initial/untouched-population checks before attribution.',
    scope='CPU implementation and selectivity proof on recorded baseline observations. No counterfactual physical trajectory '
          'has been simulated and no training/deployment change is adopted.',
)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('actual_counterfactual_implementation', 'scope')}))
