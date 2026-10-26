"""Independently perturb encoders through the actual frozen actor and V5 PD."""
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
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.v5_control import V5Control
from wheeled_tasks.chassis.paired_feedback import paired_position_error
from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL, CONTROL_FROM_POLICY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.set_num_threads(2)
OUTPUT = Path(__file__).with_name('scratch_feedback_torque007_validation001.json')
assert not OUTPUT.exists()
diagnostic_path = Path(__file__).with_name('scratch_trajectory_feedback007_read001.json')
diagnostic = json.loads(diagnostic_path.read_text())
checkpoint_path = Path(diagnostic['checkpoint'])
assert sha(checkpoint_path) == diagnostic['checkpoint_sha256']
run = checkpoint_path.parent
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
infos = checkpoint['infos']
assert infos['training_origin'] == 'scratch' and infos['completed_updates'] == 512
for name in (
    'src/wheeled_tasks/chassis/v5_control.py',
    'src/wheeled_tasks/chassis/paired_feedback.py',
    'src/wheeled_tasks/chassis/paired_leg_targets.py',
    'src/wheeled_tasks/chassis/scut_observation.py',
    'src/wheeled_tasks/v40/core.py',
):
    assert sha(ROOT / name) == sha(run / 'source_snapshot' / name) == infos['source_hashes'][name]
cfg = json.loads((run / 'config/contract.json').read_text())
assert sha(run / 'config/contract.json') == infos['mechanism_config_sha256']
prior_path = ROOT / cfg['control_math_source']
assert sha(prior_path) == cfg['control_math_sha256']
prior = json.loads(prior_path.read_text())
manifest_path = ROOT / cfg['asset_directory'] / 'manifest.json'
assert sha(manifest_path) == cfg['asset_manifest_sha256']
manifest = json.loads(manifest_path.read_text())
task = json.loads((run / 'task_contract.json').read_text())['settings']
agent_path = run / 'agent_config.json'
assert sha(agent_path) == infos['agent_config_sha256']
agent = json.loads(agent_path.read_text())
actor_config = dict(agent['actor'])
assert actor_config.pop('class_name') == 'wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel'
model_name = 'src/wheeled_algo/wheel_velocity_increment.py'
model_path = run / 'source_snapshot' / model_name
assert sha(model_path) == infos['source_hashes'][model_name]
spec = importlib.util.spec_from_file_location('verified007_actor', model_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
actor = module.WheelVelocityIncrementModel(
    TensorDict(dict(policy=torch.zeros(1, 35)), batch_size=[1]), agent['obs_groups'], 'actor', 6, **actor_config)
actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
actor.eval()

# Only decode and motor_efforts are executed. Their fields come from the
# authenticated actual contracts; no scene, springs, optimizer or trainer exists.
control = object.__new__(V5Control)
control.settings = dict(cfg['v5_control'])
control.nominal = torch.tensor([manifest['nominal_joint_pos'][name] for name in V5Control.ACTIVE])
leg_clip, wheel_clip = task['leg_action_clip'], task['wheel_action_clip']
control.action_bounds = torch.tensor([leg_clip, leg_clip, wheel_clip, leg_clip, leg_clip, wheel_clip])
control.wheel_prior = prior['actuators']['wheel']
evaluation = ROOT / 'runs/scratch_standing_20261026/eval007_final512_feedback002'
receipt_path = evaluation / 'receipt.json'
receipt = json.loads(receipt_path.read_text())
assert sha(receipt_path) == diagnostic['evaluation_receipt_sha256']
feedback_path = evaluation / 'evaluation/feedback_trace.npz'
assert sha(feedback_path) == diagnostic['feedback_trace_sha256']
rows = []
with np.load(feedback_path) as feedback:
    for row in diagnostic['rows']:
        lane = next(i for i, case in enumerate(receipt['evaluation']['cases'])
                    if case['height_m'] == .305 and case['pitch_deg'] == row['pitch_deg'])
        index = round(row['decision_time_s'] / .005)
        frame = torch.tensor(feedback['observation_policy'][index, lane])
        position = torch.tensor(feedback['initial_motor_position_control'][lane] if index == 0
                                else feedback['motor_position_control'][index - 1, lane])[None]
        dq = frame[16:22][list(CONTROL_FROM_POLICY)][None] * 10.
        measured_delta = torch.atan2((position - control.nominal).sin(),
                                     (position - control.nominal).cos())[0, list(POLICY_FROM_CONTROL)][:4]
        assert torch.allclose(measured_delta, frame[10:14], atol=1e-6, rtol=0.)

        def requested_torque(value):
            action = actor(TensorDict(dict(policy=value[None]), batch_size=[1]))
            legs, wheels, clipped = control.decode(action[:, list(CONTROL_FROM_POLICY)], position)
            aligned_position = position.clone()
            aligned_position[:, list(control.LEGS)] = legs - paired_position_error(legs, position[:, list(control.LEGS)])
            velocity = value[16:22][list(CONTROL_FROM_POLICY)][None] * 10.
            torque = control.motor_efforts(aligned_position, velocity, legs, wheels)
            # This check covers action clipping and paired target projection.
            wanted = control.nominal[list(control.LEGS)] + clipped[:, list(control.LEGS)] * .25
            assert torch.allclose(legs, wanted, atol=1e-6, rtol=0.)
            assert bool((torque.abs() < control.current_motor_bounds - .01).all())
            assert torch.allclose(torque, control.requested_motor_effort, atol=1e-6, rtol=0.)
            return torque[0, list(POLICY_FROM_CONTROL)].detach()

        native_output = actor(TensorDict(dict(policy=frame[None]), batch_size=[1]))[0]
        assert torch.allclose(native_output, torch.tensor(feedback['action_raw_policy'][index, lane]), atol=2e-6, rtol=0.)
        torque = requested_torque(frame)
        difference = torch.empty(6, 6)
        for axis in range(6):
            plus, minus = frame.clone(), frame.clone()
            plus[16 + axis] += .0001
            minus[16 + axis] -= .0001
            difference[:, axis] = (requested_torque(plus) - requested_torque(minus)) / .002
        leg_expected = torch.tensor(row['leg_command_torque_velocity_gain_nm_per_rad_s'])
        wheel_expected = torch.tensor(row['wheel_command_torque_velocity_gain_nm_per_rad_s'])
        leg_error = float((difference[:4, :4] - leg_expected).abs().max())
        wheel_error = float((difference[4:, 4:] - wheel_expected).abs().max())
        assert leg_error < .004 and wheel_error < .001
        rows.append(dict(
            pitch_deg=row['pitch_deg'], decision_time_s=row['decision_time_s'],
            leg_torque_derivative_max_error=leg_error, wheel_torque_derivative_max_error=wheel_error,
            finite_difference_leg_max_symmetric_eigenvalue=float(torch.linalg.eigvalsh(
                (difference[:4, :4] + difference[:4, :4].T) * .5).max()),
            finite_difference_wheel_max_symmetric_eigenvalue=float(torch.linalg.eigvalsh(
                (difference[4:, 4:] + difference[4:, 4:].T) * .5).max()),
            actual_V5_policy_order_torque_nm=torque.tolist(),
        ))
assert len(rows) == 68
report = dict(
    schema='scratch-feedback-actual-V5-derivative-verification-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(), diagnostic=str(diagnostic_path),
    diagnostic_sha256=sha(diagnostic_path), checkpoint_sha256=sha(checkpoint_path), rows=rows,
    all_actual_torque_derivative_checks_pass=True, no_action_clipping_or_paired_projection_active=True,
    all_perturbed_torques_unsaturated=True,
    maximum_leg_derivative_error=max(row['leg_torque_derivative_max_error'] for row in rows),
    maximum_wheel_derivative_error=max(row['wheel_torque_derivative_max_error'] for row in rows),
    reader_sha256=sha(__file__), new_PPO=0, native_steps=0, optimizer_constructed=False, hardware_ready=False,
    scope='Independent finite differences through the actual frozen actor and authenticated V5 decode/paired-error/PD. '
          'Other observations and physical positions remain fixed. These are decision-boundary torque derivatives, '
          'not closed-loop eigenvalues or proof of the first contact-loss cause.',
)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'scope')}))
