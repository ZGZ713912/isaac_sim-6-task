"""CPU ABI, measured admission, and physical-clock history regression checks."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.actuator_response import ActuatorResponse, JOINT_NAMES
from wheeled_tasks.chassis.controller_handoff import (
    CONTROL_JOINT_NAMES, RECOVERY_POLICY_JOINT_NAMES, STANDING_POLICY_JOINT_NAMES,
    RecoveryHandoffGate, map_motor_vector,
)
from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
from wheeled_tasks.chassis.physical_time_drive import PhysicalTimeWheelResponse, set_motor_response_dt

ROOT = Path(__file__).resolve().parents[1]


def test_formal_contract_pins_both_abis_and_does_not_grant_behavior_admission():
    cfg = json.loads((ROOT/'contracts/wheel_leg_recovery_handoff_evaluation_v1.json').read_text())
    assert cfg['training'] is False and cfg['new_PPO'] == 0
    assert tuple(cfg['recovery']['policy_joint_names']) == RECOVERY_POLICY_JOINT_NAMES
    assert tuple(cfg['standing']['policy_joint_names']) == STANDING_POLICY_JOINT_NAMES
    assert (cfg['standing']['physics_hz'], cfg['standing']['pd_hz'], cfg['standing']['policy_hz']) == (200, 200, 50)
    assert cfg['standing']['leg_pd'] == [60., 2.] and cfg['standing']['leg_action_scale_rad'] == .5
    assert cfg['standing']['gas_spring_feedforward'] is False
    assert cfg['standing']['physical_gas_spring_retained'] is True
    assert cfg['capture']['contiguous_seconds'] == .3
    assert cfg['admission']['CPU_tests_are_behavior_qualification'] is False
    assert cfg['admission']['actual_learned_recovery_and_continuous_handoff_required'] is True


def wheel(cls, dt=.001):
    path = ROOT/'model/wheel_identified_1562_20261028/model.json'
    cfg = dict(enabled=True, model_path=str(path.relative_to(ROOT)),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), apply_aggregate_delay=True)
    return cls(2, 'cpu', dt, cfg, root=ROOT, dtype=torch.float64)


@pytest.mark.parametrize('dt', [.001, .002, .005])
def test_physical_clock_matches_original_response_on_every_fixed_grid(dt):
    original, continuous = wheel(IdentifiedWheelResponse, dt), wheel(PhysicalTimeWheelResponse, dt)
    random = torch.Generator().manual_seed(231)
    for i in range(80):
        command = torch.randn(2, 6, generator=random, dtype=torch.float64)*8
        velocity = torch.randn(2, 6, generator=random, dtype=torch.float64)*70
        before = command.clone()
        actual = continuous.apply_torque(command, velocity)
        expected = original.apply_torque(command, velocity)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(continuous.passive_effort, original.passive_effort)
        torch.testing.assert_close(continuous.drive_bound, original.drive_bound)
        assert torch.equal(command, before)
        if i == 40:
            original.reset(torch.tensor([True, False]))
            continuous.reset(torch.tensor([True, False]))


def test_time_step_change_preserves_nonzero_pre_handoff_wheel_history():
    plant = wheel(PhysicalTimeWheelResponse)
    command, velocity = torch.zeros(2, 6, dtype=torch.float64), torch.zeros(2, 6, dtype=torch.float64)
    command[:, [2, 5]] = command.new_tensor([2., -3.])
    for _ in range(12):
        plant.apply_torque(command, velocity)
    before, identity = deepcopy(plant.history_snapshot()), id(plant)
    plant.set_dt(.005)
    assert plant.history_snapshot() == before and id(plant) == identity
    actual = plant.apply_torque(torch.zeros_like(command), velocity)
    # Analytic command interpolation at t=17ms: left reads 11.5ms,
    # right reads 13.5ms, retaining 70% of the old -3Nm command.
    torch.testing.assert_close(actual[:, 2], torch.full((2,), 2., dtype=torch.float64))
    torch.testing.assert_close(actual[:, 5], torch.full((2,), -2.1, dtype=torch.float64))
    assert plant.reset_calls == 0


def test_selected_reset_does_not_clear_other_lanes_or_rewind_physical_time():
    plant = wheel(PhysicalTimeWheelResponse)
    command = torch.ones(2, 6, dtype=torch.float64)
    velocity = torch.zeros_like(command)
    for _ in range(12):
        plant.apply_torque(command, velocity)
    before = plant.clock
    plant.reset(torch.tensor([0]))
    assert plant.clock == before
    for _, value in plant.samples:
        assert torch.count_nonzero(value[0]) == 0
        assert torch.equal(value[1], torch.ones(2, dtype=torch.float64))


def test_full_reset_primes_zero_at_current_physical_time_after_grid_change():
    plant = wheel(PhysicalTimeWheelResponse)
    command = torch.ones(2, 6, dtype=torch.float64)*2.
    velocity = torch.zeros_like(command)
    for _ in range(12):
        plant.apply_torque(command, velocity)
    plant.set_dt(.005)
    plant.apply_torque(command, velocity)
    plant.reset()
    assert plant.zero_prime_time == plant.clock
    plant.set_dt(.001)
    for _ in range(3):
        assert torch.count_nonzero(plant.apply_torque(command, velocity)[:, [2, 5]]) == 0
    actual = plant.apply_torque(command, velocity)
    torch.testing.assert_close(actual[:, [2, 5]], torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64))


def motor_response():
    cfg = dict(enabled=True, schema='chassis-actuator-response', version=1,
        joint_names=list(JOINT_NAMES), gain=1., armature=[0.]*6,
        frictionloss=[0.]*6, damping=[0.]*6,
        time_constant_s=[.002, .004, 0., .006, .008, 0.])
    return ActuatorResponse(2, 'cpu', .001, cfg, dtype=torch.float64)


def test_motor_lag_change_keeps_state_wrapper_and_exact_physical_time_response():
    plant = motor_response()
    command = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    for _ in range(7):
        plant.apply_torque(command)
    before = plant.state.clone()
    original_apply = plant.apply_torque
    calls = []
    def wrapped(value):
        calls.append(1)
        return original_apply(value)
    plant.apply_torque = wrapped
    set_motor_response_dt(plant, .005)
    assert torch.equal(plant.state, before) and plant.apply_torque is wrapped
    plant.apply_torque(command)
    coefficients = command.new_tensor([1.-math.exp(-.012/tau) if tau else 1.
        for tau in plant.cfg['time_constant_s']])
    torch.testing.assert_close(plant.state, command*coefficients, rtol=1e-12, atol=1e-12)
    assert len(calls) == 1


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf'), True, '.005'])
def test_invalid_interval_cannot_change_existing_drive_state(invalid):
    plant = wheel(PhysicalTimeWheelResponse)
    lag = motor_response()
    before, lag_before = plant.dt, lag.dt
    with pytest.raises(ValueError):
        plant.set_dt(invalid)
    with pytest.raises(ValueError):
        set_motor_response_dt(lag, invalid)
    assert plant.dt == before and lag.dt == lag_before


def test_policy_vector_names_prevent_interchanging_the_two_35d_abis():
    # All values identify the same motor axes, despite their policy positions.
    recovery = torch.tensor([[10., 20., 30., 40., 50., 60.]])
    standing = torch.tensor([[10., 30., 20., 40., 50., 60.]])
    expected = torch.tensor([[10., 20., 50., 30., 40., 60.]])
    assert torch.equal(map_motor_vector(recovery, RECOVERY_POLICY_JOINT_NAMES, CONTROL_JOINT_NAMES), expected)
    assert torch.equal(map_motor_vector(standing, STANDING_POLICY_JOINT_NAMES, CONTROL_JOINT_NAMES), expected)
    assert not torch.equal(map_motor_vector(recovery, STANDING_POLICY_JOINT_NAMES, CONTROL_JOINT_NAMES), expected)
    with pytest.raises(ValueError):
        map_motor_vector(torch.zeros(1, 35), RECOVERY_POLICY_JOINT_NAMES, CONTROL_JOINT_NAMES)
    with pytest.raises(ValueError):
        map_motor_vector(recovery, ['L_joint1']*6, CONTROL_JOINT_NAMES)


def sensors():
    return dict(completed_event=torch.zeros(2, dtype=torch.bool), recovered=torch.ones(2, dtype=torch.bool),
        stable=torch.ones(2, dtype=torch.bool), terminated=torch.zeros(2, dtype=torch.bool),
        gravity=torch.tensor([[0., 0., -1.]]).expand(2, -1).clone(),
        gyro=torch.zeros(2, 3), control_velocity=torch.zeros(2, 6), dt=.01)


def test_nominal_standing_or_reward_time_cannot_unlock_without_actual_completion():
    gate, values = RecoveryHandoffGate(2, 'cpu'), sensors()
    for _ in range(1000):
        assert not gate.update(**values).any()
    assert not gate.completed.any()


def test_completion_requires_contiguous_quiet_and_original_stable_predicate():
    gate, values = RecoveryHandoffGate(2, 'cpu'), sensors()
    values['completed_event'][0] = True
    assert not gate.update(**values).any()
    values['completed_event'].zero_()
    for _ in range(28):
        assert not gate.update(**values).any()
    assert torch.equal(gate.update(**values), torch.tensor([True, False]))
    values['stable'][0] = False
    assert not gate.update(**values).any()
    assert gate.quiet_seconds[0] == 0.


@pytest.mark.parametrize('kind', ['gravity', 'gyro', 'leg_speed', 'wheel_speed', 'nan', 'termination'])
def test_loss_of_capture_conditions_resets_the_measured_quiet_window(kind):
    gate, values = RecoveryHandoffGate(2, 'cpu'), sensors()
    values['completed_event'].fill_(True)
    for _ in range(29):
        gate.update(**values)
    values['completed_event'].zero_()
    if kind == 'gravity': values['gravity'][:, 2] = 0.
    if kind == 'gyro': values['gyro'][:, 0] = .2
    if kind == 'leg_speed': values['control_velocity'][:, 3] = .3
    if kind == 'wheel_speed': values['control_velocity'][:, 5] = .5
    if kind == 'nan': values['gravity'][:, 1] = float('nan')
    if kind == 'termination': values['terminated'].fill_(True)
    assert not gate.update(**values).any()
    assert torch.count_nonzero(gate.quiet_seconds) == 0
    if kind == 'termination':
        values = sensors()
        for _ in range(40):
            assert not gate.update(**values).any()


@pytest.mark.parametrize('invalid', [0., -.01, .03, float('nan'), True])
def test_invalid_gate_interval_rejected_before_mutation(invalid):
    gate, values = RecoveryHandoffGate(2, 'cpu'), sensors()
    values['completed_event'].fill_(True)
    values['dt'] = invalid
    with pytest.raises(ValueError):
        gate.update(**values)
    assert not gate.completed.any()
