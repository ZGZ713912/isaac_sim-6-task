"""Coarse training preserves physical units and never qualifies 2ms contact."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from train_prone_motion import configuration, learner_config, source_hashes
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.getup_feedback import GetupFeedback
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options
from wheeled_tasks.chassis.robust_getup import RobustDomain, settings

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4.json').read_text())
OLD = json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_1000hz_20261101.json').read_text())


@pytest.mark.parametrize('hz', [200, 1000])
@pytest.mark.parametrize('profile,strength', [('nominal', 0.), ('requested', 1.)])
def test_whole_configuration_authenticates_clock_and_keeps_policy_horizon(tmp_path, hz, profile, strength):
    args = train_parser().parse_args(['--physics-hz', str(hz), '--output', str(tmp_path)])
    _, plan, _ = training_options(args, ROOT)
    task.validate(plan)
    cfg, low, _ = configuration(tmp_path, plan, profile=profile, strength=strength)
    control = load_chassis_control(ROOT/cfg['control_math_source'])
    rates = task.timing(plan)
    assert rates == control['timing']
    assert rates['policy_dt'] == .01 and rates['physics_dt'] == rates['pc_control_dt'] == 1./hz
    assert rates['decimation'] == hz//100
    assert cfg['num_steps_per_env'] == 64 and 64*rates['policy_dt'] == .64
    assert learner_config(plan) == learner_config(OLD)
    assert control['actuators'] == load_chassis_control(ROOT/OLD['wheel_model']['control']['file'])['actuators']
    saved = json.loads((tmp_path/'task_contract.json').read_text())
    assert saved['physics_hz'] == hz and saved['fine_contact_tolerance_resolved'] == (hz == 1000)
    assert low['sensor_policy_dt'] == .01 and low['pd_feedback_period_ms'] == [1,1]
    assert cfg['position_iterations'] == 64 and cfg['velocity_iterations'] == 32
    assert cfg['leg_motor_friction'] == OLD['leg_motor_friction']
    frozen = source_hashes(ROOT/'contracts/wheel_leg_prone_motion_v4.json', plan)
    assert frozen[cfg['control_math_source']] == cfg['control_math_sha256']


def test_default_is_200hz_and_historical_embedded_run_options_still_replay():
    task.validate(PLAN)
    assert task.timing(PLAN)['physics_dt'] == .005
    assert task.timing(task.validate(OLD))['physics_dt'] == .001
    old_resolved = resolve_plan(OLD)
    del old_resolved['runtime_cli']['options']['physics_hz']
    assert task.validate(old_resolved) == old_resolved
    for bad in [0, 100, 500, True]:
        with pytest.raises(ValueError, match='physics-hz'):
            resolve_plan(PLAN, physics_hz=bad)
    wrong = deepcopy(PLAN)
    wrong['simulation_timing_approximation']['hardware_equivalence_verified'] = True
    with pytest.raises(ValueError, match='approximation'):
        task.validate(wrong)


@pytest.mark.parametrize('dt', [.001, .005])
def test_torque_delay_preserves_milliseconds_and_impulse_and_reset(dt):
    cfg = settings(domains_enabled=True, nominal_fraction=0., motor_strength_range=[1.,1.],
        torque_delay_ms=[3,3], motor_tau_ms=[0.,0.], sensor_policy_dt=.01, sensor_delay_policy_ticks=[1,1])
    cfg.update(torque_delay_ms=[3,3], sensor_delay_policy_ticks=[1,1])
    domain = RobustDomain(2, 'cpu', cfg, 12, dt)
    sensors = torch.zeros(2,18); sensors[:,5] = -1.
    domain.reset(torch.arange(2), sensors)
    actual_delay = (domain.torque_delay+domain.torque_delay_fraction)*dt*1000.
    torch.testing.assert_close(actual_delay, torch.full((2,), 3.))
    outputs = []
    for tick in range(6):
        outputs.append(domain.apply_torque(torch.full((2,6),float(tick == 0)), torch.zeros(2,dtype=torch.bool)).clone())
    torch.testing.assert_close(torch.stack(outputs).sum(0)*dt, torch.full((2,6),dt))
    if dt == .005:
        torch.testing.assert_close(outputs[0], torch.full((2,6),.4))
        torch.testing.assert_close(outputs[1], torch.full((2,6),.6))
    else:
        assert all(torch.count_nonzero(v) == 0 for v in outputs[:3])
        torch.testing.assert_close(outputs[3], torch.ones(2,6))
    assert domain.summary()['sensor_delay_ms_max'] == 10.
    domain.reset(torch.tensor([0]), sensors[:1])
    assert torch.count_nonzero(domain.torque_history[:,0]) == 0


def test_200hz_encoder_delay_is_fractional_not_three_five_ms_ticks():
    cfg = dict(pd_feedback_period_ms=[1,1],pd_feedback_delay_ms=[3,3],pd_encoder_bias_rad=0.)
    feedback = GetupFeedback(2,'cpu',torch.tensor([True,False]),cfg,torch.Generator().manual_seed(3),dt=.005)
    feedback.reset(torch.arange(2),torch.zeros(2,6),torch.zeros(2,6))
    assert feedback.period.tolist() == [1,1]
    result = []
    for tick in range(1,4):
        result.append(feedback.observe(torch.full((2,6),float(tick)),torch.zeros(2,6))[0].clone())
    torch.testing.assert_close(torch.stack(result)[:,0,0], torch.tensor([.4,1.4,2.4]))
    torch.testing.assert_close(torch.stack(result)[:,1,0], torch.tensor([1.,2.,3.]))


def test_feedback_period_is_in_physical_time():
    cfg = dict(pd_feedback_period_ms=[10,10],pd_feedback_delay_ms=[0,0],pd_encoder_bias_rad=0.)
    feedback = GetupFeedback(1,'cpu',torch.tensor([True]),cfg,torch.Generator().manual_seed(3),dt=.005)
    feedback.reset(torch.arange(1),torch.zeros(1,6),torch.zeros(1,6))
    values = [feedback.observe(torch.full((1,6),float(tick)),torch.zeros(1,6))[0][0,0].item() for tick in range(4)]
    assert values == [0.,0.,2.,2.]  # 10ms, not ten5ms ticks.


def test_coarse_contact_reports_real_ms_and_refuses_fine_acceptance():
    from review_standing_acceptance import support_review
    values = np.ones(20,dtype=bool); values[3] = False
    row = support_review(values,standing_start_sample=0,sample_dt_s=.005)
    assert row['raw_missing_ms'] == row['retained_loss_ms'] == 5.
    assert row['evaluated_physical_ms'] == 100.
    assert not row['temporal_resolution_sufficient'] and not row['support_pass']
    assert not support_review(np.ones(20,dtype=bool),standing_start_sample=0,sample_dt_s=.005)['support_pass']


def test_cold_200hz_trace_cannot_be_misreported_as_one_ms_contact():
    from test_prone_motion_timing import trace_at_100hz
    from evaluate_prone_motion import review_arrays
    arrays, case = trace_at_100hz()
    for key in ('wheel_force','wheel_clearance','motor_velocity','motor_effort'):
        arrays[key] = arrays[key][:,4::5]
    arrays['wheel_force'][100,0,0] = 0.
    row = review_arrays(arrays,[case],np.zeros((1,2)),PLAN)[0]
    assert row['support']['raw_missing_ms'] == 5.
    assert not row['engineering_screen_pass'] and not row['full_acceptance_qualified']
