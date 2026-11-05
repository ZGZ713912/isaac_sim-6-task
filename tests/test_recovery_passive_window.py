"""Disabled placement timers cannot alter the active recovery or actor ABI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from train_prone_motion import configuration, learner_config, source_hashes
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
from wheeled_tasks.chassis.recovery_passive_window import (
    EnableObservations, configure, contract, validate)

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT/'contracts/wheel_leg_self_righting_passive_window_v1.json'
PLAN = json.loads(FILE.read_text())
BASE = json.loads((ROOT/'contracts/wheel_leg_self_righting_return_horizon_v1.json').read_text())


def test_only_explicit_passive_window_is_added_to_the_baseline():
    assert task.validate(PLAN) == PLAN
    p = deepcopy(PLAN)
    assert p.pop('recovery_passive_window') == contract()
    assert p == BASE


@pytest.mark.parametrize('change', [lambda c:c.pop('history'), lambda c:c.update(minimum_seconds=True),
    lambda c:c.update(maximum_seconds=float('nan')), lambda c:c.update(minimum_seconds=.5),
    lambda c:c.update(settled_hold_seconds=.1), lambda c:c.update(extra=False),
    lambda c:c.update(linear_velocity_and_contact='enable_conditions')])
def test_partial_unknown_or_privileged_enable_contract_is_rejected(change):
    c=contract(); change(c)
    with pytest.raises(ValueError): validate(c)


@pytest.mark.parametrize('change', [lambda p:p.pop('recovery_return_horizon'),
    lambda p:p.update(multi_pose_recovery=None), lambda p:p.update(jump_training={}),
    lambda p:p['global_curriculum'].pop('recovery_focus'),
    lambda p:p['timing'].update(policy_dt=.02)])
def test_incompatible_courses_cannot_select_the_timers(change):
    p=deepcopy(PLAN); change(p)
    with pytest.raises(ValueError): task.validate(p)


def test_configuration_changes_only_three_existing_timer_settings(tmp_path):
    opts=dict(num_envs=64,max_iterations=250,save_interval=50,seed=261105,
        physics_hz=1000,policy_hz=100,pd_hz=200,domain_randomization='off')
    new,old=(resolve_plan(p,**opts) for p in (PLAN,BASE))
    cfg,a,_=configuration(tmp_path/'new',new,device='cpu')
    old_cfg,b,_=configuration(tmp_path/'old',old,device='cpu')
    # Each output has its own design-file path and archive digest.
    for config in (cfg,old_cfg):
        binding=config.pop('design_preflight')
        assert binding['stage'] == 'speed'
        assert binding['sha256'] == hashlib.sha256(Path(binding['file']).read_bytes()).hexdigest()
    assert cfg == old_cfg
    assert learner_config(new,'cpu') == learner_config(old,'cpu')
    assert [b[k] for k in ('passive_min_seconds','passive_max_seconds','passive_hold_seconds')] == [.05,.15,.05]
    for key in ('passive_min_seconds','passive_max_seconds','passive_hold_seconds'):
        b[key]=a[key]
    a.pop('prone_motion_task'); b.pop('prone_motion_task')
    assert a == b
    assert [a[k] for k in ('passive_min_seconds','passive_max_seconds','passive_hold_seconds')] == [1.,1.2,.05]
    saved=json.loads((tmp_path/'new/task_contract.json').read_text())
    assert saved['plan']['rise_deadline_seconds'] == BASE['rise_deadline_seconds']
    hashes=source_hashes(FILE,new)
    assert hashes['src/wheeled_tasks/chassis/recovery_passive_window.py'] == hashlib.sha256(
        (ROOT/'src/wheeled_tasks/chassis/recovery_passive_window.py').read_bytes()).hexdigest()
    assert 'docs/RECOVERY_PASSIVE_WINDOW.md' in hashes


def test_runtime_validation_cannot_change_the_active_clock_or_window(tmp_path):
    resolved=resolve_plan(PLAN,num_envs=64,max_iterations=250)
    resolved['recovery_passive_window']['minimum_seconds']=0.
    with pytest.raises(ValueError): configuration(tmp_path,resolved,device='cpu')


def event(time=1.2, vz=-1., effort=0.):
    return dict(enable_event=torch.tensor([True,False]),time=torch.tensor([time,0.]),
        velocity_world=torch.tensor([[0.,0.,vz],[0.,0.,0.]]),
        omega=torch.zeros(2,3),motor_velocity=torch.zeros(2,6),
        passive=torch.ones(2,dtype=torch.bool),passive_motor_peak=torch.tensor([effort,0.]))


def test_enable_probe_is_read_only_and_never_certifies_equilibrium_or_velocity_gate():
    state=event(vz=-1.);before={k:v.clone() for k,v in state.items()}
    observations=EnableObservations();observations.record(state)
    report=observations.report(contract(),.01)
    assert all(report['checks'].values())
    assert report['events'][0]['com_world_z_velocity_m_s'] == -1.
    assert report['static_equilibrium_certified'] is report['learned_behavior'] is False
    assert all(torch.equal(before[k],v) for k,v in state.items())


@pytest.mark.parametrize('seconds,effort,check',[(.9,0.,'minimum_disabled_window_respected'),
    (1.22,0.,'original_maximum_time_bounds_enable'),(1.,.01,'enabling_interval_zero_effort')])
def test_enable_probe_rejects_broken_timer_or_zero_effort_wiring(seconds,effort,check):
    observations=EnableObservations();observations.record(event(seconds,effort=effort))
    assert not observations.report(contract(),.01)['checks'][check]


def test_no_event_probe_does_not_pass_vacuously():
    assert not all(EnableObservations().report(contract(),.01)['checks'].values())


def test_configuration_does_not_write_simulator_state_or_add_runtime_sensor_requirements():
    settings={'opaque':object()};opaque=settings['opaque'];configure(PLAN,settings)
    assert set(settings) == {'opaque','passive_min_seconds','passive_max_seconds','passive_hold_seconds'}
    assert settings['opaque'] is opaque
