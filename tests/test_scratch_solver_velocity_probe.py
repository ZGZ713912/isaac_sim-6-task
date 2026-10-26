"""A solver diagnostic must change exactly one physical setting and verify it."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("velocity_probe",
    Path(__file__).resolve().parents[1] / "scripts/solver_velocity_iteration_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def baseline():
    cfg = dict(position_iterations=64, velocity_iterations=32, physics_dt=.001, policy_dt=.005,
        unrelated=dict(torque_limit=40., contact_threshold=2.))
    design = {"stages": {"speed": {"roles": {"train": {"contract": deepcopy(cfg)}}}}}
    return cfg, design


@pytest.mark.parametrize('iterations', [1, 4])
def test_only_velocity_count_changes_in_config_and_selected_role(iterations):
    cfg, design = baseline()
    original, original_design = deepcopy(cfg), deepcopy(design)
    contract = probe.configure(cfg, design, iterations)
    original['velocity_iterations'] = iterations
    original_design['stages']['speed']['roles']['train']['contract']['velocity_iterations'] = iterations
    assert cfg == original and design == original_design
    assert contract['new_PPO'] == 0 and not contract['adoption_by_training']


@pytest.mark.parametrize('iterations', [0, 2, 32, True, 4., None])
def test_invalid_probe_rejected_without_mutation(iterations):
    cfg, design = baseline()
    original = deepcopy((cfg, design))
    with pytest.raises(ValueError): probe.configure(cfg, design, iterations)
    assert (cfg, design) == original


def physics():
    return dict(scene_flags={
        'physxScene:solverType': 'TGS', 'physxScene:enableExternalForcesEveryIteration': False,
        'physxScene:enableStabilization': False, 'physxScene:enableCCD': False,
        'physxScene:minPositionIterationCount': 1, 'physxScene:maxPositionIterationCount': 255,
        'physxScene:minVelocityIterationCount': 0, 'physxScene:maxVelocityIterationCount': 255},
        articulation_iterations={'physxArticulation:solverPositionIterationCount':64,
                                 'physxArticulation:solverVelocityIterationCount':4},
        physical_dt_s=.001, policy_dt_s=.005,
        cylinder_convex_approximation=dict(status='observed', value=False))


@pytest.mark.parametrize('fault', [None, 'position', 'velocity', 'solver', 'external', 'dt', 'cylinder'])
def test_native_readback_is_required_before_any_evaluation(fault):
    cfg, design = baseline(); contract = probe.configure(cfg, design, 4); actual = physics()
    if fault == 'position': actual['articulation_iterations']['physxArticulation:solverPositionIterationCount'] = 32
    if fault == 'velocity': actual['articulation_iterations']['physxArticulation:solverVelocityIterationCount'] = 32
    if fault == 'solver': actual['scene_flags']['physxScene:solverType'] = 'PGS'
    if fault == 'external': actual['scene_flags']['physxScene:enableExternalForcesEveryIteration'] = True
    if fault == 'dt': actual['physical_dt_s'] = .005
    if fault == 'cylinder': actual['cylinder_convex_approximation']['value'] = True
    if fault is None: probe.assert_observed(actual, contract)
    else:
        with pytest.raises(ValueError): probe.assert_observed(actual, contract)
