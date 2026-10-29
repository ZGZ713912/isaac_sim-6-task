import json
from pathlib import Path

import numpy as np
import pytest

from wheeled_tasks.chassis.full_robot_asset import build_spec, closure_residual, fk, read_cad
from wheeled_tasks.chassis.traditional_model import equilibrium_pose, linear_seed, virtual_work

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1/urdf/轮腿整车URDFv2.0.1.urdf'


@pytest.fixture(scope='module')
def spec():
    reference = json.loads((ROOT / 'model/纯底盘_v7_flat_base107/urdf/model_spec.json').read_text())
    return build_spec(SOURCE, reference)[1]


def test_full_cad_mass_and_inertias_preserved(spec):
    _, cad = read_cad(SOURCE)
    assert len(spec['bodies']) == 21
    assert sum(b['mass'] for b in spec['bodies']) == pytest.approx(21.44174)
    for before, after in zip(cad['bodies'], spec['bodies']):
        assert before['mass'] == after['mass']
        np.testing.assert_array_equal(before['inertia'], after['inertia'])
        np.testing.assert_array_equal(before['com'], after['com'])


def test_real_loops_and_spring_sliders(spec):
    assert len(spec['constraints']) == 6
    assert sum(j['type'] == 'prismatic' for j in spec['joints']) == 2
    assert len(spec['joints']) == 20
    assert np.max(np.abs(closure_residual(spec, spec['nominal_joint_pos']))) < 2e-6


@pytest.mark.parametrize('length', [.24, .30, .334, .35])
def test_equilibria_virtual_work_and_lqr_at_lengths(spec, length):
    q = equilibrium_pose(spec, length)
    assert np.max(np.abs(closure_residual(spec, q))) < 2e-6
    frames = fk(spec, q)
    for side in ('L', 'R'):
        actual = np.linalg.norm((frames[side+'_link3'][:3, 3] - frames[side+'_link1'][:3, 3])[[0, 2]])
        assert actual == pytest.approx(length, abs=2e-5)
    gravity, spring, residual = virtual_work(spec, q)
    assert residual < 2e-7
    assert np.isfinite(gravity).all() and np.isfinite(spring).all()
    seed = linear_seed(spec, q)
    assert seed['discrete_spectral_radius'] < 1.
    assert seed['balance_gain'][0] > 0  # Full CAD outward-axis convention.


def test_pin_transport_rejects_changed_anchor_geometry():
    reference = json.loads((ROOT / 'model/纯底盘_v7_flat_base107/urdf/model_spec.json').read_text())
    reference['joints'][0]['origin'][1][3] += .01
    with pytest.raises(ValueError, match='anchor geometry changed'):
        build_spec(SOURCE, reference)
