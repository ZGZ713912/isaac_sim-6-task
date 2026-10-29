"""Physical consistency and domain checks for the optional identified curve."""
import json
import importlib.util
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve


def test_potential_gradient_equals_compression_force_and_closed_work():
    curve=IdentifiedGasSpringCurve((.02,.03,.06),(100.,300.,600.))
    s=torch.tensor([.021,.027,.04,.059],dtype=torch.float64,requires_grad=True)
    grad=torch.autograd.grad(curve.potential_energy(s).sum(),s)[0]
    torch.testing.assert_close(grad,curve.force(s))
    assert curve.potential_energy(torch.tensor(.02,dtype=torch.float64)).item()==0.
    expected=.5*(100+300)*.01+.5*(300+600)*.03
    assert curve.potential_energy(torch.tensor(.06,dtype=torch.float64)).item()==pytest.approx(expected)


@pytest.mark.parametrize("value",[.019,.061,float("nan"),float("inf")])
def test_unidentified_domain_is_rejected(value):
    curve=IdentifiedGasSpringCurve((.02,.06),(100.,600.))
    with pytest.raises(ValueError):curve.force(torch.tensor(value))


def test_nonpassive_or_unidentified_side_is_rejected():
    with pytest.raises(ValueError):IdentifiedGasSpringCurve((.02,.06),(600.,100.))
    with pytest.raises(ValueError):IdentifiedGasSpringCurve((.02,.06),(100.,600.),"right")


def test_load_export_preserves_sign_units_and_boundaries(tmp_path):
    path=tmp_path/"model.json"
    path.write_text(json.dumps(dict(kind="effective_monotone_table",side="left",
        compression_knots_m=[.02,.06],force_knots_n=[100.,600.])))
    curve=IdentifiedGasSpringCurve.from_json(path)
    torch.testing.assert_close(curve.force(torch.tensor([[.02,.04,.06]])),torch.tensor([[100.,350.,600.]]))


def test_right_leg_requires_explicit_assumption_and_keeps_axial_force_sign():
    left=IdentifiedGasSpringCurve((.02,.06),(100.,600.))
    with pytest.raises(ValueError):left.for_side("right")
    right=left.for_side("right",assume_symmetry=True)
    assert right.side=="right" and right.symmetry_assumed
    s=torch.tensor([.02,.04,.06],dtype=torch.float64)
    torch.testing.assert_close(right.force(s),left.force(s))
    torch.testing.assert_close(right.potential_energy(s),left.potential_energy(s))


def test_right_leg_load_records_assumption_instead_of_claiming_measurement(tmp_path):
    path=tmp_path/"model.json"
    path.write_text(json.dumps(dict(kind="effective_monotone_table",side="left",
        compression_knots_m=[.02,.06],force_knots_n=[100.,600.])))
    with pytest.raises(ValueError):IdentifiedGasSpringCurve.from_json(path,side="right")
    right=IdentifiedGasSpringCurve.from_json(path,side="right",assume_symmetry=True)
    assert right.symmetry_assumed
    assert right.force(.04).item()==pytest.approx(350.)


def test_closed_chain_geometry_supplies_mirrored_motor_torque_signs():
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("identify_gas_spring",root/"tools/identify_gas_spring.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    left=module.geometry(module.BUNDLE,[60.,75.,100.],side="left")
    right=module.geometry(module.BUNDLE,[60.,75.,100.],side="right")
    np.testing.assert_allclose(left["compression_m"],right["compression_m"],atol=1e-7)
    # CAD joint axes contain small left/right asymmetry, below 10 ppm here.
    np.testing.assert_allclose(left["jacobian_m_per_rad"],-right["jacobian_m_per_rad"],rtol=1e-5,atol=1e-9)
