"""Physical consistency and domain checks for the optional identified curve."""
import json

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
