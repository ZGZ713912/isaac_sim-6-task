"""Independent load/sign references and real-controller saturation checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]
MODEL = "model/gas_spring_compensation_20261028/model.json"
SETTINGS = dict(enabled=True, model_file=MODEL,
    model_sha256=hashlib.sha256((ROOT / MODEL).read_bytes()).hexdigest(), gain=1.,
    force_extrapolation="positive_c1_endpoint_trend_v1",physical_curve_outside_geometry="hold_geometry_endpoint",
    relative_soft_limits=dict(minimum=-.140638855,maximum=1.3889088544,margin=.03,projection="outward_relative_torque_v1"))


def pose(model, knot):
    left, right = (model["sides"][s]["delta_rad"][knot] for s in ("left", "right"))
    return torch.tensor([[-.42, -.42 + left, .42, .42 - right]], dtype=torch.float64)


@pytest.mark.parametrize("beta,holding", [(60.,18.912), (75.,18.282), (90.,15.840), (98.,13.292)])
def test_matches_independent_native_holding_reference(beta, holding):
    spring = GasSpringCompensation(SETTINGS, root=ROOT, device="cpu", dtype=torch.float64)
    q = pose(spring.model, round((beta - 39.5) * 4))
    tau, diagnostic = spring.evaluate(q)
    # The native fixture reports the final equilibrium near the nominal pose.
    torch.testing.assert_close(tau, tau.new_tensor([[holding, -holding, -holding, holding]]), atol=.02, rtol=0.)
    torch.testing.assert_close(diagnostic["beta_deg"], tau.new_tensor([[beta, beta]]))
    # Common rotation leaves the spring length unchanged, including multiple turns.
    shifted = q + q.new_tensor([[18.,18.,-18.,-18.]])
    torch.testing.assert_close(spring.evaluate(shifted)[0], tau, atol=1e-10, rtol=0.)


def test_whole_soft_limit_range_has_positive_compensation_and_nonfinite_inputs_fail():
    spring = GasSpringCompensation(SETTINGS, root=ROOT, device="cpu", dtype=torch.float64)
    for knot in range(277):
        tau, diagnostic = spring.evaluate(pose(spring.model, knot))
        torch.testing.assert_close(diagnostic["weight"], torch.ones((1,2), dtype=torch.float64))
        assert (diagnostic["force_n"]>0.).all() and tau.abs().min()>0.
    outside = torch.tensor([[0.,-.3,0.,-1.6]], dtype=torch.float64)
    assert spring.evaluate(outside)[0].count_nonzero() == 0
    outside[0,0] = float("nan")
    assert not torch.isfinite(spring.evaluate(outside)[0]).any()


def test_hash_and_configuration_are_authenticated():
    with pytest.raises(ValueError, match="hash"):
        GasSpringCompensation(dict(SETTINGS, model_sha256="0"*64), root=ROOT, device="cpu")
    for changes in (dict(gain=1.1), dict(force_extrapolation=None), dict(physical_curve_outside_geometry=None)):
        with pytest.raises(ValueError):
            GasSpringCompensation(dict(SETTINGS, **changes), root=ROOT, device="cpu")
    changed = deepcopy(SETTINGS)
    changed["relative_soft_limits"]["maximum"] += .01
    with pytest.raises(ValueError,match="authenticated model"):
        GasSpringCompensation(changed,root=ROOT,device="cpu")


def test_extension_is_positive_monotone_and_preserves_measured_values_and_endpoint_slopes():
    spring=GasSpringCompensation(SETTINGS,root=ROOT,device="cpu",dtype=torch.float64)
    torch.testing.assert_close(spring.physical_force(spring.compression),spring.force_n,atol=1e-10,rtol=0.)
    s=torch.linspace(spring.geometry_compression_min,spring.geometry_compression_max,1000,dtype=torch.float64)
    force=spring.physical_force(s)
    assert (force>0).all() and (torch.diff(force)>=-1e-10).all()
    eps=1e-9
    for boundary in (spring.compression[0],spring.compression[-1]):
        center=spring.physical_force(boundary)
        left=(center-spring.physical_force(boundary-eps))/eps
        right=(spring.physical_force(boundary+eps)-center)/eps
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-3)


def controller(compensated=True):
    cfg = json.loads((ROOT / "contracts/scratch_standing_base_mechanism_v2.json").read_text())
    bundle = ROOT / cfg["asset_directory"]
    manifest, spec, fit = (json.loads((bundle/n).read_text()) for n in ("manifest.json", "model_spec.json", "fit_10mpa.json"))
    prior = json.loads((ROOT / cfg["control_math_source"]).read_text())
    settings = deepcopy(cfg["v5_control"])
    if compensated:
        settings["gas_spring_compensation"] = SETTINGS
    return V5Control(manifest, spec, fit, prior, settings, "cpu", root=ROOT)


def test_feedforward_is_summed_before_final_motor_limit_without_changing_wheels():
    control = controller()
    q4 = pose(control.gas_compensation.model,142).float()
    q = torch.zeros(1,6)
    q[:,list(control.LEGS)] = q4
    dq = torch.zeros_like(q)
    legs = q4 + torch.tensor([[.20,-.20,-.20,.20]])
    wheels = torch.zeros(1,2)
    tau = control.motor_efforts(q,dq,legs,wheels)
    torch.testing.assert_close(tau[:,list(control.LEGS)], torch.tensor([[40.,-40.,-40.,40.]]))
    assert tau[:,list(control.WHEELS)].count_nonzero() == 0
    assert control.requested_motor_effort.abs().max() > 40.
    # Cancel a large position error with FF: clipping PD first gives a different result.
    legs = q4 + torch.tensor([[-.30,.30,.30,-.30]])
    tau = control.motor_efforts(q,dq,legs,wheels)
    torch.testing.assert_close(tau[:,list(control.LEGS)],
        (160.*(legs-q4)+control.gas_compensation_effort).clamp(-40.,40.))


def test_disabled_controller_keeps_existing_pd_and_new_plant_uses_measured_force():
    off = controller(False)
    q = torch.zeros(2,6)
    dq = torch.ones_like(q)
    tau = off.motor_efforts(q,dq,torch.full((2,4),.01),torch.ones(2,2))
    torch.testing.assert_close(tau[:,list(off.LEGS)],torch.full((2,4),-.9))
    on = controller(True)
    compression = torch.tensor([[.03,.04]])
    torch.testing.assert_close(on.spring_efforts(on.s0-compression),on.gas_compensation.physical_force(compression))


def test_compensated_sum_retains_inward_and_common_torque_at_soft_limits():
    control=controller()
    # Both hips carry a common 5 Nm command. At the lower boundary, spring FF
    # alone points outward; projecting it must retain this common rotation.
    q=torch.tensor([[0.,-.14,0.,0.,.14,0.]])
    legs=q[:,list(control.LEGS)]+5./160.
    tau=control.motor_efforts(q,torch.zeros_like(q),legs,torch.zeros(1,2))
    torch.testing.assert_close(tau[:,list(control.LEGS)],torch.full((1,4),5.),atol=2e-4,rtol=0.)
    # At the upper boundary FF points inward and remains active.
    q=torch.tensor([[0.,1.388,0.,0.,-1.388,0.]])
    legs=q[:,list(control.LEGS)]
    tau=control.motor_efforts(q,torch.zeros_like(q),legs,torch.zeros(1,2))
    assert tau[0,0]>0 and tau[0,1]<0 and tau[0,3]<0 and tau[0,4]>0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_agrees_with_cpu():
    cpu = GasSpringCompensation(SETTINGS,root=ROOT,device="cpu")
    gpu = GasSpringCompensation(SETTINGS,root=ROOT,device="cuda")
    q = torch.cat([pose(cpu.model,k) for k in range(277)]).float()
    torch.testing.assert_close(gpu.evaluate(q.cuda())[0].cpu(),cpu.evaluate(q)[0],atol=2e-5,rtol=2e-5)
