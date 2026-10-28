"""The training call must use measured axial force on both actual sliders."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve
from wheeled_tasks.chassis.training_gas_spring import resolve, TrainingGasSpring, force_from_knots
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())
CFG = json.loads((ROOT/"contracts/scratch_standing_base_mechanism_v2.json").read_text())
BUNDLE = ROOT/CFG["asset_directory"]
SPEC = json.loads((BUNDLE/"model_spec.json").read_text())


def runtime():
    return TrainingGasSpring(resolve(ROOT, PLAN["gas_spring_model"], SPEC), "cpu")


def test_measured_knots_preserved_and_both_sides_share_force():
    model = runtime()
    s = torch.tensor(model.curve.compression_knots_m)
    expected = torch.tensor(model.curve.force_knots_n)
    actual = model.force(s[:, None].repeat(1, 2))
    torch.testing.assert_close(actual[:, 0], expected)
    torch.testing.assert_close(actual[:, 1], expected)
    assert model.right_curve.symmetry_assumed
    assert not model.last_outside.any()


def test_public_curve_stays_strict_and_training_extension_is_labelled():
    model = runtime()
    s = torch.tensor([[.0, .08], [.0213, .07]])
    with pytest.raises(ValueError):
        model.curve.force(s)
    # Keep the old endpoint-hold adapter available for historical reproduction.
    old_settings = deepcopy(model.settings)
    old_settings["outside_identified_domain"] = "endpoint_hold_unvalidated"
    old_model = TrainingGasSpring(old_settings, "cpu")
    force = old_model.force(s)
    torch.testing.assert_close(force[:, 0], force.new_full((2,), model.curve.force_knots_n[0]))
    torch.testing.assert_close(force[:, 1], force.new_full((2,), model.curve.force_knots_n[-1]))
    assert old_model.last_outside.all()


def test_trend_extension_is_positive_monotone_preserves_every_measured_knot():
    model = runtime()
    s = torch.linspace(0., .08, 8001)
    force = model.force(s)
    assert torch.isfinite(force).all() and (force > 0.).all()
    assert (force.diff() >= 0.).all()
    x = torch.tensor(model.curve.compression_knots_m)
    torch.testing.assert_close(model.force(x), torch.tensor(model.curve.force_knots_n))
    assert model.force(torch.tensor(.017964294)) < 5.
    assert 760. < model.force(torch.tensor(.072795039)) < 780.
    assert model.settings["resolved_trend_extension"]["extrapolation_validated_by_measurement"] is False


def test_tail_matches_endpoint_value_derivative_and_work_gradient():
    model = runtime()
    x, y = (torch.tensor(v, dtype=torch.float64) for v in
            (model.curve.compression_knots_m, model.curve.force_knots_n))
    slopes = y.diff()/x.diff()
    call = lambda s: force_from_knots(s, x, y, slopes, "positive_c1_endpoint_trend_v1", .08)
    for edge, slope in ((x[0], slopes[0]), (x[-1], slopes[-1])):
        for sign in (-1., 1.):
            s = (edge+sign*1e-10).clone().requires_grad_()
            force = call(s)
            gradient = torch.autograd.grad(force, s)[0]
            torch.testing.assert_close(gradient, slope, rtol=1e-6, atol=1e-5)
    for value, index, slope in ((.018, 0, slopes[0]), (.0728, -1, slopes[-1])):
        s = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        rate = slope/y[index]
        work = y[index]/rate*torch.expm1(rate*(s-x[index]))
        torch.testing.assert_close(torch.autograd.grad(work, s)[0], call(s))


def test_trend_contract_sha_and_endpoint_slopes_are_authenticated(tmp_path):
    declaration = deepcopy(PLAN["gas_spring_model"])
    declaration["trend_extension"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="trend extension contract SHA"):
        resolve(ROOT, declaration, SPEC)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_compression_refused(bad):
    with pytest.raises(ValueError):
        runtime().force(torch.tensor([[bad, .04]]))


def test_potential_gradient_and_closed_work_within_measured_domain():
    curve = runtime().curve
    s = torch.tensor([.03, .04, .05, .06], dtype=torch.float64, requires_grad=True)
    force = torch.autograd.grad(curve.potential_energy(s).sum(), s)[0]
    torch.testing.assert_close(force, curve.force(s))
    assert curve.potential_energy(torch.tensor(curve.compression_knots_m[0], dtype=torch.float64)) == 0


@pytest.mark.parametrize("kind", ["hash", "symmetry", "endpoint_policy", "mounting"])
def test_input_authentication_and_installation_are_required(kind):
    declaration, spec = deepcopy(PLAN["gas_spring_model"]), deepcopy(SPEC)
    if kind == "hash": declaration["sha256"] = "0"*64
    if kind == "symmetry": declaration["right_leg_symmetry_assumed"] = False
    if kind == "endpoint_policy": declaration["outside_identified_domain"] = "silently_extrapolate"
    if kind == "mounting": spec["spring_binding"]["R_spring_slide"]["compression_at_q_zero_m"] += .001
    with pytest.raises(ValueError): resolve(ROOT, declaration, spec)


def test_actual_controller_call_replaces_polynomial_and_keeps_domain_multiplier():
    read = lambda name: json.loads((BUNDLE/name).read_text())
    prior = json.loads((ROOT/CFG["control_math_source"]).read_text())
    control = V5Control(read("manifest.json"), SPEC, read("fit_10mpa.json"), prior, CFG["v5_control"], "cpu")
    model = runtime()
    domain = SimpleNamespace(spring=torch.ones(3, 1))
    env = SimpleNamespace(v5=control, domain=domain, dt=.001, startup_report={})
    s = torch.tensor([[.0213, .0213], [.04, .04], [.07, .07]])
    position = control.s0-s
    old = control.spring_efforts(position)
    model.install(env)
    actual = control.spring_efforts(position)
    torch.testing.assert_close(actual, model.force(s))
    assert (actual-old).abs().max() > 100
    torch.testing.assert_close(actual[:, 0], actual[:, 1])
    domain.spring[:] = .8
    torch.testing.assert_close(control.spring_efforts(position), actual*.8)
    assert env.startup_report["identified_gas_spring"]["physical_call_hz"] == 1000


def test_installed_232mm_coordinate_and_unmeasured_prone_reset_are_recorded():
    model = runtime()
    bank = json.loads((ROOT/"reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())
    name = json.loads((ROOT/PLAN["prone_geometry_contract"]).read_text())["reset_pose_name"]
    pose = next(p for p in bank["poses"] if p["name"] == name)
    compression = torch.tensor([[SPEC["spring_binding"][n]["compression_at_q_zero_m"]
        - pose["full_joint_positions"][n] for n in ("L_spring_slide", "R_spring_slide")]])
    model.force(compression)
    assert model.last_outside.all()  #103deg spawn is beyond the measured101deg edge.
    assert model.settings["installed_stroke_m"] == .08
