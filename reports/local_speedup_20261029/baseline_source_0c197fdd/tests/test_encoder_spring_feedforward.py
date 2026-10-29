"""Virtual work, physical closure and real PD-call tests; no PPO or simulator."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from build_v6_height_lookup import LookupBuilder, digest
from wheeled_tasks.chassis.encoder_spring_feedforward import EncoderGasSpringFeedforward, resolve
from wheeled_tasks.chassis.training_gas_spring import resolve as resolve_spring
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())
CFG = json.loads((ROOT/"contracts/scratch_standing_base_mechanism_v2.json").read_text())
BUNDLE = ROOT/CFG["asset_directory"]
SPEC = json.loads((BUNDLE/"model_spec.json").read_text())
NAMES = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
SLIDERS = ("L_spring_slide", "R_spring_slide")


def settings():
    spring = resolve_spring(ROOT, PLAN["gas_spring_model"], SPEC)
    return resolve(ROOT, PLAN["gas_spring_feedforward"], spring, BUNDLE/"model_spec.json")


@pytest.fixture(scope="module")
def model():
    return EncoderGasSpringFeedforward(settings())


@pytest.fixture(scope="module")
def builder():
    return LookupBuilder(BUNDLE, digest(BUNDLE/"manifest.json"))


def pose(builder, beta, common=(0., 0.)):
    q = dict(builder.nominal)
    start = [math.degrees(sign*q[builder.passive_name(side)]+math.pi-2.3573)
             for side, sign in (("L", 1.), ("R", -1.))]
    # A distant one-shot solve can select the other four-bar assembly. Preserve
    # the physical nominal branch by continuation, like the delivered pose bank.
    for values in np.linspace(start, beta, 40)[1:]:
        fixed = dict(builder.active)
        for side, b, sign in zip(("L", "R"), values, (1., -1.)):
            del fixed[side*2+"_joint1"]
            fixed[builder.passive_name(side)] = sign*(math.radians(b)-(math.pi-2.3573))
        q = builder.solve(fixed, q)
    start = dict(q)
    for fraction in np.linspace(0., 1., 20)[1:]:
        fixed = {n: start[n] for n in builder.active}
        for side, shift in zip(("L", "R"), common):
            for name in (side+"_joint1", side*2+"_joint1"):
                fixed[name] += fraction*shift
        q = builder.solve(fixed, q)
    return q


def enc(q):
    return torch.tensor([[q[name] for name in NAMES]], dtype=torch.float64)


@pytest.mark.parametrize("beta,common", [((41., 106.), (0., 0.)), ((56., 100.), (.9, -.7)),
    ((65.37, 85.11), (-1.2, 1.3)), ((85., 65.), (2.8, -2.8)), ((100., 56.), (0., 0.))])
def test_closed_chain_slider_and_independent_motor_virtual_work(model, builder, beta, common):
    q = pose(builder, beta, common)
    holding = model(enc(q)).detach().numpy()[0]
    actual_compression = np.array([SPEC["spring_binding"][n]["compression_at_q_zero_m"]-q[n] for n in SLIDERS])
    np.testing.assert_allclose(model.last["compression_m"][0].numpy(), actual_compression, atol=2e-6, rtol=0.)
    assert model.last["geometry_valid"].all()
    independent_jacobian = np.zeros((2, 4))
    # Re-solve the complete closed chain after EACH driven-axis perturbation.
    # This does not use our lookup interpolation, mount equation or derivative.
    for column, name in enumerate(NAMES):
        fixed = {n: q[n] for n in builder.active}
        plus, minus = dict(fixed), dict(fixed)
        plus[name] += 1e-5
        minus[name] -= 1e-5
        qp, qm = builder.solve(plus, q), builder.solve(minus, q)
        independent_jacobian[:, column] = [(qp[n]-qm[n])/2e-5 for n in SLIDERS]
    f = model.last["axial_force_n"][0].detach().numpy()
    expected = -f @ independent_jacobian
    np.testing.assert_allclose(holding, expected, atol=.003, rtol=0.)
    # Cancelling the spring is distinct from adding an extra physical spring.
    np.testing.assert_allclose(holding+f @ independent_jacobian, 0., atol=.003)


def test_work_gradient_same_geometry_and_force_no_fixed_lever_arm(model):
    q = torch.tensor([[-.42, -.25, .42, .10], [-.42, .55, .42, -.75]], dtype=torch.float64, requires_grad=True)
    holding = model(q)
    values = model.last
    jac = torch.stack([torch.autograd.grad(values["length_m"][:, side].sum(), q, retain_graph=True)[0]
                       for side in range(2)], 1)
    expected = -(jac*values["axial_force_n"][:, :, None]).sum(1)
    torch.testing.assert_close(holding, expected, atol=1e-10, rtol=1e-10)
    assert (holding[0]-holding[1]).abs().max() > 1.


def test_mirrored_signs_and_independent_encoder_windings(model, builder):
    q = enc(pose(builder, (75., 75.)))
    actual = model(q).clone()
    assert (actual*actual.new_tensor([1., -1., -1., 1.]) > 0.).all()
    torch.testing.assert_close(actual[:, 0], -actual[:, 1])
    torch.testing.assert_close(actual[:, 2], -actual[:, 3])
    # Axial curves match; CAD mirror rounding need not produce identical torque.
    torch.testing.assert_close(actual[:, :2], -actual[:, 2:], atol=.003, rtol=0.)
    torch.testing.assert_close(model(q+q.new_tensor([2., -3., 4., -5.])*math.tau), actual, atol=1e-10, rtol=1e-10)


def test_c1_knot_torque_and_numerical_stop_extension(model):
    for side in range(2):
        x = model.get(side, "delta_rad")
        samples = x[1:-1:9]
        def holding(d):
            q = torch.tensor([[-.42, .13742282595395358, .42, -.1374155762580851]], dtype=torch.float64).repeat(len(d), 1)
            q[:, 2*side+1] = q[:, 2*side]+d
            return model(q)[:, 2*side:2*side+2]
        assert (holding(samples+1e-9)-holding(samples-1e-9)).abs().max() < 1e-4
        for edge in (x[0], x[-1]):
            assert (holding(edge.reshape(1)+1e-9)-holding(edge.reshape(1)-1e-9)).abs().max() < 1e-4
        holding(torch.stack((x[0]-.0005, x[-1]+.0005)))
        assert model.last["geometry_valid"][:, side].all()


def test_entire_soft_branch_including_unmeasured_ends_has_compensation(model):
    left = model.get(0, "delta_rad")
    # Dense relative-angle coverage, including BOTH exact projected soft stops.
    delta = torch.linspace(float(left[0]), float(left[-1]), 4001, dtype=torch.float64)
    right = model.get(1, "delta_rad")
    right_delta = torch.linspace(float(right[-1]), float(right[0]), 4001, dtype=torch.float64)
    q = torch.stack((delta*0.-.42, delta-.42, delta*0.+.42, right_delta+.42), -1)
    holding = model(q)
    assert model.last["geometry_valid"].all()
    assert torch.isfinite(holding).all() and (holding.abs() > .01).all()
    assert model.last["outside_identified"][[0, -1]].all()
    assert model.last["axial_force_n"][0].min() > 760.
    assert model.last["axial_force_n"][-1].max() < 5.
    assert holding.abs().max() < 40.


def test_invalid_branch_and_nonfinite_encoder_fail_safely(model):
    q = torch.tensor([[-.42, 2., .42, -.1374155762580851]], dtype=torch.float64)
    torque = model(q)
    assert not model.last["geometry_valid"][0, 0]
    assert (torque[0, :2] == 0.).all() and (torque[0, 2:].abs() > 0.).all()
    for value in (float("nan"), float("inf"), -float("inf")):
        q[0, 0] = value
        with pytest.raises(ValueError, match="Nonfinite"):
            model(q)


def production_motor_wrapper(env, settings):
    source = ast.parse((ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py").read_text())
    node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "motor_efforts")
    namespace = dict(self=env, settings=settings, torch=torch, native_efforts=env.v5.motor_efforts,
                     __name__="wheeled_tasks.chassis.robust_getup_env", __package__="wheeled_tasks.chassis")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_production_wrapper", "exec"), namespace)
    return namespace["motor_efforts"]


@pytest.mark.parametrize("correct_scaling", [False, True])
def test_actual_pd_call_uses_sampled_encoders_unclipped_sum_and_final_limit(model, builder, correct_scaling):
    read = lambda n: json.loads((BUNDLE/n).read_text())
    control = V5Control(read("manifest.json"), SPEC, read("fit_10mpa.json"),
        json.loads((ROOT/CFG["control_math_source"]).read_text()), CFG["v5_control"], "cpu")
    encoder = enc(pose(builder, (75., 75.)))
    q = torch.zeros(1, 6, dtype=torch.float64)
    q[:, list(control.LEGS)] = encoder
    holding = model(encoder).clone()
    # Deliberately make PD exceed40Nm. Clipping it before adding a negative
    # compensation produces a different answer, so this detects the old order.
    pd = holding.new_tensor([[52., 52., -52., -52.]])
    target = encoder+pd/160.+math.tau
    wheel = torch.zeros(1, 2, dtype=torch.float64)
    observed = q.clone()
    env = SimpleNamespace(v5=control, spring_feedforward=model,
        paired_feedback_enabled=torch.ones(1, 1, dtype=torch.bool),
        getup_feedback=SimpleNamespace(observe=lambda q0, dq0: (observed, torch.zeros_like(observed))),
        domain=SimpleNamespace(kp=torch.ones(1, 1), kd=torch.ones(1, 1)))
    fn = production_motor_wrapper(env, dict(paired_feedback_winding=True, gap_correct_pd_scaling=correct_scaling))
    # Raw simulator q is intentionally different: compensation must use encoder q.
    tau = fn(q+1., q+10., target, wheel)
    torch.testing.assert_close(model.last["encoder_position"], encoder)
    torch.testing.assert_close(model.last["pd_effort_nm"], pd, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tau[:, list(control.LEGS)], (pd+holding).clamp(-40., 40.))
    assert not torch.allclose((pd+holding).clamp(-40., 40.), (pd.clamp(-40., 40.)+holding).clamp(-40., 40.))
    assert tau[:, list(control.WHEELS)].eq(0.).all()


def test_model_binding_is_explicit_and_authenticated():
    resolved = settings()
    assert resolved["spring_model_sha256"] == PLAN["gas_spring_model"]["sha256"]
    declaration = deepcopy(PLAN["gas_spring_feedforward"])
    declaration["geometry_sha256"] = "0"*64
    with pytest.raises(ValueError):
        resolve(ROOT, declaration, resolve_spring(ROOT, PLAN["gas_spring_model"], SPEC), BUNDLE/"model_spec.json")


def test_training_config_and_startup_expose_compensation_without_changing_abi(tmp_path):
    from train_prone_motion import configuration
    cfg, task, _ = configuration(tmp_path, PLAN, profile="requested", strength=1.)
    resolved = cfg["v5_control"]["encoder_spring_feedforward"]
    assert resolved["enabled"] and resolved["total_leg_limit_nm"] == 40.
    assert task["prone_motion_task"]["gas_spring_feedforward"]["enabled"]
    fake = SimpleNamespace(device="cpu", dt=.001, startup_report={})
    EncoderGasSpringFeedforward(resolved).install(fake)
    assert fake.startup_report["encoder_spring_feedforward"]["physical_spring_retained"]
    assert not fake.startup_report["encoder_spring_feedforward"]["privileged_force_or_slider_input"]
