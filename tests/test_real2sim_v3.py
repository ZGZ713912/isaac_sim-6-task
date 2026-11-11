"""Left-leg 2026-11-10 real2sim parameters: moving friction and gas-spring high-beta tail."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation, load_compensation
from wheeled_tasks.chassis.leg_motor_friction import LegMotorFriction, declaration, digest, load

ROOT = Path(__file__).resolve().parents[1]
FRICTION = "model/dm_leg_friction_20261110/model.json"
GAS = ROOT/"model/gas_spring_20261110/gas_spring_model.json"
BUNDLE = ROOT/"model/wheel_leg_v2.0.2/friction_v1/isaac"


def gas_component(path=GAS):
    runtime = json.loads((BUNDLE/"runtime.json").read_text())
    settings = dict(runtime["v5_control"]["gas_spring_compensation"])
    settings["model_file"] = str(path)
    settings["model_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return GasSpringCompensation(settings, root=ROOT, device="cpu")


def beta_probe(component, beta):
    left = component.model["sides"]["left"]
    kn = torch.tensor(left["delta_rad"])
    bt = torch.tensor(left["beta_deg"])
    i = int(torch.searchsorted(bt, torch.tensor([float(beta)])).clamp(1, len(bt)-1))
    u = (beta-float(bt[i-1]))/(float(bt[i])-float(bt[i-1]))
    delta = float(kn[i-1]+u*(kn[i]-kn[i-1]))
    torque, _ = component.evaluate(torch.tensor([[0.0, delta, 0.0, 0.0]]))
    return float(torque[0, 0])


def test_identified_friction_supersedes_the_20261031_prior():
    cfg, model = load(declaration(ROOT, FRICTION), ROOT)
    params = json.loads((ROOT/"model/dm_leg_friction_20261110/parameters.json").read_text())
    assert cfg["enabled"] and model["coulomb_nm"] == [2.7, 3.4, 2.7, 3.4]
    assert model["source_parameter_sha256"] == digest(ROOT/"model/dm_leg_friction_20261110/parameters.json")
    assert params["motors"]["right_knee"]["symmetry_source"] == "left_knee"
    for motor in params["motors"].values():
        assert motor["viscous_coefficient_nm_s_rad"] is None
        assert motor["static_breakaway_nm"] is None
        assert motor["stribeck_velocity_rad_s"] is None


def test_friction_effort_is_signed_coulomb_at_the_native_cap():
    component = LegMotorFriction(1, "cpu", declaration(ROOT, FRICTION), root=ROOT)
    positive = component.effort(torch.tensor([[1., 1., 1., 1.]])).clone()[0]
    negative = component.effort(torch.tensor([[-1., -1., -1., -1.]])).clone()[0]
    assert positive.tolist() == pytest.approx([-2.7, -3.4, -2.7, -3.4])
    assert negative.tolist() == pytest.approx([2.7, 3.4, 2.7, 3.4])
    assert component.native_effort_limit == 43.4


def test_gas_spring_high_beta_tail_holds_the_measured_torque():
    component = gas_component()
    for beta, measured in ((102.0, 14.29), (108.0, 12.93)):
        assert beta_probe(component, beta) == pytest.approx(measured, abs=.6)


def test_gas_spring_measured_beta_band_is_unchanged():
    component = gas_component()
    for beta, measured in ((90.0, 15.74), (96.0, 15.14)):
        assert beta_probe(component, beta) == pytest.approx(measured, abs=.5)


def test_gas_spring_legacy_curve_keeps_the_exponential_tail():
    legacy = gas_component(BUNDLE/"gas_spring_model.json")
    assert beta_probe(legacy, 108.0) < 1.0


def test_unknown_low_compression_rule_is_rejected(tmp_path):
    model = json.loads(GAS.read_text())
    model["force_extension"]["low_compression_kind"] = "invented"
    path = tmp_path/"gas.json"
    path.write_text(json.dumps(model))
    runtime = json.loads((BUNDLE/"runtime.json").read_text())
    settings = dict(runtime["v5_control"]["gas_spring_compensation"])
    settings["model_file"] = str(path)
    settings["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="low-compression"):
        load_compensation(settings, ROOT)


# ---- real2sim v3 plant timing (contract + closed loop) ----
import math

from wheeled_tasks.wheelbipe_port import real2sim
from wheeled_tasks.wheelbipe_port.training_profile import configure_profile, specification


def nominal_timing(count=1):
    return real2sim.LegFeedbackTiming(real2sim.load(), count, "cpu", .005, 0.)


def test_contract_pins_models_and_nominal_mapping():
    spec = real2sim.load()
    timing = nominal_timing()
    assert spec["leg_friction"]["model_path"] == FRICTION
    assert float(timing.feedback_ticks) == pytest.approx(.8)
    assert float(timing.actuation_ticks) == pytest.approx(.2)
    assert float(timing.alpha) == pytest.approx(1-math.exp(-5/4.5))


def test_random_lanes_stay_inside_measured_ranges():
    spec = real2sim.load()
    timing = real2sim.LegFeedbackTiming(spec, 4096, "cpu", .005, torch.ones(4096, 1))
    assert float(timing.age_ms.min()) >= 1. and float(timing.age_ms.max()) <= 2.5
    assert float(timing.act_ms.min()) >= 0. and float(timing.act_ms.max()) <= 2.
    assert float(timing.tau_ms.min()) >= 3.5 and float(timing.tau_ms.max()) <= 6.
    friction = real2sim.sample_friction(spec, 4096, torch.ones(4096, 1), "cpu")
    assert 2.3 <= float(friction[:, 0].min()) and float(friction[:, 1].max()) <= 4.6
    fixed = real2sim.sample_friction(spec, 8, torch.zeros(8, 1), "cpu")
    assert fixed[0].tolist() == pytest.approx([2.7, 3.4, 2.7, 3.4])
    scale = real2sim.sample_armature_scale(spec, 4096, torch.ones(4096, 1), "cpu")
    assert float((scale[:, 1]-1).abs().max()) <= .2+1e-6 and float((scale[:, 0]-1).abs().max()) <= .06+1e-6


def test_physics_rate_other_than_200hz_is_rejected():
    with pytest.raises(ValueError, match="200 Hz"):
        real2sim.LegFeedbackTiming(real2sim.load(), 1, "cpu", .001, 0.)


def test_reset_seeds_history_and_quantizes():
    timing = nominal_timing(2)
    q, dq = torch.full((2, 4), .3), torch.full((2, 4), 1.)
    mq, mv = timing.update(q, dq)
    assert torch.allclose(mq, q)
    assert torch.allclose(mv, torch.round(dq/timing.quantum)*timing.quantum)
    applied = timing.actuate(torch.full((2, 4), 10.))
    assert torch.allclose(applied, torch.full((2, 4), 8.))  # 20 % previous (zero) torque
    timing.reset(torch.tensor([1]))
    assert timing.pending.tolist() == [False, True]


def free_leg_response(J, timing=None, kp=80., kd=1.5, steps=400, h=.005):
    """Semi-implicit Euler inertia under the same PD as V5Control."""
    q = torch.full((1, 4), .05)
    v = torch.zeros(1, 4)
    trace = []
    for _ in range(steps):
        qm, vm = (q, v) if timing is None else timing.update(q, v)
        u = -kp*qm-kd*vm
        if timing is not None:
            u = timing.actuate(u)
        v = v+h*u/J
        q = q+h*v
        trace.append(float(q[0, 0]))
    return trace


def dominant_hz(trace, h=.005):
    signs = [i for i in range(1, len(trace)) if (trace[i-1] < 0) != (trace[i] < 0)]
    return (len(signs)-1)/2/((signs[-1]-signs[0])*h)


def half_cycle_peaks(trace):
    return [abs(trace[i]) for i in range(1, len(trace)-1)
            if abs(trace[i]) >= abs(trace[i-1]) and abs(trace[i]) >= abs(trace[i+1])]


def test_nominal_timing_reproduces_the_measured_lightly_damped_leg_mode():
    # Report section 6: the real legs ring at 17-24 Hz; the zero-delay plant cannot.
    for J, low, high in ((.016, 16., 22.), (.02, 14., 19.)):
        assert max(abs(x) for x in free_leg_response(J)[100:]) < 1e-6
        timing = nominal_timing()
        timing.quantum = 1e-12  # linear mode only
        trace = free_leg_response(J, timing)
        assert low < dominant_hz(trace[:200]) < high
        peaks = half_cycle_peaks(trace)
        ratio = peaks[4]/peaks[3]
        assert .5 < ratio < 1.  # lightly damped, still stable at nominal timing


def test_quantized_velocity_leaves_only_a_sub_milliradian_limit_cycle():
    tail = free_leg_response(.02, nominal_timing())[200:]
    assert max(abs(x) for x in tail) < 1e-3


def test_v5_profile_declares_real2sim_and_anti_collapse_settings():
    spec = specification("foundation-jump-v5")
    v4 = specification("foundation-jump-v4")
    for key in ("jump", "task_fractions", "phases", "reward_overrides", "parameter_overrides", "control"):
        assert spec[key] == v4[key]
    assert spec["ppo_overrides"] == {"init_noise_std": .5}
    assert spec["leg_target_projection_penalty"] == 4.
    assert spec["control"]["leg_kp_nm_per_rad"] == 80. and spec["control"]["leg_kd_nm_s_per_rad"] == 1.5
    assert real2sim.load(spec["real2sim"])["schema"] == "wheelbipe-real2sim-v3"
    assert specification("foundation-jump-v5", 10000)["total_updates"] == 10000
