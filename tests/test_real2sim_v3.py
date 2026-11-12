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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device placement")
def test_plant_tail_and_timing_run_on_cuda():
    runtime = json.loads((BUNDLE/"runtime.json").read_text())
    settings = dict(runtime["v5_control"]["gas_spring_compensation"])
    settings["model_file"] = str(GAS)
    settings["model_sha256"] = hashlib.sha256(GAS.read_bytes()).hexdigest()
    component = GasSpringCompensation(settings, root=ROOT, device="cuda")
    assert component.physical_force(torch.full((4, 2), .024, device="cuda")).is_cuda
    spec = real2sim.load()
    amp = torch.ones(8, 1, device="cuda")
    timing = real2sim.LegFeedbackTiming(spec, 8, "cuda", .005, amp)
    q = torch.zeros(8, 4, device="cuda")
    mq, mv = timing.update(q, q)
    assert mq.is_cuda and mv.is_cuda and timing.actuate(q).is_cuda and timing.observe(q).is_cuda
    assert real2sim.sample_friction(spec, 8, amp, "cuda").is_cuda
    assert real2sim.sample_armature_scale(spec, 8, amp, "cuda").is_cuda


# ---- right leg measured 2026-11-11 (real2sim v3.1, foundation-jump-v6) ----
FRICTION_R = "model/dm_leg_friction_20261111/model.json"
V31 = "contracts/wheelbipe_real2sim_v3p1_20261111.json"


def test_right_leg_friction_is_measured_not_mirrored():
    cfg, model = load(declaration(ROOT, FRICTION_R), ROOT)
    assert model["coulomb_nm"] == [2.7, 3.4, 1.85, 1.75]
    component = LegMotorFriction(1, "cpu", declaration(ROOT, FRICTION_R), root=ROOT)
    assert component.summary()["right_symmetry_assumed"] is False
    params = json.loads((ROOT/"model/dm_leg_friction_20261111/parameters.json").read_text())
    for motor in ("right_hip", "right_knee"):
        assert "symmetry_source" not in params["motors"][motor]
        assert params["motors"][motor]["derivation"].startswith("right measured")
    old = json.loads((ROOT/"model/dm_leg_friction_20261110/parameters.json").read_text())
    for motor in ("left_hip", "left_knee"):
        assert params["motors"][motor] == old["motors"][motor]


def test_previous_symmetric_model_still_loads_unchanged():
    _, model = load(declaration(ROOT, FRICTION), ROOT)
    assert model["coulomb_nm"] == [2.7, 3.4, 2.7, 3.4]
    assert LegMotorFriction(1, "cpu", declaration(ROOT, FRICTION), root=ROOT).summary()["right_symmetry_assumed"] is True


@pytest.mark.parametrize("edit", ["both", "neither", "mirror_tag"])
def test_right_source_declaration_must_be_unambiguous(tmp_path, edit):
    import shutil
    source = ROOT/"model/dm_leg_friction_20261111"
    target = tmp_path/"model/dm_leg_friction_20261111"
    shutil.copytree(source, target)
    params = json.loads((target/"parameters.json").read_text())
    if edit == "both":
        params["right_symmetry_authorized"] = True
    elif edit == "neither":
        params["right_measured"] = False
    else:
        params["motors"]["right_hip"]["symmetry_source"] = "left_hip"
    (target/"parameters.json").write_text(json.dumps(params))
    model = json.loads((target/"model.json").read_text())
    model["files_sha256"]["parameters.json"] = digest(target/"parameters.json")
    (target/"model.json").write_text(json.dumps(model))
    with pytest.raises(ValueError):
        load(declaration(tmp_path, "model/dm_leg_friction_20261111/model.json"), tmp_path)


def test_v31_samples_each_motor_inside_its_own_range():
    spec = real2sim.load(V31)
    assert spec["contract_path"] == V31
    nominal = real2sim.sample_friction(spec, 4, torch.zeros(4, 1), "cpu")
    assert nominal[0].tolist() == pytest.approx([2.7, 3.4, 1.85, 2.7])
    drawn = real2sim.sample_friction(spec, 8192, torch.ones(8192, 1), "cpu")
    for i, (low, high) in enumerate(([2.3, 3.3], [2.9, 4.6], [1.6, 2.4], [1.5, 4.1])):
        assert low-1e-6 <= float(drawn[:, i].min()) and float(drawn[:, i].max()) <= high+1e-6
    v3 = real2sim.load()
    assert v3["contract_path"] == real2sim.CONTRACT
    assert spec["leg_feedback"] == v3["leg_feedback"]
    assert spec["gas_spring_plant"] == real2sim.load()["gas_spring_plant"]


def test_running_v5_experiment_is_untouched_and_v6_only_swaps_the_plant():
    v5, v6 = specification("foundation-jump-v5"), specification("foundation-jump-v6")
    assert v5["real2sim"] == "contracts/wheelbipe_real2sim_v3_20261110.json"
    assert real2sim.load(v5["real2sim"])["leg_friction"]["model_path"] == FRICTION
    assert v6["real2sim"] == V31
    for key in set(v5) | set(v6):
        if key not in ("schema", "real2sim", "changes", "initialization"):
            assert v5[key] == v6[key], key
