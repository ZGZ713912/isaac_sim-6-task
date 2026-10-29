"""Verify the actual prone-task loop, input provenance and one inertia draw."""
from copy import deepcopy
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch

from train_prone_motion import configuration
from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
from wheeled_tasks.chassis.training_wheel_response import configure, scaled_parameters, validate_timing

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())


@pytest.fixture
def configured(tmp_path):
    cfg, settings, _ = configuration(tmp_path, PLAN, profile="requested", strength=1.)
    return cfg, settings


def test_task_selects_authenticated_independent_model_and_matching_recorded_control(configured):
    cfg, settings = configured
    wheel = cfg["identified_wheel_response"]
    assert wheel["model_sha256"] == PLAN["wheel_model"]["model"]["sha256"]
    assert wheel["apply_aggregate_delay"]
    assert cfg["control_math_source"] == PLAN["wheel_model"]["control"]["file"]
    for field in ("armature", "frictionloss", "damping", "time_constant_s"):
        assert cfg["actuator_response"][field][2] == cfg["actuator_response"][field][5] == 0.
    assert cfg["v5_control"]["identified_gas_spring"]["enabled"]
    assert cfg["stages"][0]["vx_max"] == 2.
    assert settings["sensor_delay_policy_ticks"] == [0, 1]
    validate_timing(settings)


def test_actual_v5_controller_accepts_new_scope_and_uses_recorded_P05(configured):
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.v5_control import V5Control
    from wheeled_tasks.v40.core import motor_torque_limit
    cfg, _ = configured
    bundle = ROOT/cfg["asset_directory"]
    read = lambda name:json.loads((bundle/name).read_text())
    control = load_chassis_control(ROOT/cfg["control_math_source"])
    v5 = V5Control(read("manifest.json"), read("model_spec.json"), read("fit_10mpa.json"), control, cfg["v5_control"], "cpu")
    q = v5.nominal.expand(2,-1).clone()
    dq = torch.zeros_like(q)
    dq[:,[2,5]] = 2.
    actual = v5.motor_efforts(q,dq,q[:,[0,1,3,4]],torch.full((2,2),4.))
    torch.testing.assert_close(actual[:,[2,5]],torch.ones(2,2))
    torch.testing.assert_close(actual[:,[0,1,3,4]],torch.zeros(2,4))
    torch.testing.assert_close(motor_torque_limit(torch.tensor([0.,100.,-100.]),v5.wheel_prior),torch.full((3,),4.5))
    bad = deepcopy(v5.wheel_prior)
    bad["gear_ratio"] = 15.8
    with pytest.raises(ValueError,match="15.62 output"):
        motor_torque_limit(torch.zeros(1),bad)


@pytest.mark.parametrize("key", ["model", "control", "evidence"])
def test_task_refuses_changed_delivered_input(configured, key):
    cfg, settings = configured
    declaration = deepcopy(PLAN["wheel_model"])
    declaration[key]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="input SHA"):
        configure(ROOT, cfg, declaration, settings)


@pytest.mark.parametrize("key,bad", [("torque_delay_ms", [0,1]), ("motor_tau_ms", [0.,1.]),
    ("pd_feedback_delay_ms", [0,1]), ("pd_feedback_period_ms", [1,2])])
def test_no_aggregate_delay_stacking(configured, key, bad):
    _, settings = configured
    settings[key] = bad
    with pytest.raises(ValueError, match="overlap"):
        validate_timing(settings)


def test_inertia_scales_cad_and_added_armature_with_existing_draw_exactly_once(configured):
    cfg, _ = configured
    wheels = IdentifiedWheelResponse(3, "cpu", .001, cfg["identified_wheel_response"], root=ROOT)
    domain = SimpleNamespace(wheel_ids=[1, 3],
        mass_scale=torch.tensor([[1., 1., 1., 1.], [1., .92, 1., 1.08], [1., 1.04, 1., .97]]),
        inertia_scale=torch.tensor([[1., 1., 1., 1.], [1., .94, 1., 1.06], [1., 1.05, 1., .98]]))
    state = torch.random.get_rng_state().clone()
    scale, added, total = scaled_parameters(wheels, domain, 3)
    assert torch.equal(state, torch.random.get_rng_state())
    cad = total.new_tensor([r["cad_inertia_kg_m2"] for r in wheels.rows])
    torch.testing.assert_close(total, scale*cad+added)
    torch.testing.assert_close(scale, (domain.mass_scale*domain.inertia_scale)[:, [1,3]])
    assert not torch.equal(total[:,0], total[:,1])
    _, nominal_added, nominal_total = scaled_parameters(wheels, None, 3)
    torch.testing.assert_close(nominal_total, nominal_added+cad)


@pytest.mark.parametrize("fault", [None, "cad", "armature"])
def test_native_installation_requires_matching_cad_and_added_armature(configured, monkeypatch, fault):
    import sys
    from wheeled_tasks.chassis.training_wheel_response import install
    cfg, settings = configured
    wheels = IdentifiedWheelResponse(2, "cpu", .001, cfg["identified_wheel_response"], root=ROOT)
    cad = torch.diag(torch.tensor([.000296, .000296, .000575])).expand(2,2,3,3).clone()
    if fault == "cad": cad[1,1,2,2] += .0001
    armatures = torch.zeros(2,6)
    def write(*, armature, joint_ids):
        armatures[:,joint_ids] = armature * (1.01 if fault == "armature" else 1.)
    robot = SimpleNamespace(joint_names=["L_joint1","LL_joint1","L_joint3","R_joint1","RR_joint1","R_joint3"],
        body_names=["L_link3","R_link3"], root_view=SimpleNamespace(get_inertias=lambda:cad.reshape(2,2,9),
        get_dof_armatures=lambda:armatures), write_joint_armature_to_sim_index=write)
    env = SimpleNamespace(robot=robot, identified_wheels=wheels, getup_cfg=settings, dynamics_randomization=None,
        num_envs=2, dt=.001, policy_dt=.005, startup_report={})
    monkeypatch.setitem(sys.modules,"warp", SimpleNamespace(to_torch=lambda a:a))
    if fault:
        with pytest.raises(RuntimeError, match="CAD inertia|armature domain readback"):
            install(env)
    else:
        install(env)
        assert env.startup_report["training_identified_wheels"]["native_properties_verified"]
        torch.testing.assert_close(armatures[:,[2,5]]+.000575, wheels.inertia)


def test_actual_robust_physics_loop_clears_queued_torque_and_keeps_passive_losses(configured):
    cfg, _ = configured
    plant = IdentifiedWheelResponse(2, "cpu", .001, cfg["identified_wheel_response"], root=ROOT)
    # Execute the production torque/effort blocks, not a parallel test version.
    source = (ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py").read_text()
    control = textwrap.dedent(source.split('            tau = self.v5.motor_efforts(q, dq, legs, wheels)',1)[1]
        .split('            if gap_quality is not None:',1)[0])
    control = 'tau = self.v5.motor_efforts(q, dq, legs, wheels)\n'+control
    effort_block = textwrap.dedent(source.split('            effort = torch.zeros_like(self.nominal)',1)[1]
        .split('            effort[:, self.spring_ids]',1)[0])
    effort_block = 'effort = torch.zeros_like(self.nominal)\n'+effort_block
    request = torch.ones(2,6)
    env = SimpleNamespace(identified_wheels=plant, v5=SimpleNamespace(current_motor_bounds=torch.full((2,6),4.5),
        motor_efforts=lambda *args:request.clone()), actuator_response=SimpleNamespace(apply_torque=lambda x:x),
        passive=torch.zeros(2,dtype=torch.bool), finished=torch.zeros(2,dtype=torch.bool),
        nominal=torch.zeros(2,6), ids=list(range(6)))
    dq = torch.zeros(2,6)
    dq[:,[2,5]] = torch.tensor([[20.,-20.],[20.,-20.]])
    scope = dict(torch=torch,self=env,q=torch.zeros_like(dq),dq=dq,legs=None,wheels=None)
    observed = []
    for tick in range(10):
        env.v5.current_motor_bounds.fill_(4.5)
        exec(control,scope)
        exec(effort_block,scope)
        observed.append(scope['tau'].clone())
    assert all(torch.count_nonzero(t[:,2])==0 for t in observed[:5])
    assert observed[5][0,2] == .5 and observed[6][0,2] == 1.
    assert all(torch.count_nonzero(t[:,5])==0 for t in observed[:3])
    assert observed[3][0,5] == .5 and observed[4][0,5] == 1.
    env.passive[0] = True
    exec(control,scope)
    exec(effort_block,scope)
    assert torch.count_nonzero(scope['tau'][0]) == 0
    assert torch.count_nonzero(plant.history[:,0]) == 0
    assert torch.count_nonzero(scope['tau'][1]) == 6
    torch.testing.assert_close(scope['effort'][:,[2,5]], scope['tau'][:,[2,5]]+plant.passive_effort)
    assert plant.passive_effort[0].abs().sum() > 0  # disabled wheel still dissipates energy
    assert (plant.passive_effort*dq[:,[2,5]] <= 0).all()


def test_training_freezes_every_delivered_evidence_dependency(tmp_path):
    from train_prone_motion import source_hashes
    hashes = source_hashes(ROOT/"contracts/wheel_leg_prone_motion_v1.json", PLAN)
    evidence = json.loads((ROOT/PLAN["wheel_model"]["evidence"]["file"]).read_text())
    for name, value in evidence["files_sha256"].items():
        assert hashes[name] == value
