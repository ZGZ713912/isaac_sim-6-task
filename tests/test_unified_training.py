"""Branch integration: frozen dev outputs, one compensation owner and 100 Hz."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_prone_motion import configuration, learner_config, source_hashes
from test_encoder_spring_feedforward import production_motor_wrapper
from test_gas_spring_compensation import SETTINGS, controller
from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation
from wheeled_tasks.chassis.tensor_cuda_graph import TensorCudaGraph

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "contracts/wheel_leg_prone_motion_v3.json"
PLAN = json.loads(PLAN_PATH.read_text())


@pytest.fixture
def reference():
    path = ROOT / "tests/data/unified_dev_gas_reference.npz"
    declaration = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == declaration["fixture_sha256"]
    assert declaration["source_commit"] == "7359170bd47cdd679e10ea66300a13b3c7ee3bd6"
    with np.load(path, allow_pickle=False) as archive:
        return {key: torch.from_numpy(archive[key].copy()) for key in archive.files}


def test_force_and_holding_keep_frozen_dev_outputs(reference):
    spring = GasSpringCompensation(SETTINGS, root=ROOT, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(spring.physical_force(reference["compression"]), reference["force"],
                               atol=0., rtol=0.)
    torch.testing.assert_close(spring.evaluate(reference["q4"])[0], reference["holding"],
                               atol=0., rtol=0.)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_native_coverage_scans_actual_soft_limits_in_float32(device):
    from evaluate_prone_motion import compensation_coverage
    spring = GasSpringCompensation(SETTINGS, root=ROOT, device=device)
    coverage = compensation_coverage(spring, dev_gas=True, device=device)
    assert spring.last["geometry_valid"].all() and torch.isfinite(coverage).all()
    assert (coverage.abs() > .01).all() and (coverage.abs() < 40.).all()
    encoder = spring.last["encoder_position"]
    relative = (encoder.unflatten(-1, (2, 2))*encoder.new_tensor([[-1., 1.], [1., -1.]])).sum(-1)
    limits = SETTINGS["relative_soft_limits"]
    torch.testing.assert_close(relative[[0, -1]], relative.new_tensor([
        [limits["minimum"]]*2, [limits["maximum"]]*2]), atol=1e-7, rtol=0.)


@pytest.mark.parametrize("correct_scaling", [False, True])
def test_real_recovery_wrapper_matches_dev_and_adds_compensation_once(reference, correct_scaling):
    control = controller()
    q, dq, legs, wheels = (reference[key] for key in ("q", "dq", "legs", "wheels"))
    # Exercise the actual production closure without importing native Kit.
    env = SimpleNamespace(v5=control, spring_feedforward=control.gas_compensation,
        getup_feedback=SimpleNamespace(observe=lambda _q, _dq: (q, dq)),
        domain=SimpleNamespace(kp=torch.ones(len(q), 1), kd=torch.ones(len(q), 1)))
    wrapper = production_motor_wrapper(env, dict(gap_correct_pd_scaling=correct_scaling))
    actual = wrapper(q+2., dq+3., legs, wheels)
    torch.testing.assert_close(actual, reference["effort"], atol=1e-5, rtol=0.)
    torch.testing.assert_close(env.spring_feedforward.last["encoder_position"], q[:, list(control.LEGS)])
    # Saturation must act on the unclipped PD + FF sum.
    expected_pd = 160.*(legs-q[:, list(control.LEGS)])-2.5*dq[:, list(control.LEGS)]
    torch.testing.assert_close(env.spring_feedforward.last["pd_effort_nm"], expected_pd)
    torch.testing.assert_close(env.spring_feedforward.last["projected_effort_nm"], actual[:, list(control.LEGS)])


@pytest.mark.parametrize("profile,strength", [("nominal", 0.), ("requested", 1.)])
def test_default_unified_configuration_keeps_rates_curriculum_and_one_backend(tmp_path, profile, strength):
    cfg, settings, _ = configuration(tmp_path, PLAN, profile=profile, strength=strength)
    gas = cfg["v5_control"]
    assert gas["gas_spring_compensation"] == PLAN["gas_spring_feedforward"] == SETTINGS
    assert "encoder_spring_feedforward" not in gas and "identified_gas_spring" not in gas
    assert cfg["policy_dt"] == .01 and cfg["physics_dt"] == .001
    assert cfg["num_steps_per_env"] == PLAN["rollout_steps"] == 64
    assert gas["leg_kp"] == 160. and gas["leg_kd"] == 2.5
    assert cfg["wheel_reduction"]["gear_ratio"] == 15.62
    assert json.loads((tmp_path/"task_contract.json").read_text())["static_load_initialization"] is None
    old = json.loads((ROOT / "contracts/wheel_leg_prone_motion_v2.json").read_text())
    for field in ("domain_sequence", "roles_per20", "max_updates", "snapshot_interval",
                  "forward_speed_m_s", "spin_target_rpm", "post_completion_quiet_seconds"):
        assert PLAN[field] == old[field]
    assert learner_config(PLAN)["resume"] is False
    frozen = source_hashes(PLAN_PATH, PLAN)
    evidence = json.loads((ROOT/cfg["gas_spring_compensation_evidence"]["file"]).read_text())
    assert all(frozen[path] == sha for path, sha in evidence["files_sha256"].items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dev_force_and_holding_graphs_agree_and_do_not_alias(reference):
    eager = GasSpringCompensation(SETTINGS, root=ROOT, device="cuda")
    replay = GasSpringCompensation(SETTINGS, root=ROOT, device="cuda")
    q = reference["q"].cuda()[:, list(controller().LEGS)]
    compression = reference["compression"].float().cuda()
    with torch.inference_mode():
        replay.tensor_replay = TensorCudaGraph(replay._evaluate, q)
        replay.force_tensor_replay = TensorCudaGraph(replay._evaluate_force, compression)
        previous = None
        for turn in (0., 6.283185307179586, -6.283185307179586):
            positions = q+turn
            pd = torch.linspace(-60., 60., q.numel(), device="cuda").reshape_as(q)
            torch.testing.assert_close(replay.motor_efforts(pd, positions),
                                       eager.motor_efforts(pd, positions), atol=0., rtol=0.)
            if previous is not None:
                for key, value in previous[0].items():
                    torch.testing.assert_close(value, previous[1][key], atol=0., rtol=0.)
            previous = (dict(replay.last), {key: value.clone() for key, value in replay.last.items()})
            torch.testing.assert_close(replay.physical_force(compression), eager.physical_force(compression),
                                       atol=0., rtol=0.)
        assert not torch.isfinite(replay.evaluate(q+float("nan"))[0]).all()
        with pytest.raises(ValueError, match="Nonfinite"):
            replay.physical_force(compression+float("nan"))
