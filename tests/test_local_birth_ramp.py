"""Birth v2 physical gains, training ownership, and exact local revision limits."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from wheeled_tasks.chassis import local_scratch as local
from wheeled_tasks.chassis.local_birth_revision import REVISION, validate_contracts, validate_parent, validate_plan
from wheeled_tasks.chassis.nonstandard_start import (
    NonstandardStart, PROFILE_FILE, nonstandard_start_settings, ramped_start_settings,
    validate_nonstandard_start,
)

ROOT = Path(__file__).resolve().parents[1]
JOINTS = json.loads((ROOT / PROFILE_FILE).read_text())["tree_joint_names"]


def rig(count=2, evaluation=False):
    settings = ramped_start_settings(0)
    settings["iteration_probability_schedule"] = [{"actor_update": 0, "probabilities": [0., 1., 0.]}]
    return NonstandardStart(count, "cpu", .005, settings, ROOT, JOINTS, evaluation=evaluation)


def test_final_effort_gain_whole_interval_mask_history_and_returned_state_bootstrap():
    module = rig()
    module.reset(torch.tensor([0]), 1024)
    torque = torch.tensor([[40., -30., 4.5, -40., 12., -4.5]]).repeat(2, 1)
    actions = torch.arange(12).reshape(2, 6).float()
    for frame in range(1, 23):
        module.begin_policy()
        before = torque.clone()
        out = module.apply_torque(torque)
        assert torch.equal(torque, before)
        torch.testing.assert_close(out[0], torque[0] * min(frame / 20, 1))
        assert torch.equal(out[1], torque[1])
        assert not module.holding.any()
        assert not module.held_interval.any()
        assert torch.equal(module.effective_action_history(actions), actions)
        assert module.learning_mask.tolist() == [frame > 20, True]
        module.advance()
        assert module.bootstrap_mask(torch.zeros(2, dtype=torch.bool)).tolist() == [frame >= 20, True]
        assert not module.bootstrap_mask(torch.ones(2, dtype=torch.bool)).any()
    assert module.capturing.tolist() == [True, False]


def test_partially_conditioned_multiphysics_interval_and_reset_cannot_relabel_samples():
    module = rig()
    module.reset(torch.tensor([0]), 1024)
    module.episode_steps[0] = 19
    module.begin_policy()
    for _ in range(4):
        module.apply_torque(torch.ones(2, 6))
        module.advance()
    assert module.learning_mask.tolist() == [False, True]
    assert module.conditioned_interval.tolist() == [True, False]
    module.reset(torch.tensor([1]), 1024)
    assert module.learning_mask.tolist() == [False, True]
    assert module.bootstrap_mask(torch.zeros(2, dtype=torch.bool)).tolist() == [True, False]
    module.cancel_reset(torch.tensor([1]))
    assert module.bootstrap_mask(torch.zeros(2, dtype=torch.bool)).all()
    module.begin_policy()
    assert module.learning_mask.all()


def test_v2_disabled_evaluation_consumes_no_rng_and_no_gain_or_mask_changes():
    module = rig(evaluation=True)
    gen = torch.Generator().manual_seed(71)
    before = gen.get_state().clone()
    module.reset(torch.arange(2), 1024, generator=gen)
    assert torch.equal(gen.get_state(), before)
    module.begin_policy()
    effort = torch.randn(2, 6)
    assert torch.equal(module.apply_torque(effort), effort)
    assert module.learning_mask.all() and not module.conditioning.any()


@pytest.mark.parametrize("patch", [
    {"version": "v6_nonstandard_start_v1"}, {"hold_seconds": .2},
    {"release_ramp_seconds": .2}, {"action_history": "clear"},
    {"application": "zero_all_active_motor_effort_after_response"},
    {"capture_seconds": 2.}, {"capture_vx_cap_m_s": 2.}, {"unknown": 1}])
def test_v2_rejects_mixed_or_changed_mechanics(patch):
    with pytest.raises(ValueError):
        validate_nonstandard_start(ramped_start_settings(0) | patch)


def test_v1_cannot_silently_adopt_v2_fields():
    with pytest.raises(ValueError):
        validate_nonstandard_start(nonstandard_start_settings(0) | {"release_ramp_seconds": .1})


@pytest.fixture
def target():
    with TemporaryDirectory(prefix="birth-ramp-test-", dir=ROOT / "runs") as d:
        path = local.write_bundle(ROOT, Path(d) / "config", 512, plan_file=local.BIRTH_RAMP_PLAN_FILE)
        yield json.loads(path.read_text())


def test_target_independent_design_and_exact_nonbirth_contract_identity(target):
    _, old = local.build_bundle(ROOT, ROOT / "runs/source-fixture", 512)
    proof = validate_contracts(old["worker"], target, ROOT)
    assert proof["additional_critic_warmup"] == 0
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    receipt = design_preflight(target, load_chassis_control(ROOT / target["control_math_source"]), ROOT,
        stage="speed", role="train", seed=290927, num_envs=512, block_index=0)
    assert receipt["budget"]["inherited_consumption"] == 1088
    assert receipt["budget"]["update_ceiling"] == 18912
    assert receipt["runtime_expected"]["nonstandard_start"]["release_ramp_seconds"] == .1


@pytest.mark.parametrize("key,value", [("entropy_coef", .005), ("position_iterations", 128),
    ("target_num_envs", 1024), ("gamma", .99), ("initial_noise_std", .05)])
def test_contract_migration_cannot_smuggle_other_changes(target, key, value):
    _, old = local.build_bundle(ROOT, ROOT / "runs/source-fixture", 512)
    source = old["worker"]
    with pytest.raises(ValueError):
        validate_contracts(source | {key: value}, target, ROOT)
    with pytest.raises(ValueError):
        validate_contracts(source, target | {key: value}, ROOT)


def test_v5_refuses_fresh_initialization_and_foreign_named_parent(target):
    from test_local_scratch import invocation
    args = invocation()
    with pytest.raises(ValueError, match="full-state continuation"):
        local.validate_invocation(target, args, ROOT)
    args.resume_local_birth_revision = True
    args.resume = ROOT / "runs/missing-model_final.pt"
    with pytest.raises(ValueError):
        local.validate_invocation(target, args, ROOT)


@pytest.mark.parametrize("field,value", [("update_ceiling", 20000),
    ("prior_experiment_consumption", 0), ("initialization", "random_actor_critic_fresh_optimizer"),
    ("curriculum_origins", {"nonstandard_start": 1024}), ("entropy_coef", .005)])
def test_plan_cannot_reset_budget_or_move_clocks(field, value):
    plan = deepcopy(local.read_plan(ROOT, local.BIRTH_RAMP_PLAN_FILE))
    plan[field] = value
    with pytest.raises(ValueError):
        validate_plan(plan, ROOT)


@pytest.mark.skipif(not Path(REVISION["checkpoint_file"]).exists(), reason="local sealed pilot008 artifact required")
def test_real_sealed_parent_and_first_block_budget_are_authenticated(target):
    from test_local_scratch import invocation
    assert validate_parent(REVISION["checkpoint_file"], target, ROOT)["total_prior_consumption"] == 2112
    args = invocation()
    args.resume = Path(REVISION["checkpoint_file"])
    args.resume_local_birth_revision = True
    args.updates = 128
    assert local.validate_invocation(target, args, ROOT) == 1024
    args.updates = 129
    with pytest.raises(ValueError):
        local.validate_invocation(target, args, ROOT)
    args.resume_local_birth_revision = False
    args.updates = 128
    with pytest.raises(ValueError):
        local.validate_invocation(target, args, ROOT)


def test_actual_gaussian_ppo_excludes_entire_ramp_from_actor_and_critic():
    from test_recovery_rollout_mask import algorithm, observation
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    module = rig()
    module.reset(torch.tensor([0]), 1024)
    alg = algorithm(steps=24, count=2, normalize_obs=True)
    adapter = attach_recovery_rollout_mask(alg, num_envs=2)
    for frame in range(24):
        module.begin_policy()
        module.apply_torque(torch.ones(2, 6))
        module.advance()
        with torch.no_grad():
            alg.act(observation(2, frame / 10))
        rewards = torch.ones(2)
        rewards[~module.learning_mask] = torch.nan
        alg.process_env_step(observation(2, (frame + 1) / 10), rewards, torch.zeros(2), {
            "policy_learning_mask": module.learning_mask.clone(),
            "policy_bootstrap_mask": module.bootstrap_mask(torch.zeros(2, dtype=torch.bool))})
    alg.compute_returns(observation(2, 2.4))
    assert adapter.learning[:, 0].tolist() == [False] * 20 + [True] * 4
    assert alg.storage.returns[:20, 0].eq(0.).all()
    assert alg.storage.advantages[:20, 0].eq(0.).all()
    assert torch.isfinite(alg.storage.returns).all()
    losses = alg.update()
    assert adapter.metrics["valid_transitions"] == 28
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
