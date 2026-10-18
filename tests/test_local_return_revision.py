"""Authenticate the longer return window and preserve native learning state."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from wheeled_tasks.chassis import local_scratch as local
from wheeled_tasks.chassis.local_return_revision import (
    REVISION, validate_contracts, validate_parent, validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def target():
    with TemporaryDirectory(prefix="return-window-test-", dir=ROOT / "runs") as directory:
        path = local.write_bundle(ROOT, Path(directory) / "config", 512,
                                  plan_file=local.RETURN_PLAN_FILE)
        yield json.loads(path.read_text())


def source():
    return json.loads((Path(REVISION["checkpoint_file"]).parent / "contract.json").read_text())


def test_exact_independent_design_and_nonuniform_history_budget(target):
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    previous = source()
    receipt = validate_contracts(previous, target, ROOT)
    assert receipt["additional_critic_warmup"] == 0
    assert (previous["num_steps_per_env"], target["num_steps_per_env"]) == (96, 384)
    assert target["lam"] == 1. and target["gamma"] == previous["gamma"]
    assert target["entropy_coef"] == previous["entropy_coef"] == .0005
    assert target["nonstandard_start"] == previous["nonstandard_start"]
    assert target["locomotion_difficulty"] == previous["locomotion_difficulty"]
    preflight = design_preflight(target, load_chassis_control(ROOT / target["control_math_source"]), ROOT,
        stage="speed", role="train", seed=290927, num_envs=512, block_index=0)
    budget = preflight["budget"]
    assert budget["inherited_consumption"] == 1216
    assert budget["update_ceiling"] == 18784
    assert budget["transition_ceiling"] == 132120576 + (18784 - 2688) * 512 * 384
    assert budget["transition_ceiling"] != 18784 * 512 * 384
    assert budget["simulated_seconds_per_env_ceiling"] == 32194.56
    assert 1216 + 2688 + 128 == 4032


@pytest.mark.parametrize("field,value", [
    ("num_steps_per_env", 192), ("lam", .99), ("gamma", .99),
    ("entropy_coef", 0.), ("policy_dt", .02), ("physics_dt", .001),
    ("position_iterations", 128), ("target_num_envs", 1024),
    ("ppo_minibatch_samples", 12288), ("curriculum_critic_warmup_updates", 10),
    ("initial_noise_std_policy_order", [.2] * 6), ("termination_event_cost", 4.),
    ("nonstandard_start", {}), ("locomotion_difficulty", {}),
])
def test_other_contract_changes_rejected(target, field, value):
    with pytest.raises(ValueError):
        validate_contracts(source(), target | {field: value}, ROOT)
    source_value = 49152 if field == "ppo_minibatch_samples" else value
    with pytest.raises(ValueError):
        validate_contracts(source() | {field: source_value}, target, ROOT)


@pytest.mark.parametrize("field,value", [
    ("update_ceiling", 18912), ("prior_experiment_consumption", 1088),
    ("initialization", "random_actor_critic_fresh_optimizer"),
    ("entropy_coef", 0.), ("curriculum_origins", {"locomotion_difficulty": 2688}),
    ("ppo_timing", {"num_steps_per_env": 384, "lam": .99}),
])
def test_plan_cannot_refund_restart_or_change_other_timing(field, value):
    plan = deepcopy(local.read_plan(ROOT, local.RETURN_PLAN_FILE))
    with pytest.raises(ValueError):
        validate_plan(plan | {field: value}, ROOT)


def test_parent_is_exact_finalized_treatment2688(target):
    proof = validate_parent(REVISION["checkpoint_file"], target, ROOT)
    assert proof["parent_updates"] == 2688 and proof["total_prior_consumption"] == 3904
    assert proof["parent_training_transitions"] == 132120576
    parent = Path(REVISION["checkpoint_file"]).parent
    for wrong in (parent / "initial_state/model_final.pt", parent.parent / "pilot014/model_final.pt"):
        with pytest.raises(ValueError):
            validate_parent(wrong, target, ROOT)


def test_first_named_block_requires_full_state_and_is_bounded(target):
    from test_local_scratch import invocation
    args = invocation()
    args.updates = 128
    with pytest.raises(ValueError, match="full-state continuation"):
        local.validate_invocation(target, args, ROOT)
    args.resume = Path(REVISION["checkpoint_file"])
    args.resume_local_return_revision = True
    assert local.validate_invocation(target, args, ROOT) == 2688
    args.updates = 129
    with pytest.raises(ValueError):
        local.validate_invocation(target, args, ROOT)
    args.updates = 128
    args.resume_local_return_revision = False
    with pytest.raises(ValueError):
        local.validate_invocation(target, args, ROOT)
    args.resume_local_return_revision = args.resume_local_entropy_revision = True
    with pytest.raises(ValueError, match="exactly one"):
        local.validate_invocation(target, args, ROOT)


@pytest.mark.parametrize("field,value", [
    ("consumed_updates", 0), ("learning_rate_scale", 1.), ("stage_actor_update", 0),
    ("actor_updates_consumed", 2688), ("global_actor_update_offset", 1),
])
def test_external_clocks_and_budget_reset_rejected(target, field, value):
    from test_local_scratch import invocation
    args = invocation()
    args.resume = Path(REVISION["checkpoint_file"])
    args.resume_local_return_revision = True
    args.updates = 128
    setattr(args, field, value)
    with pytest.raises(ValueError):
        local.validate_invocation(target, args, ROOT)


@pytest.mark.parametrize("row", REVISION["evidence"])
def test_every_bound_evidence_hash_is_required(row, monkeypatch):
    original = local.digest
    monkeypatch.setattr(local, "digest", lambda path: "0" * 64
        if Path(path) == ROOT / row["file"] else original(path))
    with pytest.raises(ValueError, match="evidence changed"):
        local.read_plan(ROOT, local.RETURN_PLAN_FILE)


def test_384_native_lambda_one_returns_are_discounted_segment_sums():
    from test_recovery_rollout_mask import algorithm, collect
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    alg = algorithm(steps=384, per_batch=True)
    alg.gamma = .9974905699336811
    attach_recovery_rollout_mask(alg, num_envs=2)
    learning = torch.ones(384, 2, dtype=torch.bool)
    learning[:20, 0] = False
    learning[140:160, 1] = False
    bootstrap = torch.ones_like(learning)
    bootstrap[139, 1] = False
    bootstrap[-1, 0] = False
    dones = torch.zeros_like(learning)
    dones[300, 0] = True
    obs = collect(alg, learning, bootstrap, dones)
    tail = alg.critic(obs).detach().squeeze(-1).double()
    alg.compute_returns(obs)
    expected = torch.zeros(384, 2, dtype=torch.float64)
    # Independent direct n-step sums, including invalid ownership and reset cuts.
    for start in range(384):
        for lane in range(2):
            if not learning[start, lane]:
                continue
            discount = 1.
            for tick in range(start, 384):
                expected[start, lane] += discount * (tick + lane + 1)
                if dones[tick, lane] or not bootstrap[tick, lane]:
                    break
                if tick + 1 < 384 and not learning[tick + 1, lane]:
                    break
                discount *= alg.gamma
                if tick == 383:
                    expected[start, lane] += discount * tail[lane]
    torch.testing.assert_close(alg.storage.returns.squeeze(-1).double(), expected, rtol=3e-5, atol=.005)
    assert not torch.count_nonzero(alg.storage.advantages.squeeze(-1)[~learning])


def test_native_optimizer_state_survives_window_change_and_advances():
    from test_recovery_rollout_mask import algorithm, collect
    from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
    old = algorithm(steps=96)
    old.gamma, old.lam = .9974905699336811, .9872585449014338
    attach_recovery_rollout_mask(old, num_envs=2)
    obs = collect(old, torch.ones(96, 2, dtype=torch.bool))
    old.compute_returns(obs)
    old.update()
    saved = deepcopy(old.save())
    new = algorithm(steps=384)
    new.gamma = old.gamma
    attach_recovery_rollout_mask(new, num_envs=2)
    assert new.load(deepcopy(saved), None, strict=True)
    restored = new.save()
    for key in ("actor_state_dict", "critic_state_dict"):
        for name, tensor in saved[key].items():
            assert torch.equal(tensor, restored[key][name])
    for parameter, state in saved["optimizer_state_dict"]["state"].items():
        for name, tensor in state.items():
            assert torch.equal(tensor, restored["optimizer_state_dict"]["state"][parameter][name])
    learning = torch.ones(384, 2, dtype=torch.bool)
    learning[:20, 0] = False
    obs = collect(new, learning)
    new.compute_returns(obs)
    new.update()
    for parameter, state in new.optimizer.state_dict()["state"].items():
        assert state["step"] == saved["optimizer_state_dict"]["state"][parameter]["step"] + 4
        assert all(torch.isfinite(tensor).all() for tensor in state.values())
    for key, model in (("actor_state_dict", new.actor), ("critic_state_dict", new.critic)):
        assert any(not torch.equal(tensor, model.state_dict()[name])
                   for name, tensor in saved[key].items())
    assert new.storage.step == 0 and new.storage.num_transitions_per_env == 384
