"""Actual pinned PPO/Adam equivalence, including compact conditioning masks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from rsl_rl.storage import RolloutStorage

from wheeled_algo.native_ppo_audit import NativePPOAudit, validate_local_audit_request
from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask
from test_recovery_rollout_mask import algorithm, collect


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("per_batch", [False, True])
def test_observation_preserves_native_loss_models_adam_rng_and_normalizers(masked, per_batch):
    # Build a real learned Adam parent, then load its complete state in both arms.
    parent = algorithm(normalize_obs=True, per_batch=per_batch)
    attach_recovery_rollout_mask(parent, num_envs=2)
    all_valid = torch.ones(6, 2, dtype=torch.bool)
    obs = collect(parent, all_valid)
    parent.compute_returns(obs)
    parent.update()
    saved = deepcopy(parent.save())
    learning = all_valid.clone()
    if masked:
        learning[1:3, 0] = False
    arms = []
    for observed in (False, True):
        alg = algorithm(normalize_obs=True, per_batch=per_batch)
        alg.load(deepcopy(saved), load_cfg=None, strict=True)
        alg.num_learning_epochs, alg.num_mini_batches = 5, 4
        attach_recovery_rollout_mask(alg, num_envs=2)
        obs = collect(alg, learning)
        alg.compute_returns(obs)
        generator = RolloutStorage.mini_batch_generator
        audit = NativePPOAudit(alg) if observed else None
        torch.manual_seed(901)
        try:
            result = alg.update()
            rng = torch.get_rng_state().clone()
        finally:
            if audit:
                audit.close()
        assert RolloutStorage.mini_batch_generator is generator
        assert "step" not in alg.optimizer.__dict__
        arms.append((result, deepcopy(alg.save()), rng))
        if audit:
            report = audit.report()
            assert report["optimizer_calls"] == report["committed_optimizer_calls"] == 20
            assert report["all_recorded_values_finite"] and report["closed"]
            assert report["rows"][0]["adam_step_before_min"] == 4
            assert report["rows"][-1]["adam_step_after_max"] == 24
            # Masked compact population is 10; native drops its 2 remainder
            # samples, yielding four batches of 2. Unmasked yields batches of 3.
            assert {row["samples"] for row in report["rows"]} == {2 if masked else 3}
            assert all(row["actor_parameter_delta_l2"] > 0 and row["critic_parameter_delta_l2"] > 0
                       for row in report["rows"])
            assert max(row["critic_gradient_l2_before_clip"] for row in report["rows"]) > alg.max_grad_norm
            assert max(row["critic_gradient_l2_after_clip"] for row in report["rows"]) <= alg.max_grad_norm + 1e-6
    assert_tree_equal(arms[0], arms[1])


def test_native_clipping_metrics_and_cleanup_on_failed_optimizer():
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    obs = collect(alg, torch.ones(6, 2, dtype=torch.bool))
    alg.compute_returns(obs)
    # Force known ratios and flat clipped-value regions without changing observer.
    alg.storage.actions_log_prob += 2
    alg.storage.values -= 1
    alg.storage.returns += 20
    generator = RolloutStorage.mini_batch_generator
    audit = NativePPOAudit(alg)
    try:
        alg.update()
        row = audit.rows[0]
        assert row["ratio_clip_fraction"] == 1
        assert row["value_clipped_flat_fraction"] == 1
    finally:
        audit.close()
    assert RolloutStorage.mini_batch_generator is generator
    assert not alg.critic._forward_hooks
    assert all(not parameter._backward_hooks for model in (alg.actor, alg.critic) for parameter in model.parameters())
    audit.close()  # Idempotent.


def test_failed_step_is_preserved_and_previous_step_override_is_restored():
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    obs = collect(alg, torch.ones(6, 2, dtype=torch.bool))
    alg.compute_returns(obs)
    def fail():
        raise RuntimeError("fixture Adam failure")
    alg.optimizer.step = fail
    generator = RolloutStorage.mini_batch_generator
    audit = NativePPOAudit(alg)
    with pytest.raises(ValueError, match="Only one"):
        NativePPOAudit(alg)
    try:
        with pytest.raises(RuntimeError, match="fixture Adam failure"):
            alg.update()
    finally:
        audit.close()
    assert alg.optimizer.step is fail
    assert RolloutStorage.mini_batch_generator is generator
    assert audit.report()["committed_optimizer_calls"] == 0
    assert len(audit.rows) == 1 and not audit.rows[0]["optimizer_committed"]


@pytest.mark.parametrize("change", [dict(local=False), dict(resume=None), dict(updates=2),
                                   dict(scene=1), dict(revision=True)])
def test_bounded_request_rejects_other_training_paths(change):
    request = dict(local=True, resume=Path("parent.pt"), updates=1, scene=0, revision=False)
    request.update(change)
    with pytest.raises(ValueError, match="one same-contract"):
        validate_local_audit_request(True, **request)
    validate_local_audit_request(False, **request)


def test_original_logger_receives_stable_tags_and_optimizer_step():
    alg = algorithm()
    attach_recovery_rollout_mask(alg, num_envs=2)
    obs = collect(alg, torch.ones(6, 2, dtype=torch.bool))
    alg.compute_returns(obs)
    audit = NativePPOAudit(alg)
    try:
        alg.update()
    finally:
        audit.close()
    class Writer:
        def __init__(self):
            self.rows = []
        def add_scalar(self, tag, value, step):
            self.rows.append((tag, value, step))
    writer = Writer()
    audit.log(writer, 2944)
    assert all(tag.startswith("LocalDiagnostics/NativePPOAudit/") and step == 2944
               for tag, value, step in writer.rows)
    assert ("LocalDiagnostics/NativePPOAudit/optimizer_calls", 4, 2944) in writer.rows


def test_invalid_measurement_is_json_safe_without_hiding_native_failure():
    audit = NativePPOAudit(algorithm())
    try:
        audit.rows.append(dict(optimizer_committed=False, value_mse=float("nan")))
        report = audit.report()
        assert not report["all_recorded_values_finite"]
        assert report["rows"][0]["value_mse"] is None
        json.dumps(report, allow_nan=False)
    finally:
        audit.close()
