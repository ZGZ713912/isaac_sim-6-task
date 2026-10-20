"""Counterfactual branch isolation must preserve the other trained branch."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_flat_base import diagnostic_ablation
from wheeled_algo.command_residual import CommandResidualActor
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor
from wheeled_algo.onnx_actor_import import recover_actor


def means():
    source = Path(__file__).resolve().parents[1] / "models/wheel_leg_flat_base_20261018/source.onnx"
    original, _ = recover_actor(source)
    parent = CommandResidualActor(original)
    current = GuardedFeedbackActor(parent)
    with torch.no_grad():
        current.command.residual[2].bias.fill_(.2)
        current.feedback[2].bias.fill_(.3)
    return parent, current


def test_feedback_off_changes_only_feedback_last_layer():
    _, current = means()
    before = deepcopy(current.state_dict())
    result = diagnostic_ablation(current, "feedback_off")
    for name, value in result.state_dict().items():
        if name.startswith("feedback.2."):
            assert not value.count_nonzero()
        else:
            assert torch.equal(value, before[name])
    x = torch.randn(19, 35)
    assert torch.equal(result(x), result.command(x))


def test_parent_command_preserves_trained_feedback():
    parent, current = means()
    before = deepcopy(current.feedback.state_dict())
    result = diagnostic_ablation(current, "parent_command", parent)
    assert all(torch.equal(v, parent.state_dict()[n]) for n, v in result.command.state_dict().items())
    assert all(torch.equal(v, before[n]) for n, v in result.feedback.state_dict().items())
    with pytest.raises(ValueError, match="command-only"):
        diagnostic_ablation(current, "parent_command")


def test_full_is_exact_for_both_architectures_and_real_ablation_rejects_parent():
    parent, current = means()
    for mean in (parent, current):
        before = deepcopy(mean.state_dict())
        assert diagnostic_ablation(mean, "full") is mean
        assert all(torch.equal(v, before[n]) for n, v in mean.state_dict().items())
    for variant in ("feedback_off", "parent_command", "tracking_gate"):
        with pytest.raises(ValueError, match="guarded feedback"):
            diagnostic_ablation(parent, variant)


def test_tracking_gate_diagnostic_changes_function_without_changing_weights():
    _, current = means()
    before = deepcopy(current.state_dict())
    diagnostic_ablation(current, "tracking_gate")
    assert current.tracking_gate
    assert all(torch.equal(v, before[n]) for n,v in current.state_dict().items())
    with pytest.raises(ValueError, match="legacy parent"):
        diagnostic_ablation(current, "tracking_gate")
