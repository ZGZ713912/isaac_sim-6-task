"""Verify the diagnostic changes only later trained-height leg targets."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE = Path(__file__).resolve().parents[1] / "scripts/hold_initial_leg_target_probe.py"
spec = importlib.util.spec_from_file_location("hold_probe", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Probe = module.HoldInitialLegTargetProbe


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(35 * 6).reshape(35, 6).float() / 100.)
        self.calls = 0

    def forward(self, observation):
        self.calls += 1
        return observation["policy"] @ self.weight


def frame(count=42):
    value = torch.zeros(count, 35)
    value[:, 3] = torch.tensor([.23] * 7 + [.305] * 7 + [.42] * 7).repeat(count // 21) * 5.
    return {"policy": value}


def test_first_action_matches_then_only_four_leg_channels_of_mid_lanes_change():
    actor = Actor()
    probe = Probe(actor)
    observation = frame()
    original = deepcopy(observation)
    initial = probe(observation)
    assert torch.equal(initial, observation["policy"] @ actor.weight)
    assert torch.equal(original["policy"], observation["policy"])
    observation["policy"][:, 4:10] += torch.arange(6).float()
    native = observation["policy"] @ actor.weight
    later = probe(observation)
    selected = probe.selected
    assert selected.sum() == 14
    assert torch.equal(later[selected, :4], initial[selected, :4])
    assert torch.equal(later[~selected], native[~selected])
    assert torch.equal(later[:, 4:], native[:, 4:])
    assert not torch.equal(later[selected, :4], native[selected, :4])
    assert actor.calls == probe.calls == 2


def test_no_parameter_rng_or_observation_mutation_and_native_output_is_archived():
    actor = Actor()
    before = deepcopy(actor.state_dict())
    probe = Probe(actor)
    observation = frame()
    state = torch.get_rng_state().clone()
    value = observation["policy"].clone()
    native = value @ actor.weight
    probe(observation)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(value, observation["policy"])
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in before.items())
    assert torch.equal(probe.native_actions[0], native[:21])
    assert probe.native_actions[0].requires_grad is False


@pytest.mark.parametrize("kind", ["ABI", "case_order", "changed_height", "changed_count", "nonfinite"])
def test_invalid_scope_or_native_result_is_rejected(kind):
    actor = Actor()
    probe = Probe(actor)
    observation = frame()
    if kind == "ABI":
        observation["policy"] = observation["policy"][:, :34]
    elif kind == "case_order":
        observation["policy"][0, 3] = .305 * 5.
    elif kind == "nonfinite":
        observation["policy"][0, 4] = float("nan")
    else:
        probe(observation)
        if kind == "changed_height":
            observation["policy"][7, 3] = .42 * 5.
        else:
            observation["policy"] = observation["policy"][:21]
    with pytest.raises(ValueError):
        probe(observation)


def test_short_diagnostic_cannot_claim_complete_record(tmp_path):
    probe = Probe(Actor())
    probe(frame())
    with pytest.raises(ValueError, match="every actor call"):
        probe.finish(tmp_path, None, 256, None)
    assert not list(tmp_path.iterdir())


def test_complete_record_preserves_native_outputs_and_logs_existing_event_writer(tmp_path):
    class Writer:
        def __init__(self):
            self.rows = []
            self.flushed = False

        def add_scalar(self, tag, value, step):
            self.rows.append((tag, value, step))

        def flush(self):
            self.flushed = True

    actor = Actor()
    probe = Probe(actor)
    observation = frame(21)
    with torch.inference_mode():
        for _ in range(1600):
            probe(observation)
    writer = Writer()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = probe.finish(tmp_path, writer, 256, digest)
    path = tmp_path / "unmodified_actor_actions.npz"
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["action_policy"].shape == (1600, 21, 6)
        assert np.array_equal(arrays["action_policy"][0], probe.native_actions[0].numpy())
        assert np.flatnonzero(arrays["selected"]).tolist() == list(range(7, 14))
    assert result["original_actions"]["sha256"] == digest(path)
    assert result["actor_calls"] == actor.calls == 1600
    assert result["export_forbidden"] and not result["adoption_by_training"]
    assert not result["learned_policy"] and not result["behavior_qualification"]
    assert writer.flushed
    assert writer.rows == [("LegHoldProbe/actor_calls", 1600, 256),
        ("LegHoldProbe/selected_original_cases", 7, 256),
        ("LegHoldProbe/native_wheel_output_max_error", 0., 256)]
    with pytest.raises(ValueError, match="Preserve"):
        probe.finish(tmp_path, writer, 256, digest)
