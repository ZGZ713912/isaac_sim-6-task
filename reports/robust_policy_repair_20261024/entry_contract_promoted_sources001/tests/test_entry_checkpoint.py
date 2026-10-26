"""Entry provenance must reject mutated heads, scopes and learning clocks."""
from copy import deepcopy

import pytest
import torch

from wheeled_algo.entry_checkpoint import make_entry_learning_record, validate_entry_learning


def pair(tmp_path):
    actor = {"mlp.log_std": torch.zeros(6), "mlp.mean.recovery.0.weight": torch.ones(3, 2)}
    for layer in (0, 2, 4):
        actor[f"mlp.mean.entry_recovery.{layer}.weight"] = torch.ones(3, 2)
        actor[f"mlp.mean.entry_recovery.{layer}.bias"] = torch.zeros(3)
    settings = dict(gap_entry_recovery_split_version="frozen-v1", gap_entry_optimizer_layout="append-v1")
    source = dict(actor_state_dict=actor, infos=dict(completed_updates=1501, settings=settings))
    path = tmp_path / "source.pt"
    torch.save(source, path)
    final = deepcopy(source)
    final["actor_state_dict"]["mlp.mean.entry_recovery.4.bias"].add_(.1)
    final["infos"].update(completed_updates=1509, entry_recovery_only_tuning=True)
    final["infos"]["entry_recovery_learning"] = make_entry_learning_record(
        path, source, final["actor_state_dict"], 1509, 8)
    return path, source, final


def test_trained_entry_record_preserves_scope_and_inputs(tmp_path):
    _, source, final = pair(tmp_path)
    before = deepcopy(final)
    validate_entry_learning(final, source)
    assert final["infos"] == before["infos"]
    assert all(torch.equal(v, before["actor_state_dict"][k]) for k, v in final["actor_state_dict"].items())


@pytest.mark.parametrize("key", ["mlp.mean.entry_recovery.4.bias", "mlp.log_std", "mlp.mean.recovery.0.weight"])
def test_record_rejects_changed_tensor(tmp_path, key):
    _, source, final = pair(tmp_path)
    final["actor_state_dict"][key].add_(.01)
    with pytest.raises(ValueError):
        validate_entry_learning(final, source)


@pytest.mark.parametrize("key,value", [("new_optimizer_updates", True), ("new_optimizer_updates", -1),
                                      ("completed_learning_clock", 1510), ("source_learning_clock", 1500)])
def test_record_rejects_bad_clock(tmp_path, key, value):
    _, source, final = pair(tmp_path)
    final["infos"]["entry_recovery_learning"][key] = value
    with pytest.raises(ValueError):
        validate_entry_learning(final, source)


def test_record_builder_rejects_changed_frozen_network(tmp_path):
    path, source, final = pair(tmp_path)
    final["actor_state_dict"]["mlp.log_std"].add_(.1)
    with pytest.raises(ValueError):
        make_entry_learning_record(path, source, final["actor_state_dict"], 1509, 8)


def test_record_rejects_missing_optimizer_contract(tmp_path):
    _, source, final = pair(tmp_path)
    final["infos"]["settings"].pop("gap_entry_optimizer_layout")
    with pytest.raises(ValueError):
        validate_entry_learning(final, source)
