"""Getup may translate; standing starts holding its completion position."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_prone_motion import configuration, source_hashes
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.getup_translation import CONTRACT, enabled, validate
from wheeled_tasks.chassis.prone_motion_cli import train_parser, training_options
from wheeled_tasks.chassis.quiet_standing import QuietStanding, standing_request_mask

PATH = ROOT / "contracts/wheel_leg_prone_motion_v4_free_translation_v1.json"
PLAN = json.loads(PATH.read_text())
PARENT = json.loads((ROOT / "contracts/wheel_leg_prone_motion_v4_cold_lifecycle_v1.json").read_text())


def test_new_plan_only_changes_getup_translation():
    new = deepcopy(PLAN)
    assert validate(new.pop("getup_translation")) == CONTRACT
    assert new == PARENT
    assert task.validate(PLAN) == PLAN and task.validate(PARENT) == PARENT
    assert enabled(PLAN) and not enabled(PARENT)
    assert PLAN["rise_training_abort_excursion_m"] == .35  # Diagnostic reference retained.


@pytest.mark.parametrize("field", list(CONTRACT))
def test_partial_contract_cannot_silently_disable_a_rule(field):
    bad = deepcopy(PLAN)
    bad["getup_translation"].pop(field)
    with pytest.raises(ValueError, match="translation"):
        task.validate(bad)


@pytest.mark.parametrize("distance", [.351, 1., 3.])
def test_active_getup_can_translate_without_terminating_but_keeps_timeout(distance):
    args = (torch.tensor([[0., 0., -1.]]), torch.tensor([.22]), torch.tensor([200.]),
            torch.tensor([distance]), torch.tensor([1.]), torch.tensor([True]), torch.tensor([False]))
    old, new = task.failures(*args, PARENT), task.failures(*args, PLAN)
    assert old["rise_excursion"].item() and not new["rise_excursion"].item()
    assert not new["refall"].item() and not new["rise_timeout"].item()
    expired = task.failures(*args[:4], torch.tensor([20.]), *args[5:], PLAN)
    assert expired["rise_timeout"].item()


def terms(rising, excursion, anchor_error, *, free):
    n = len(rising)
    return task.reward_terms(torch.tensor([[0., 0., -1.]] * n), torch.full((n,), .305),
        torch.zeros(n, 3), torch.zeros(n, 3), torch.tensor([[0., 0., .305]] * n),
        torch.zeros(n, 6), torch.zeros(n, 6), torch.zeros(n), torch.tensor(excursion),
        torch.tensor(anchor_error), torch.tensor(rising), torch.ones(n, dtype=torch.bool),
        position_tail=PLAN["quiet_position_tail"], free_rise_translation=free)


def test_only_rising_displacement_cost_is_removed_and_standing_cost_is_unchanged():
    old = terms([True, False], [1., 1.], [1., .2], free=False)
    new = terms([True, False], [1., 1.], [1., .2], free=True)
    assert old["rise_position"][0] < 0 and torch.equal(new["rise_position"], torch.zeros(2))
    assert new["quiet_position"][0] == 0 and new["quiet_position"][1] < 0
    for key in set(old) - {"rise_position"}:
        assert torch.equal(old[key], new[key]), key


def test_completion_starts_a_new_standing_anchor_at_the_arrival_position():
    quiet = QuietStanding(1, "cpu", .01)
    command, passive, recovery = torch.zeros(1, 3), torch.tensor([False]), torch.tensor([True])
    for x in [0., .2, .6, 1.]:
        request = standing_request_mask(command, passive, recovery, torch.tensor([False]))
        quiet.update(request, torch.tensor([[x, 0.]]), torch.zeros(1, 6), torch.zeros(1, 6))
        assert not quiet.active.any()
    request = standing_request_mask(command, passive, recovery, torch.tensor([True]))
    quiet.update(request, torch.tensor([[1., 0.]]), torch.zeros(1, 6), torch.zeros(1, 6))
    assert torch.equal(quiet.anchor, torch.tensor([[1., 0.]]))
    assert torch.equal(quiet.displacement(torch.tensor([[1., 0.]])), torch.zeros(1, 2))
    assert quiet.displacement(torch.tensor([[1.1, 0.]])).norm().item() == pytest.approx(.1)


def test_after_completion_body_contact_and_refall_protection_still_apply():
    args = (torch.tensor([[0., 0., -1.]]), torch.tensor([.305]), torch.tensor([200.]),
            torch.tensor([1.]), torch.tensor([2.]), torch.tensor([False]), torch.tensor([False]))
    assert task.failures(*args, PLAN)["refall"].item()
    assert not task.failures(*args, PLAN)["rise_excursion"].item()


def test_cli_config_and_frozen_sources_carry_the_explicit_translation_contract(tmp_path):
    args = train_parser().parse_args(["--plan", str(PATH), "--num-envs", "256", "--seed", "107"])
    _, resolved, _ = training_options(args, ROOT)
    assert task.validate(resolved) == resolved
    cfg, settings, _ = configuration(tmp_path, resolved)
    written = json.loads((tmp_path / "task_contract.json").read_text())
    assert written["getup_translation"] == CONTRACT
    assert settings["prone_motion_task"]["getup_translation"] == CONTRACT
    hashes = source_hashes(PATH, resolved)
    for name in ["src/wheeled_tasks/chassis/getup_translation.py", "tests/test_getup_translation.py",
                 "docs/GETUP_TRANSLATION_20261103.md", str(PATH.relative_to(ROOT))]:
        assert name in hashes
