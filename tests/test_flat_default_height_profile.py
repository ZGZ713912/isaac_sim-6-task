"""Default-height dynamics must be explicit, paired, and source authenticated."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flat_default_height_profile import (default_height_settings, read_default_height_contract,
    SOURCE, BASE_ENVELOPE, ENVELOPE)
from wheeled_tasks.chassis.skill_commands import profile_command

ROOT = Path(__file__).resolve().parents[1]


def test_actual_catalogue_preserves_static_cases_thresholds_seeds_and_motion_profiles():
    catalogue = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v18.json").read_text())["evaluation"]
    settings = deepcopy(catalogue); settings["protocol_id"] = BASE_ENVELOPE
    before = deepcopy(settings)
    result = default_height_settings(settings, dict(schema="flat-rmcs-packaged-default-height-v1", height_m=.23))
    assert settings == before and result["protocol_id"] == ENVELOPE
    changed = []
    for old, new in zip(before["cases"], result["cases"], strict=True):
        if not any(abs(v) > 0 for v in old["command"][:2]):
            assert old == new
            continue
        changed.append(old["name"])
        expected = deepcopy(old); expected["command"][2] = .23
        if "command" in expected.get("skill", {}): expected["skill"]["command"][2] = .23
        assert new == expected
        if old.get("skill", {}).get("kind") in ("forward", "backward", "rotate", "curve", "start_stop"):
            t = torch.tensor([0., .2, 1., 4., 8., 14.])
            a = profile_command(t, old["command"], old.get("skill", {}))
            b = profile_command(t, new["command"], new.get("skill", {}))
            assert torch.equal(a[:, :2], b[:, :2]) and torch.equal(b[:, 2], torch.full_like(b[:, 2], .23))
    assert changed and result["default_height_moving_cases"] == changed


def test_height_is_read_from_the_exact_authenticated_yaml(tmp_path):
    path = tmp_path / "rmcs_ws/src" / SOURCE; path.parent.mkdir(parents=True)
    path.write_text("wheel_leg_chassis_controller:\n  ros__parameters:\n    default_command_height: 0.23\n    command_height_min: 0.23\n    command_height_max: 0.42\n")
    reference = dict(source_sha256={SOURCE:hashlib.sha256(path.read_bytes()).hexdigest()})
    assert read_default_height_contract(tmp_path, reference)["height_m"] == .23
    path.write_text(path.read_text().replace("0.23", "0.24", 1))
    with pytest.raises(ValueError, match="differs from the authenticated"):
        read_default_height_contract(tmp_path, reference)


@pytest.mark.parametrize("height", [.1, .5, float("nan"), None])
def test_invalid_default_height_does_not_relabel_existing_evidence(height):
    with pytest.raises(ValueError, match="Invalid authenticated"):
        default_height_settings(dict(protocol_id=BASE_ENVELOPE, cases=[]),
            dict(schema="flat-rmcs-packaged-default-height-v1", height_m=height))


def test_short_diagnostic_protocol_cannot_be_relabelled_as_default_height_coverage():
    with pytest.raises(ValueError, match="ordinary RMCS"):
        default_height_settings(dict(protocol_id="flat-base-paired107-diagnostic-ablation-v1"), {})
