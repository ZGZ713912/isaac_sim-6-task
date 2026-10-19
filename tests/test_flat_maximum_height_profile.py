"""A high static stance cannot substitute for the packaged moving-height limit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flat_default_height_profile import BASE_ENVELOPE, SOURCE
from flat_maximum_height_profile import ENVELOPE, maximum_height_settings, read_maximum_height_contract
from wheeled_tasks.chassis.skill_commands import profile_command

ROOT = Path(__file__).resolve().parents[1]


def test_authenticated_upper_boundary_and_real_motion_profiles(tmp_path):
    path = tmp_path / "rmcs_ws/src" / SOURCE
    path.parent.mkdir(parents=True)
    path.write_text("wheel_leg_chassis_controller:\n  ros__parameters:\n    default_command_height: 0.23\n    command_height_min: 0.23\n    command_height_max: 0.42\n")
    reference = dict(source_sha256={SOURCE: hashlib.sha256(path.read_bytes()).hexdigest()})
    contract = read_maximum_height_contract(tmp_path, reference)
    assert contract["height_m"] == .42 and contract["command_height_bounds_m"] == [.23, .42]
    settings = json.loads((ROOT / "contracts/v6_task_campaign_speed_worker_v18.json").read_text())["evaluation"]
    settings["protocol_id"] = BASE_ENVELOPE
    before = deepcopy(settings)
    result = maximum_height_settings(settings, contract)
    assert settings == before and result["protocol_id"] == ENVELOPE
    moving = 0
    for old, new in zip(before["cases"], result["cases"], strict=True):
        if not any(abs(v) > 0 for v in old["command"][:2]):
            assert old == new
            continue
        moving += 1
        assert new["command"][:2] == old["command"][:2] and new["command"][2] == .42
        if old.get("skill", {}).get("kind") in ("forward", "backward", "rotate", "curve", "start_stop"):
            t = torch.tensor([0., .2, 1., 4., 8., 14.])
            a = profile_command(t, old["command"], old.get("skill", {}))
            b = profile_command(t, new["command"], new.get("skill", {}))
            assert torch.equal(a[:, :2], b[:, :2])
            assert torch.equal(b[:, 2], torch.full_like(b[:, 2], .42))
    assert moving > 0
    path.write_text(path.read_text().replace("0.42", "0.40"))
    with pytest.raises(ValueError, match="differs from the authenticated"):
        read_maximum_height_contract(tmp_path, reference)


def test_nominal_height_cannot_be_labelled_as_the_upper_boundary():
    with pytest.raises(ValueError, match="Invalid authenticated maximum-height"):
        maximum_height_settings(dict(protocol_id=BASE_ENVELOPE), dict(
            schema="flat-rmcs-packaged-maximum-height-v1", height_m=.305,
            command_height_bounds_m=[.23, .42]))
