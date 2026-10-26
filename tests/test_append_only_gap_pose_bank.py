"""An added reset library cannot silently alter old mechanics or indices."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gap_bank_migration", ROOT / "scripts/migrate_gap_pose_bank.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def banks():
    old = dict(poses=[dict(name="old", full_joint_positions={"hip": 0.})],
               asset_manifest_sha256="asset", state_writes="episode_reset_only")
    new = deepcopy(old)
    new["poses"].append(dict(name="new", full_joint_positions={"hip": .1}, rmcs_reference=True))
    new["extension"] = dict(source_sha256="source", original_pose_count=1,
                            appended_pose_count=1, original_poses_unchanged=True)
    return old, new


def test_valid_extension_retains_all_old_pose_indices():
    a, b = banks()
    assert module.validate_append_only_bank(a, b, "source")["original_poses_and_indices_exact"]


@pytest.mark.parametrize("change", ["old_pose", "reorder", "asset", "write_contract", "source_hash",
                                      "count", "duplicate_name", "missing_joint", "nonfinite", "unproven_reference"])
def test_bank_corruption_is_rejected_before_checkpoint_changes(change):
    a, b = banks()
    if change == "old_pose":
        b["poses"][0]["full_joint_positions"]["hip"] = .2
    elif change == "reorder":
        b["poses"].reverse()
    elif change == "asset":
        b["asset_manifest_sha256"] = "other"
    elif change == "write_contract":
        b["state_writes"] = "each_step"
    elif change == "source_hash":
        b["extension"]["source_sha256"] = "other"
    elif change == "count":
        b["extension"]["appended_pose_count"] = 3
    elif change == "duplicate_name":
        b["poses"][1]["name"] = "old"
    elif change == "missing_joint":
        b["poses"][1]["full_joint_positions"] = {}
    elif change == "nonfinite":
        b["poses"][1]["full_joint_positions"]["hip"] = float("nan")
    else:
        b["poses"][1]["rmcs_reference"] = False
    with pytest.raises(ValueError):
        module.validate_append_only_bank(a, b, "source")
