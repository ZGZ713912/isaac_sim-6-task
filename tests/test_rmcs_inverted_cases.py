import json
from pathlib import Path

from evaluate_gap_robustness import cases


def test_fully_inverted_reference_coverage_does_not_change_the_old_suite():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "reports/real_robot_robustness_20261024/pose_bank_gap002.json").read_text())
    new = json.loads((root / "reports/robust_policy_repair_20261024/pose_bank_rmcs003.json").read_text())
    assert new["poses"][:len(old["poses"])] == old["poses"]
    assert cases(old) == cases(new)
    selected = cases(new, group="rmcs-inverted")
    assert len(selected) == 72 and len({c["name"] for c in selected}) == 72
    assert {c["imu_case"] for c in selected} == {"clean", "combined"}
    assert all(c["recovery"] and c["pose_index"] >= len(old["poses"]) for c in selected)
    for case in selected:
        pose = new["poses"][case["pose_index"]]
        assert abs(pose["roll_deg"]) == 180. or abs(pose["pitch_deg"]) == 180.
