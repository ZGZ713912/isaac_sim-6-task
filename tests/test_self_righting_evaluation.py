import json
import math
from pathlib import Path

from evaluate_self_righting import cases_for

ROOT = Path(__file__).resolve().parents[1]


def test_independent_matrix_keeps_every_family_class_and_heading():
    bank = json.loads((ROOT / "model/recovery_multi_v202_fine_pd200_20261105/pose_bank.json").read_text())
    cases, missing = cases_for(bank)
    assert len(cases) + len(missing) == 7 * 4 * 2
    assert len({(r["family"], r["leg_class"], r["heading_rad"]) for r in cases + missing}) == 56
    for case in cases:
        source = bank["poses"][case["pose_index"]]["recovery"]
        assert source["split"] == "holdout" and not source.get("dynamic")


def test_missing_admission_remains_a_failed_trial():
    cases, missing = cases_for({"poses": []})
    assert cases == [] and len(missing) == 56
    assert not any(row["passed"] for row in missing)


def test_single_posture_contract_has_no_unseen_pose_claim():
    bank = json.loads((ROOT / "model/recovery_standard_v202_fine_pd200_20261105/pose_bank.json").read_text())
    cases, missing = cases_for(bank, standard=True)
    assert missing == [] and len(cases) == 2
    assert {c["heading_rad"] for c in cases} == {0., math.pi / 2}
    assert {c["split"] for c in cases} == {"standard"}
