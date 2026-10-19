"""A good low-speed pilot must not hide deterioration at unpractised commands."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_flat_base import validate_large_block_evidence


def evidence():
    basic = ["stand_230mm", "stand_305mm", "stand_430mm", "forward_05", "backward_05", "rotate_1",
             "rotate_1_reverse", "start_stop_05"]
    envelope = dict(status="evaluated", checkpoint_sha256="candidate", source_onnx_sha256="source", seed=1,
        asset_manifest_sha256="asset", model_spec_sha256="model", settings=dict(protocol_id="envelope"),
        policy_hz=50, physics_hz=200, result=dict(passed=False, cases={}))
    for direction, sign in (("forward", 1), ("backward", -1)):
        for speed in (1, 2, 3):
            envelope["result"]["cases"][f"{direction}_{speed}"] = dict(command=[sign * speed, 0, .305], episodes=4,
                checks=dict(survival=True, velocity=False), vx_mae_m_s=.2, yaw_mae_rad_s=.01,
                height_mae_m=.001, stand_drift_max_m=0., closure_gap_max_m=.0001, passed=False)
    for name in ("rotate_4", "rotate_4_reverse"):
        envelope["result"]["cases"][name] = deepcopy(envelope["result"]["cases"]["forward_1"])
        envelope["result"]["cases"][name]["command"] = [0., -4. if name.endswith("reverse") else 4., .305]
    baseline = deepcopy(envelope)
    baseline.pop("checkpoint_sha256")
    pilot = deepcopy(envelope)
    pilot["settings"]["protocol_id"] = "flat-base-paired107-fixed-v1"
    pilot["result"] = dict(passed=True, cases={name: {} for name in basic})
    return pilot, envelope, baseline


def test_preserved_original_allows_learning_after_basic_accuracy_passes():
    assert validate_large_block_evidence(*evidence(), "candidate", "source")["no_lost_baseline_checks"]


def test_large_command_error_regression_is_caught_even_if_old_absolute_check_failed():
    pilot, envelope, baseline = evidence()
    envelope["result"]["cases"]["forward_3"]["vx_mae_m_s"] = .24
    with pytest.raises(ValueError, match="tracking regressed"):
        validate_large_block_evidence(pilot, envelope, baseline, "candidate", "source")


@pytest.mark.parametrize("field,value", (("checkpoint_sha256", "other"), ("physics_hz", 1000)))
def test_evidence_from_another_candidate_or_clock_cannot_open_large_training(field, value):
    pilot, envelope, baseline = evidence()
    envelope[field] = value
    with pytest.raises(ValueError, match="exact checkpoint"):
        validate_large_block_evidence(pilot, envelope, baseline, "candidate", "source")
