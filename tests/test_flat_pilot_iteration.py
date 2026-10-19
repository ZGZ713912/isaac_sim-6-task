"""The bounded pilot driver must stop on regression and reject mixed artifacts."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from iterate_flat_base import assess


def artifacts(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"a sealed candidate fixture")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    names = ["stand_230mm", "stand_305mm", "stand_430mm", "forward_05", "backward_05", "rotate_1",
             "rotate_1_reverse", "start_stop_05"]
    def trial(cases, hz, protocol):
        return dict(status="evaluated", checkpoint_sha256=sha, source_onnx_sha256="source", seed=1,
            asset_manifest_sha256="asset", model_spec_sha256="model", settings=dict(protocol_id=protocol),
            policy_hz=50, physics_hz=hz, result=dict(passed=True, cases={n: dict(command=[0., 0., .305], episodes=4,
                checks=dict(survival=True, mechanics=True), vx_mae_m_s=.02, yaw_mae_rad_s=.01,
                height_mae_m=.001, stand_drift_max_m=.01, closure_gap_max_m=.0001, passed=True) for n in cases}))
    basic = trial(names, 200, "flat-base-paired107-fixed-v1")
    high = [f"{d}_{s}" for d in ("forward", "backward") for s in (1, 2, 3)] + ["rotate_4", "rotate_4_reverse"]
    envelope = trial(names + high, 200, "envelope")
    for name in high:
        envelope["result"]["cases"][name]["command"] = [3., 0., .305]
    deployment = trial(names, 1000, "flat-base-paired107-fixed-v1")
    sources = deepcopy([basic, envelope, deployment])
    for source in sources:
        source.pop("checkpoint_sha256")
    return checkpoint, basic, envelope, deployment, sources


def test_original_behavior_loss_stops_the_driver_before_another_training_block(tmp_path):
    checkpoint, basic, envelope, deployment, sources = artifacts(tmp_path)
    assert assess(checkpoint, basic, envelope, deployment, sources, "source")["scale_candidate"]
    deployment["result"]["cases"]["stand_305mm"]["checks"]["mechanics"] = False
    deployment["result"]["passed"] = False
    result = assess(checkpoint, basic, envelope, deployment, sources, "source")
    assert not result["no_lost_original_checks"] and not result["scale_candidate"]


def test_driver_rejects_a_stale_evaluation_from_another_checkpoint(tmp_path):
    checkpoint, basic, envelope, deployment, sources = artifacts(tmp_path)
    envelope["checkpoint_sha256"] = "other"
    with pytest.raises(ValueError, match="exact completed checkpoint"):
        assess(checkpoint, basic, envelope, deployment, sources, "source")


def test_a_worse_descendant_cannot_hide_behind_the_weaker_original_baseline(tmp_path):
    checkpoint, basic, envelope, deployment, sources = artifacts(tmp_path)
    previous = deepcopy([basic, envelope, deployment])
    sources[2]["result"]["cases"]["stand_305mm"]["checks"]["velocity"] = False
    previous[2]["result"]["cases"]["stand_305mm"]["checks"]["velocity"] = True
    deployment["result"]["cases"]["stand_305mm"]["checks"]["velocity"] = False
    envelope["result"]["cases"]["forward_3"]["vx_mae_m_s"] += .03
    result = assess(checkpoint, basic, envelope, deployment, sources, "source", previous=previous)
    assert result["no_lost_original_checks"]
    assert not result["no_previous_regressions"] and not result["scale_candidate"]
    assert any(r.get("metric") == "vx_mae_m_s" for r in result["previous_regressions"])
    assert any(r.get("lost_checks") == ["velocity"] for r in result["previous_regressions"])


def test_high_speed_regression_stops_even_when_original_and_candidate_both_miss_precision(tmp_path):
    checkpoint, basic, envelope, deployment, sources = artifacts(tmp_path)
    for artifact, error in ((sources[1], .3), (envelope, .33)):
        case = artifact["result"]["cases"]["backward_3"]
        case.update(vx_mae_m_s=error, passed=False)
        case["checks"]["velocity"] = False
        artifact["result"]["passed"] = False
    result = assess(checkpoint, basic, envelope, deployment, sources, "source")
    assert result["no_lost_original_checks"]
    assert not result["no_original_tracking_regressions"]
    assert not result["scale_candidate"]
    assert result["original_tracking_regressions"][0]["case"] == "backward_3"
