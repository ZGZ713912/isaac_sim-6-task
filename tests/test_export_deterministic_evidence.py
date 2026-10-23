"""Diagnostic sampling must never qualify a deterministic deployment graph."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_robust_getup import assess, verify_deterministic_evidence


def test_plain_mean_and_parent_reports_remain_valid():
    for variant in ("robust", "parent"):
        verify_deterministic_evidence(dict(actor_variant=variant, evaluation_overrides={},
            ablation_variants=False, quiet_alpha_override=None, fixed_ground_friction=None))


@pytest.mark.parametrize("field,value", [
    ("sampled_policy_exposure_diagnostic", dict(eligible_for_mean_qualification=False)),
    ("sampled_policy_exposure_diagnostic", dict(eligible_for_mean_qualification=True)),
    ("sampled_policy_exposure_diagnostic", {}),
    ("evaluation_counterfactual", {}),
    ("evaluation_overrides", dict(recovery_leg_steps={"robust": .3})),
    ("evaluation_overrides", dict(neutralize_bias="gyro")),
    ("ablation_variants", True),
    ("quiet_alpha_override", [1.] * 6),
    ("quiet_wheel_output_alpha", 1.),
    ("early_roll_authority_degrees", [20., 45.]),
    ("quiet_wheel_position_release_band_m", [.01, .025]),
    ("quiet_wheel_gyro_release_band_rad_s", [.05, .2]),
    ("recovery_leg_step_override", .3),
    ("fixed_ground_friction", .5),
])
def test_even_baseline_equivalent_diagnostic_metadata_is_rejected(field, value):
    with pytest.raises(ValueError, match="Mean qualification rejects"):
        verify_deterministic_evidence({field: value})


@pytest.mark.parametrize("sampled_on_parent", [False, True])
def test_assessor_rejects_sampling_before_analysis_or_checkpoint_access(tmp_path, sampled_on_parent):
    candidate, parent = tmp_path / "candidate", tmp_path / "parent"
    candidate.mkdir()
    parent.mkdir()
    clean = dict(actor_variant="robust", evaluation_overrides={})
    sampled = dict(clean, sampled_policy_exposure_diagnostic=dict(eligible_for_mean_qualification=False))
    for directory, report in ((candidate, clean if sampled_on_parent else sampled),
                              (parent, sampled if sampled_on_parent else clean)):
        (directory / "report.json").write_text(json.dumps(report))
    checkpoint = tmp_path / "opaque-checkpoint.pt"
    checkpoint.write_bytes(b"Hash only: no torch load or simulation is needed")
    with pytest.raises(ValueError, match="Mean qualification rejects"):
        assess([candidate] * 2, [parent] * 2, checkpoint, {})
