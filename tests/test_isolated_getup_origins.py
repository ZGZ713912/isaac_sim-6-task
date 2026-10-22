"""Reject overlapping worlds unless complete native isolation is requested."""
import pytest

from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins


def test_legacy_layout_keeps_existing_large_population_support():
    assert not validate_isolated_origins({}, 4096)


def test_isolated_origins_require_both_explicit_ground_contracts():
    with pytest.raises(ValueError, match="square triangle-mesh"):
        validate_isolated_origins(dict(full_rl_isolated_origins=True), 128)
    with pytest.raises(ValueError, match="square triangle-mesh"):
        validate_isolated_origins(dict(full_rl_isolated_origins=True, full_rl_square_ground=True), 128)
    cfg = dict(full_rl_isolated_origins=True, full_rl_square_ground=True, flat_triangle_mesh=True)
    assert validate_isolated_origins(cfg, 128)
    with pytest.raises(ValueError, match="bounded"):
        validate_isolated_origins(cfg, 257)


def test_export_pairing_rejects_changed_isolation_even_when_origins_match():
    from export_robust_getup import verify_native_pairing
    case = dict(lanes=[dict(name="stand", pose_index=1, command=[0., 0., .305], recovery=False,
                            world_origin_m=[0., 0., 0.])])
    assert verify_native_pairing(case, case)["matched_native_origins"]
    with pytest.raises(ValueError, match="isolation"):
        verify_native_pairing(dict(case, native_origin_mode="isolated_overlap_v1"), case)


def test_static_translation_quantization_cannot_hide_a_nonzero_native_velocity():
    import numpy as np
    from analyze_robust_getup import position_velocity_consistency
    t = np.arange(201) * .005
    v = np.zeros((len(t), 3))
    v[:, 0] = .004
    true = np.zeros_like(v)
    true[:, 0] = t * .004
    expected = position_velocity_consistency(true, v, t)
    assert expected["position_velocity_displacement_error_m"] < 1e-15
    pinned = position_velocity_consistency(np.zeros_like(v), v, t)
    assert pinned["position_velocity_drift_rate_error_m_s"] == pytest.approx(.004)
