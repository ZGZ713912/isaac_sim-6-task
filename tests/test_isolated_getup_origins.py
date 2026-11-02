"""Reject overlapping worlds unless complete native isolation is requested."""
import pytest

from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins, layout_origins, GRID, OVERLAP


def test_legacy_layout_keeps_existing_large_population_support():
    assert not validate_isolated_origins({}, 4096)


def test_isolated_origins_require_both_explicit_ground_contracts():
    with pytest.raises(ValueError, match="square triangle-mesh"):
        validate_isolated_origins(dict(full_rl_isolated_origins=True), 128)
    with pytest.raises(ValueError, match="square triangle-mesh"):
        validate_isolated_origins(dict(full_rl_isolated_origins=True, full_rl_square_ground=True), 128)
    cfg = dict(full_rl_isolated_origins=True, full_rl_square_ground=True, flat_triangle_mesh=True)
    assert validate_isolated_origins(cfg, 128)
    assert validate_isolated_origins(cfg, 512)
    with pytest.raises(ValueError, match="bounded"):
        validate_isolated_origins(cfg, 513)


@pytest.mark.parametrize("count", [1, 20, 128, 256, 512])
def test_grid_keeps_origins_unique_and_precision_bounded(count):
    origins = layout_origins(count, GRID)
    assert len(set(origins)) == count
    assert max(abs(v) for origin in origins for v in origin[:2]) <= 44.
    assert all(origin[2] == 0. for origin in origins)
    assert layout_origins(count) == [(0., 0., 0.)] * count
    if count > 1:
        assert min(sum((x-y)**2 for x,y in zip(a,b))**.5
                   for i,a in enumerate(origins) for b in origins[i+1:]) == 4.


def test_grid_cannot_bypass_collision_isolation():
    with pytest.raises(ValueError, match="collision isolation"):
        validate_isolated_origins(dict(native_environment_layout=GRID), 256)
    with pytest.raises(ValueError, match="Unknown"):
        layout_origins(256, "grid_without_filters")


def test_cli_layout_reaches_native_contract_and_preserves_historical_plans(tmp_path):
    import json
    from pathlib import Path
    from train_prone_motion import configuration
    from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
    from wheeled_tasks.chassis.prone_motion import validate
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root/"contracts/wheel_leg_prone_motion_v4.json").read_text())
    for flag, mode in (("grid", GRID), ("overlap", OVERLAP)):
        plan = resolve_plan(base, environment_layout=flag)
        cfg, settings, _ = configuration(tmp_path/flag, plan)
        assert cfg["native_environment_layout"] == settings["native_origin_mode"] == mode
        assert validate(plan) == plan
        frozen = json.loads((tmp_path/flag/"config/design.json").read_text())
        assert frozen["stages"]["speed"]["roles"]["train"]["contract"]["native_environment_layout"] == mode
    base.pop("native_environment_layout")
    legacy = resolve_plan(base)
    legacy["runtime_cli"]["options"].pop("environment_layout")
    assert validate(legacy) == legacy
    cfg, settings, _ = configuration(tmp_path/"historical", legacy)
    assert "native_environment_layout" not in cfg
    assert settings["native_origin_mode"] == OVERLAP


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
