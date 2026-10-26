"""Longer evaluation must strengthen duration without relaxing any original gate."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/extended_scratch_standing_evaluation.py"
spec = importlib.util.spec_from_file_location("extended_stand", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def configuration():
    config = dict(episode_seconds=8., physics_dt=.001, policy_dt=.005, unchanged={"leg_kd": 2.5})
    design = dict(stages=dict(speed=dict(roles=dict(train=dict(contract=deepcopy(config))))))
    settings = dict(episode_seconds=8., unchanged=True)
    return config, design, settings


def contract():
    return module.configure(*configuration(), 30)


def cases():
    rows = []
    for height in (.23, .305, .42):
        for pitch in (0., -.25, .25, -.5, .5, -1., 1.):
            rows.append(dict(height_m=height, pitch_deg=pitch, full_horizon=True, physical_ms=30000,
                missing_physical_ms=0, tilt_peak_deg=.2, height_mae_m=.001,
                forward_speed_rms_m_s=.002, leg_speed_peak_rad_s=.5, closure_gap_peak_m=.0001,
                continuous_support_pass=True, standing_pass=True))
    return dict(case_count=21, cases=rows, continuous_support_pass=21, standing_pass=21)


def test_only_three_matching_timeout_fields_change():
    config, design, settings = configuration()
    old = deepcopy((config, design, settings))
    value = module.configure(config, design, settings, 30)
    for item in (config, design["stages"]["speed"]["roles"]["train"]["contract"], settings):
        assert item["episode_seconds"] == 30.
        item["episode_seconds"] = 8.
    assert (config, design, settings) == old
    assert value["policy_frames"] == 6000 and value["physical_frames"] == 30000
    assert value["startup_grace_s"] == 0 and value["contact_threshold_n"] == 2.


@pytest.mark.parametrize("bad", [8, 15, 30., True])
def test_unapproved_duration_is_rejected_without_mutation(bad):
    values = configuration()
    old = deepcopy(values)
    with pytest.raises(ValueError):
        module.configure(*values, bad)
    assert values == old


def test_only_the_function_horizon_literal_changes_and_source_is_preserved(tmp_path):
    source = 'def evaluate(env):\n    horizon = round(8. / env.policy_dt)\n    return horizon, .003, "same"\n'
    path = tmp_path / "extended.py"
    evaluate, metadata = module.extend_function(source, 30, {}, path)
    assert evaluate(SimpleNamespace(policy_dt=.005)) == (6000, .003, "same")
    assert metadata["original_ast_sha256"] != metadata["extended_ast_sha256"]
    assert ".003" in source and "0.003" in path.read_text()
    with pytest.raises(ValueError, match="Preserve"):
        module.extend_function(source, 30, {}, path)


@pytest.mark.parametrize("source", [
    'def wrong(env):\n    horizon = round(8. / env.policy_dt)\n',
    'def evaluate(env):\n    horizon = round(9. / env.policy_dt)\n',
    'def evaluate(env):\n    horizon = round(8. / env.dt)\n',
    'def evaluate(env):\n    horizon = round(8. / env.policy_dt)\n    horizon = 3\n',
])
def test_unrecognized_archived_evaluation_is_rejected(source, tmp_path):
    with pytest.raises(ValueError):
        module.extend_function(source, 30, {}, tmp_path / "bad.py")
    assert not list(tmp_path.iterdir())


def test_runtime_checks_actual_clock_and_timeout():
    env = SimpleNamespace(dt=.001, policy_dt=.005, max_episode_length=6000, getup_cfg={"episode_seconds": 30.})
    assert module.assert_runtime(env, contract())["policy_frames"] == 6000
    for field, bad in (("dt", .002), ("policy_dt", .02), ("max_episode_length", 1600)):
        other = deepcopy(env)
        setattr(other, field, bad)
        with pytest.raises(ValueError):
            module.assert_runtime(other, contract())


def test_complete_original_population_and_strict_acceptance():
    result = cases()
    module.assert_result(result, contract())
    # Even one missing physical millisecond invalidates continuous support.
    result["cases"][7]["missing_physical_ms"] = 1
    with pytest.raises(ValueError, match="strict"):
        module.assert_result(result, contract())
    result["cases"][7].update(continuous_support_pass=False, standing_pass=False)
    result.update(continuous_support_pass=20, standing_pass=20)
    module.assert_result(result, contract())


@pytest.mark.parametrize("change", ["8s_pass", "truncated", "population", "aggregate", "nonfinite", "relaxed_speed"])
def test_false_long_pass_cannot_be_accepted(change):
    result = cases()
    row = result["cases"][7]
    if change == "8s_pass":
        row["physical_ms"] = 8000
    elif change == "truncated":
        row["physical_ms"] = 29999
    elif change == "population":
        result["cases"].reverse()
    elif change == "aggregate":
        result["standing_pass"] = 20
    elif change == "nonfinite":
        row["leg_speed_peak_rad_s"] = float("nan")
    else:
        row["leg_speed_peak_rad_s"] = 3.1
    with pytest.raises(ValueError):
        module.assert_result(result, contract())
