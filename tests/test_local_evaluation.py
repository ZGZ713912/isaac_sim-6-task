"""Local pilot selection must preserve native cases and reject evaluator edits."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wheeled_tasks.chassis import local_evaluation as local
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.preflight import design_preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    with TemporaryDirectory(dir=ROOT / "runs") as directory:
        file = local.write_bundle(ROOT, Path(directory) / "exam")
        yield json.loads(file.read_text())


def test_local_diagnostic_keeps_native_cases_thresholds_and_unassisted_200hz(config):
    plan = local.read_plan(ROOT)
    source = json.loads((ROOT / plan["source_design_file"]).read_text())
    cases = source["stages"]["speed"]["roles"]["train"]["contract"]["evaluation"]["cases"]
    assert config["evaluation"]["cases"] == [c for c in cases if c["name"] in local.CASES]
    assert fixed_suite_contract(config) == config
    assert all(config[module]["enabled"] is False for module in (
        "jump_assist", "nonstandard_start", "ordinary_posture_curriculum", "release_stop"))
    receipt = design_preflight(config, load_chassis_control(ROOT / config["control_math_source"]), ROOT,
        stage="speed", role="quick", seed=config["evaluation"]["seed"], num_envs=32)
    assert receipt["decimation"] == 1
    assert receipt["budget"]["training_allowed"] is False


@pytest.mark.parametrize("mutation", ["criterion", "assistance", "seed", "control"])
def test_diagnostic_rejects_changed_cases_or_control(config, mutation):
    changed = deepcopy(config)
    if mutation == "criterion": changed["evaluation"]["cases"][0]["height_mae_m_max"] = .1
    if mutation == "assistance": changed["nonstandard_start"]["enabled"] = True
    if mutation == "seed": changed["evaluation"]["seed"] += 1
    if mutation == "control": changed["policy_dt"] = .02
    with pytest.raises(ValueError, match="independent frozen"):
        local.validate_config(changed, ROOT)
