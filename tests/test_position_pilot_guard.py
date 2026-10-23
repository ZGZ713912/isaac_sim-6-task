"""A wrong actual experiment contract must stop before expensive native work."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def guard_module(tmp_path):
    path = Path(__file__).resolve().parents[1]/"reports/unified_robust_20261021/iteration058_guard.py"
    spec = importlib.util.spec_from_file_location("position_guard_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORTS = tmp_path
    return module


@pytest.mark.parametrize("actual,accepted", [(.35,False),(.06,True)])
def test_actual_child_settings_checked_and_mismatch_terminated(tmp_path,actual,accepted):
    guard = guard_module(tmp_path)
    output = tmp_path/"run"
    child = tmp_path/"child.py"
    child.write_text("from pathlib import Path\nimport json,time\n"
        f"p=Path({str(output)!r});p.mkdir()\n"
        f"(p/'task_contract.json').write_text(json.dumps({{'settings':{{'quiet_exploration_fraction':{actual},'quiet_wheel_exploration_fraction':.35}}}}))\n"
        "time.sleep(1.0)\n"
        "(p/'progress.json').write_text(json.dumps({'status':'completed','new_optimizer_updates':0}))\n")
    expected = dict(quiet_exploration_fraction=.06,quiet_wheel_exploration_fraction=.35)
    if accepted:
        result = guard.checked_native("child",[sys.executable,str(child)],output,expected)
        assert result["status"] == "completed"
        receipt = json.loads((tmp_path/"iteration058_task_guard.json").read_text())
        assert receipt["actual_settings_exact"] and not receipt["model_initial_exists_at_check"]
    else:
        with pytest.raises(ValueError,match="disagrees with CPU preflight"):
            guard.checked_native("child",[sys.executable,str(child)],output,expected)
        assert not (output/"progress.json").exists()
        receipt = json.loads((tmp_path/"child_process.json").read_text())
        assert receipt["status"] == "rejected_during_startup" and receipt["returncode"] is not None
