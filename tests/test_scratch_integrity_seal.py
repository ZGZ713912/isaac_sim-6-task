"""Reject false seals using the real ended005 data without editing any artifact."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/seal_scratch_standing.py"
spec = importlib.util.spec_from_file_location("scratch_seal", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RUN = ROOT / "runs/scratch_standing_20261026/pilot005_static_load"
IDENTITY = ROOT / "reports/robust_policy_repair_20261024/scratch_pilot005_closed_seal001.json"


@pytest.fixture
def identity():
    if not IDENTITY.exists():
        pytest.skip("Opt-in recorded-run audit requires the local005 artifacts")
    return json.loads(IDENTITY.read_text())


def test_a_live_original_process_is_rejected_before_checkpoint_read(monkeypatch, identity):
    monkeypatch.setattr(module, "original_process_live", lambda _: True)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("A running experiment must be rejected before any checkpoint load")

    monkeypatch.setattr(module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="actually exit"):
        module.seal(RUN, identity)


@pytest.mark.parametrize("defect,failed_check", [
    ("clock", "completed_update_clocks_agree"),
    ("nonempty_initial_Adam", "initial_Adam_empty"),
    ("bad_Adam_step", "Adam_steps_match"),
    ("missing_parameter", "Adam_covers_every_parameter"),
    ("nonfinite_Adam", "checkpoint_and_Adam_finite"),
])
def test_real_ended_run_cannot_hide_broken_learning_state(monkeypatch, identity, defect, failed_check):
    native_load, native_read = module.torch.load, module.read

    def read(path):
        value = native_read(path)
        if Path(path).name == "progress.json" and defect == "clock":
            value["completed_updates"] -= 1
        return value

    def load(path, *args, **kwargs):
        value = native_load(path, *args, **kwargs)
        if Path(path).name == "model_initial.pt" and defect == "nonempty_initial_Adam":
            value["optimizer_state_dict"]["state"][0] = {"step": module.torch.tensor(1.)}
        elif Path(path).name == "model_final.pt":
            optimizer = value["optimizer_state_dict"]
            key = next(iter(optimizer["state"]))
            if defect == "bad_Adam_step":
                optimizer["state"][key]["step"] += 1
            elif defect == "nonfinite_Adam":
                optimizer["state"][key]["exp_avg"].flatten()[0] = float("nan")
            elif defect == "missing_parameter":
                optimizer["state"].pop(key)
                for group in optimizer["param_groups"]:
                    if key in group["params"]:
                        group["params"].remove(key)
        return value

    monkeypatch.setattr(module, "read", read)
    monkeypatch.setattr(module.torch, "load", load)
    result = module.seal(RUN, identity)
    assert result["checks"][failed_check] is False
    assert not result["all_checks_pass"]
    assert not result["behavior_qualification"] and not result["hardware_ready"]
