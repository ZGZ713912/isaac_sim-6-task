"""Poor behavior and observer errors must never stop or restart a learner."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("repair_campaign_lifecycle",ROOT/"scripts/run_wheelbipe_repair_campaign.py")
campaign=importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_poor_diagnostics_at_old_review_boundaries_do_not_interrupt_training():
    snapshots=[dict(active=True,phase="running",update=update,
                    height_error_m=.15,forward_speed_m_s=-.2,gate={"passed":False})
               for update in (999,1000,1999,2000,5999,6000,11999,12000,19999)]
    terminal=dict(active=False,phase="completed",update=20000)
    session=Mock()
    session.status.side_effect=[*snapshots,terminal]
    seen=[]
    sleep=Mock()
    assert campaign.wait_for_training(session,seen.append,sleep=sleep)==terminal
    assert seen==[*snapshots,terminal]
    assert sleep.call_count==len(snapshots)
    session.command.assert_not_called()
    session.start.assert_not_called()


def test_manual_release_is_not_automatically_resumed():
    session=Mock()
    stopped=dict(active=False,phase="stopped",update=1234,requested="stop",stop_reason="manual")
    session.status.return_value=stopped
    assert campaign.wait_for_training(session,Mock(),sleep=Mock())==stopped
    session.command.assert_not_called()
    session.start.assert_not_called()


def test_campaign_observer_failure_preserves_the_running_learner(tmp_path,monkeypatch):
    pilot=tmp_path/"pilot"
    (pilot/"onnx").mkdir(parents=True)
    point=pilot/"model.pt"
    point.write_bytes(b"pilot checkpoint")
    onnx=pilot/"onnx/policy.onnx"
    onnx.write_bytes(b"pilot onnx")
    (pilot/"onnx/contract.json").write_text(json.dumps({"onnx_sha256":campaign.file_pin(onnx)["sha256"]}))
    admission=pilot/"receipt.json"
    admission.write_text(json.dumps({"latest_checkpoint":str(point),
                                    "checkpoint_sha256":campaign.file_pin(point)["sha256"]}))
    root=tmp_path/"new-session"
    session=Mock(root=root)
    def create(*args,**kwargs):
        root.mkdir()
        return session
    monkeypatch.setattr(campaign,"WheelbipeSession",SimpleNamespace(create=create))
    monkeypatch.setattr(campaign.signal,"signal",Mock())
    monkeypatch.setattr(campaign,"validate_admission",Mock())
    monkeypatch.setattr(campaign,"wait_for_training",Mock(side_effect=RuntimeError("observer failed")))
    with pytest.raises(RuntimeError,match="observer failed"):
        campaign.main(["--session",str(root),"--admission",str(admission),
                       "--num-envs","1024","--seed","119","--python","/usr/bin/python3"])
    session.start.assert_called_once_with()
    session.command.assert_not_called()
    result=json.loads((root/"campaign.json").read_text())
    assert result["learner_stop_requested_by_campaign_error"] is False
    assert result["automatic_behavior_stop_enabled"] is False
