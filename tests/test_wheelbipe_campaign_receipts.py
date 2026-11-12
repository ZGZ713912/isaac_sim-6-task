"""Large native data receipts must not weaken the small control-file bound."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("repair_campaign",ROOT/"scripts/run_wheelbipe_repair_campaign.py")
campaign=importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_per_lane_native_data_can_exceed_control_json_limit(tmp_path):
    path=tmp_path/"native.json"
    value={"body_masses_kg":[[.1234567890123]*21 for _ in range(1024)]}
    path.write_text(json.dumps(value,indent=2))
    assert path.stat().st_size>256*1024
    assert campaign.data_json(path)==value
    with pytest.raises(ValueError,match="control file"):
        campaign.read_json(path)


def test_data_reader_rejects_symlinks_and_oversized_payloads(tmp_path):
    target=tmp_path/"data.json"
    target.write_text("{}")
    link=tmp_path/"link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        campaign.data_json(link)
    with target.open('wb') as stream:
        stream.truncate(8*1024**2+1)
    with pytest.raises(ValueError,match="8MiB"):
        campaign.data_json(target)
