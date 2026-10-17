"""Same-experiment continuation must preserve the seal, contract and paid clocks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wheeled_tasks.chassis import local_scratch as local
from test_local_scratch import invocation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parent():
    with TemporaryDirectory(dir=ROOT / "runs") as directory:
        directory = Path(directory)
        worker = local.write_bundle(ROOT, directory / "config", 512)
        config = json.loads(worker.read_text())
        run = directory / "block"
        run.mkdir()
        checkpoint = run / "model_final.pt"
        checkpoint.write_bytes(b"metadata validation fixture; not a torch checkpoint")
        checkpoint.with_suffix(".contract.json").write_bytes(worker.read_bytes())
        seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True,
            checkpoint_sha256=local.digest(checkpoint), contract_sha256=local.digest(worker),
            learning_updates=64, learning_lineage_updates=64, stage_actor_update=64,
            actor_updates_consumed=64, global_actor_update_offset=0, role="learning_state")
        checkpoint.with_suffix(".checkpoint.json").write_text(json.dumps(seal))
        (run / "completion.json").write_text(json.dumps(dict(status="completed", parent_updates=0, successful_updates=64)))
        for name in ("design", "budget"):
            relative = Path(config["design_preflight"]["file"]).parent / f"{name}.json"
            target = run / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        yield checkpoint, config


def test_finalized_local_state_and_cumulative_ceiling(parent):
    checkpoint, config = parent
    args = invocation()
    args.resume, args.updates = checkpoint, local.read_plan(ROOT)["update_ceiling"] - 64
    assert local.validate_invocation(config, args, ROOT) == 64
    args.updates += 1
    with pytest.raises(ValueError, match="cumulative"):
        local.validate_invocation(config, args, ROOT)


@pytest.mark.parametrize("field,value", [("learning_lineage_updates", 63),
    ("actor_updates_consumed", 0), ("stage_actor_update", 0), ("global_actor_update_offset", 10)])
def test_continuation_refuses_lost_consumption_or_clock(parent, field, value):
    checkpoint, config = parent
    file = checkpoint.with_suffix(".checkpoint.json")
    seal = json.loads(file.read_text()); seal[field] = value
    file.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="clocks"):
        local.validate_parent(checkpoint, config)


def test_continuation_authenticates_checkpoint_and_exact_contract(parent):
    checkpoint, config = parent
    changed = deepcopy(config); changed["seed"] = 1
    with pytest.raises(ValueError, match="exact experiment"):
        local.validate_parent(checkpoint, changed)
    checkpoint.write_bytes(b"changed after seal")
    with pytest.raises(ValueError, match="manifest"):
        local.validate_parent(checkpoint, config)


def test_new_source_root_reuses_exact_contract_paths_and_hash(parent, tmp_path):
    checkpoint, config = parent
    plan = local.read_plan(ROOT)
    for name in (local.PLAN_FILE, plan["source_worker_file"], plan["source_design_file"],
                 plan["reward_reference_file"], plan["control_file"], plan["prior_experiment_receipt_file"]):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    worker = local.restore_bundle(tmp_path, checkpoint)
    assert worker.read_bytes() == checkpoint.with_suffix(".contract.json").read_bytes()
    assert local.validate_config(config, tmp_path)["inherited_consumption"] == 576
