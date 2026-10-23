"""Check the zero-output physical velocity feature and exact inherited training state."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from wheeled_algo.rolling_velocity import TABLE_RELATIVE, TABLE_SHA256


def main():
    torch.set_num_threads(1)
    old_path, new_path = RUNS / "quiet_velocity036/model_migrated.pt", RUNS / "quiet_rolling050/model_migrated.pt"
    old, new = [torch.load(path, map_location="cpu", weights_only=False) for path in (old_path, new_path)]
    old_mean, _ = load_mean(old_path, migration_source_root=old_path.parent / "source")
    new_mean, _ = load_mean(new_path)
    assert all(equal(value, new["actor_state_dict"][name]) for name, value in old["actor_state_dict"].items())
    added = set(new["actor_state_dict"]) - set(old["actor_state_dict"])
    heads = {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    geometry = {"mlp.mean.rolling_velocity." + name for name in new_mean.rolling_velocity.state_dict()}
    assert added == heads | geometry
    assert all(new["actor_state_dict"][name].shape == (4, 1) and new["actor_state_dict"][name].count_nonzero() == 0 for name in heads)
    assert equal(old["critic_state_dict"], new["critic_state_dict"])
    assert equal(old["optimizer_state_dict"]["state"], new["optimizer_state_dict"]["state"])
    old_groups, new_groups = old["optimizer_state_dict"]["param_groups"], new["optimizer_state_dict"]["param_groups"]
    for before, after in zip(old_groups, new_groups, strict=True):
        assert equal({k:v for k,v in before.items() if k != "params"}, {k:v for k,v in after.items() if k != "params"})
        if before["name"] == "actor":
            assert after["params"][:-3] == before["params"]
            assert all(identifier not in new["optimizer_state_dict"]["state"] for identifier in after["params"][-3:])
        else:
            assert equal(before, after)
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(old["infos"][key], new["infos"][key])
    assert new["infos"]["source_sha256"][TABLE_RELATIVE] == TABLE_SHA256
    assert hashlib.sha256((new_path.parent / "source" / TABLE_RELATIVE).read_bytes()).hexdigest() == TABLE_SHA256
    checks = {}
    with torch.inference_mode():
        for name, stride in (("quiet_velocity_baseline036", 16), ("side_confirmation032", 40), ("rolling_command_feedback049", 20)):
            data = np.load(RUNS / name / "robust/trajectory.npz")
            x = torch.from_numpy(data["observations"][::stride].reshape(-1, 35).copy())
            assert torch.equal(old_mean(x), new_mean(x))
            checks[name] = dict(actual_inputs=len(x), maximum_raw_action_error=0., exact=True)
    with tempfile.TemporaryDirectory(prefix="rolling050-negative-") as folder:
        new["actor_state_dict"]["mlp.mean.rolling_velocity.nominal"][0] += .01
        bad = Path(folder) / "bad.pt"
        torch.save(new, bad)
        try:
            load_mean(bad)
        except ValueError as error:
            assert "geometry buffers changed" in str(error)
        else:
            raise AssertionError("Tampered estimator was accepted")
    output = dict(schema="rolling050-zero-migration-audit-v1", checkpoint_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
        initial_clock=441, optimizer_updates=0, robust_consumed=961,
        added_trainable_values=12, added_parameter_slots=3, added_frozen_geometry_buffers=len(geometry),
        old_actor_critic_adam_groups_rng_exact=True, geometry_archive_sha256=TABLE_SHA256,
        changed_geometry_actual_load_rejected=True, numeric=checks)
    (REPORTS / "iteration050_migration.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
