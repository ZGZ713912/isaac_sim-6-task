#!/usr/bin/env python3
"""Explicit zero-update migration; preserve weights, optimizer, clocks and RNG."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from train_robust_getup import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT/"runs/unified_robust_20261022/composed_getup115/model_migrated.pt")
    parser.add_argument("--bank", type=Path, default=ROOT/"reports/real_robot_robustness_20261024/pose_bank_gap002.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settled-entry", action="store_true", help="Zero-update orbit-v2 handoff experiment")
    parser.add_argument("--actuator-timing",choices=("stress","native"))
    parser.add_argument("--upright-target-limit",action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.gap_robust_actor import entry_gate
    if args.settled_entry:
        from wheeled_algo.gap_entry_handoff import handoff_entry_gate as entry_gate
    from wheeled_tasks.chassis.gap_robustness import apply_profile
    torch.set_num_threads(2)
    before, _ = load_mean(args.parent)
    state = torch.load(args.parent, map_location="cpu", weights_only=False)
    legacy = json.loads((ROOT/"reports/unified_robust_20261021/pose_bank002.json").read_text())
    bank = json.loads(args.bank.read_text())
    assert bank["poses"][:len(legacy["poses"])] == legacy["poses"]
    assert bank["asset_manifest_sha256"] == legacy["asset_manifest_sha256"]
    original_info = deepcopy(state["infos"])
    args.output.mkdir(parents=True, exist_ok=False)
    settings = deepcopy(original_info["settings"])
    if args.actuator_timing is not None:
        settings["gap_actuator_timing_profile"] = args.actuator_timing
    apply_profile(settings)
    if args.settled_entry:
        settings["gap_entry_handoff_version"] = "orbit-v2"
    if args.upright_target_limit:
        if settings.get("gap_entry_handoff_version") != "orbit-v2":
            raise ValueError("Final upright limit requires orbit-v2")
        settings["gap_upright_target_limit_version"] = "error-v1"
    info = state["infos"]
    info.update(settings=settings, bank_sha256=digest(args.bank),
        migrated_from_sha256=digest(args.parent),
        gap_migration=dict(parent=str(args.parent), parent_sha256=digest(args.parent),
            optimizer_updates=0, migration_scope="sensor-only action function and reset/domain/reward contract"))
    sources = sorted(set((ROOT/"src/wheeled_algo").glob("*.py"))
        | set((ROOT/"src/wheeled_tasks/chassis").glob("*.py")))
    for relative in original_info["source_sha256"]:
        target = args.output/"source"/relative
        source = ROOT/relative
        if source.exists() and str(source) not in {str(p) for p in sources}:
            info["source_sha256"][relative] = digest(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    for path in sources:
        relative = path.relative_to(ROOT)
        info["source_sha256"][str(relative)] = digest(path)
        target = args.output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    contract = dict(schema="real-gap-migration-v1", parent_sha256=digest(args.parent),
        bank_sha256=digest(args.bank), settings=settings, optimizer_updates=0,
        full_training_admitted=False, hardware_ready=False)
    write_json(args.output/"task_contract.json", contract)
    info["task_contract_sha256"] = digest(args.output/"task_contract.json")
    checkpoint = args.output/"model_migrated.pt"
    torch.save(state, checkpoint)
    after, _ = load_mean(checkpoint)
    for key, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[key]), key
    seeds = json.loads((ROOT/"reports/real_robot_runaway_20261024/real_start_seeds.json").read_text())
    x = torch.tensor([s["observation"] for s in seeds["cases"]], dtype=torch.float32)
    with torch.inference_mode():
        old, new = before(x), after(x)
    errors = [dict(name=s["name"], entry_weight=float(entry_gate(x)[i]),
        old_target_error_rad=(old[i, :4]*.25-x[i, 10:14]).tolist(),
        new_target_error_rad=(new[i, :4]*.25-x[i, 10:14]).tolist()) for i,s in enumerate(seeds["cases"])]
    path = ROOT/"runs/unified_robust_20261022/composed_static_original115/robust/trajectory.npz"
    observations = np.load(path)["observations"].reshape(-1, 35)
    nonentry = exact = 0
    max_output_change = 0.
    max_gate = 0.
    with torch.inference_mode():
        for block in np.array_split(observations, max(1, len(observations)//4096)):
            obs = torch.as_tensor(block)
            gate = entry_gate(obs)
            max_gate = max(max_gate, float(gate.max()))
            mask = gate[:, 0] == 0.
            nonentry += int(mask.sum())
            old_output, new_output = before(obs), after(obs)
            exact += int((old_output[mask] == new_output[mask]).all(-1).sum())
            max_output_change = max(max_output_change, float((old_output-new_output).abs().max()))
    receipt = dict(schema="real-gap-migration-receipt-v1", checkpoint_sha256=digest(checkpoint),
        parent_sha256=digest(args.parent), optimizer_updates=0, completed_updates=info["completed_updates"],
        original_bank_poses_preserved=len(legacy["poses"]), new_bank_poses=len(bank["poses"]),
        model_tensors_exact=True, optimizer_rng_and_learning_clock_preserved=True,
        real_entry_probe=errors, existing_static_frames=len(observations),
        existing_static_max_entry_gate=max_gate, nonentry_frames=nonentry, nonentry_exact_frames=exact,
        existing_static_max_action_change=max_output_change,
        behavior_qualified=False, full_training_admitted=False, hardware_ready=False)
    if not args.upright_target_limit:
        assert nonentry == exact
    write_json(args.output/"migration.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
