#!/usr/bin/env python3
"""Change only the declared sensor geometry, retaining tensors, Adam and RNG."""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import argparse
    import torch
    from export_robust_getup import load_mean
    from train_robust_getup import digest, write_json
    from wheeled_tasks.chassis.standing_geometry107 import extend_settings, table_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-consumed", type=int, required=True)
    parser.add_argument("--continuous-sensors", action="store_true")
    parser.add_argument("--sensor-boundaries", action="store_true")
    parser.add_argument("--entry-reference", action="store_true")
    args = parser.parse_args()
    if sum((args.continuous_sensors, args.sensor_boundaries, args.entry_reference)) > 1:
        parser.error("Migrate one observation contract revision at a time")
    if args.output.exists():
        raise ValueError("Migration never overwrites an existing checkpoint")
    load_mean(args.source)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = deepcopy(original)
    info = migrated["infos"]
    if args.entry_reference:
        from wheeled_tasks.chassis.standing_recovery_reference import extend_settings as extend_reference
        info["settings"] = extend_reference(info["settings"])
    elif args.sensor_boundaries:
        from wheeled_tasks.chassis.standing_sensor_uncertainty import extend_settings as extend_uncertainty
        info["settings"] = extend_uncertainty(info["settings"])
    elif args.continuous_sensors:
        from wheeled_tasks.chassis.standing_sensor_winding import extend_settings as extend_winding
        info["settings"] = extend_winding(info["settings"])
    else:
        info["settings"] = extend_settings(info["settings"])
    critical_source = "src/wheeled_tasks/chassis/standing_geometry107.py"
    info["source_sha256"][critical_source] = digest(ROOT / critical_source)
    if args.continuous_sensors or args.sensor_boundaries or args.entry_reference:
        for name in ("standing_sensor_winding.py", "continuous_encoders.py"):
            critical_source = "src/wheeled_tasks/chassis/" + name
            info["source_sha256"][critical_source] = digest(ROOT / critical_source)
    if args.sensor_boundaries or args.entry_reference:
        for critical_source in ("src/wheeled_tasks/chassis/standing_sensor_uncertainty.py",
                                "src/wheeled_tasks/chassis/robust_getup_env.py",
                                "scripts/train_robust_getup.py", "scripts/export_robust_getup.py",
                                "scripts/migrate_standing_geometry107.py"):
            info["source_sha256"][critical_source] = digest(ROOT / critical_source)
    if args.entry_reference:
        critical_source = "src/wheeled_tasks/chassis/standing_recovery_reference.py"
        info["source_sha256"][critical_source] = digest(ROOT / critical_source)
    info["source_sha256"][str(table_path().relative_to(ROOT))] = digest(table_path())
    migration_key = ("observation_entry_reference_migration" if args.entry_reference else
                     "observation_sensor_projection_migration" if args.sensor_boundaries else
                     "observation_winding_migration" if args.continuous_sensors else "observation_geometry_migration")
    info[migration_key] = dict(
        source=str(args.source.resolve()), source_sha256=digest(args.source),
        new_optimizer_updates=0, tensor_changes=0, sensor_only=True,
        old_knots_unchanged=True, original_observation_contract=original["infos"]["settings"]["standing_position_observation"])
    info["hardware_ready"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    load_mean(args.output)
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(v, b[k]) for k,v in a.items())
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
        return a == b
    checks = {k:equal(original[k], migrated[k]) for k in original if k != "infos"}
    checks.update({k:equal(original["infos"][k], info[k]) for k in
                   ("completed_updates", "environment_rng", "domain_rng", "torch_rng", "cuda_rng")})
    if not all(checks.values()):
        raise ValueError("Observation-only migration changed learned or random state")
    write_json(args.output.with_suffix(".json"), dict(
        schema="standing-sensor-checkpoint-migration-v1", continuous_sensors=args.continuous_sensors,
        sensor_boundaries=args.sensor_boundaries, entry_reference=args.entry_reference, checks=checks,
        source_sha256=digest(args.source), checkpoint_sha256=digest(args.output),
        geometry_sha256=digest(table_path()), experiment_consumed=args.experiment_consumed,
        completed_updates=info["completed_updates"], new_optimizer_updates=0,
        qualification=None, hardware_ready=False))
    print(args.output)


if __name__ == "__main__":
    main()
