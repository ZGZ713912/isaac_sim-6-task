"""Labelled historical forward under the current native observer and guards."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import json
import shutil
import torch
import export_robust_getup as export
import evaluate_robust_getup_batch as batch

R = ROOT / 'reports/robust_policy_repair_20261024'
source = ROOT / 'runs/unified_robust_20261022/offline_recovery096/model_migrated.pt'
archive = source.parent / 'source'
proof_file = R / 'historical_mean_compatibility_preflight001.json'
proof = json.loads(proof_file.read_text())
assert all(proof['checks'].values())
assert export.digest(source) == proof['checkpoint_sha256']
for name, row in proof['changed_sources'].items():
    assert export.digest(ROOT / name) == row['current']
    assert export.digest(archive / name) == row['expected'] == row['archive']
checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
assert export.digest(checkpoint) == '9a3610dc396c97d5bf3a0965f2be1217ac3e225e857604554c84b991d7f9eee9'
metadata = dict(schema='historical-full-mean-current-observer-diagnostic-v1',
    baseline_checkpoint=str(checkpoint), baseline_sha256=export.digest(checkpoint),
    historical_checkpoint=str(source), historical_sha256=export.digest(source), historical_clock=569,
    historical_source_archive=str(archive), compatibility_proof_sha256=export.digest(proof_file),
    applied_variants=['robust', 'early_roll20_to_40', 'early_roll20_to_50'],
    historical_mean_function_used=True, current_observer_retained=True, current_native_physics_and_guards_retained=True,
    current_actor_target_limiter_retained=False, historical_episode_reproduction_claimed=False,
    learned_weights_or_checkpoint_files_changed=False, sampling_std_used=False,
    eligible_for_checkpoint_mean_qualification=False, new_PPO=0, wrapper_sha256=export.digest(__file__))
native_load = export.load_mean
labelled_class = None


def attach_artifact_label(cls):
    key = '_declared_historical_full_mean_overrides'
    if 'evaluation_overrides' in cls.__dict__:
        raise ValueError('Existing metadata definition must be retained')

    def get(env):
        return env.__dict__.setdefault(key, {'historical_full_mean_probe': metadata})

    def set_value(env, value):
        env.__dict__[key] = dict(value, historical_full_mean_probe=metadata)

    cls.evaluation_overrides = property(get, set_value)


def load(path, **kwargs):
    global labelled_class
    mean, info = native_load(path, **kwargs)
    if Path(path).resolve() == checkpoint:
        with torch.random.fork_rng(devices=[]):
            historical, legacy_info = native_load(source, migration_source_root=archive)
        assert legacy_info['completed_updates'] == 569
        assert type(historical).__name__ == 'RobustGetupActor'
        assert historical.recovery_knee_step == .15 and historical.recovery_previous_common_version is None
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        attach_artifact_label(RobustGetupEnv)
        labelled_class = RobustGetupEnv
        output.mkdir(parents=True, exist_ok=True)
        (output / 'runtime_function_readback.json').write_text(json.dumps(
            dict(metadata, actual_runtime_class=type(historical).__name__,
                 historical_settings=legacy_info['settings']), indent=2) + '\n')
        for file in (Path(__file__).resolve(), proof_file):
            target = output / 'source' / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
        mean = historical
    return mean, info


export.load_mean = load
try:
    batch.main()
finally:
    export.load_mean = native_load
    if labelled_class is not None:
        del labelled_class.evaluation_overrides
