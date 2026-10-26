"""Single labelled donor diagnosis with an untouched same-SHA reference mean."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(Path(__file__).resolve().parent)]
import json
import shutil
import torch
import export_robust_getup as export
import evaluate_robust_getup_batch as batch
from historical_recovery_donor_probe001 import apply, DONOR, DONOR_SHA

checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
reference = Path(sys.argv[sys.argv.index('--reference-checkpoint') + 1]).resolve()
output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
assert checkpoint != reference
assert export.digest(checkpoint) == export.digest(reference) == '9a3610dc396c97d5bf3a0965f2be1217ac3e225e857604554c84b991d7f9eee9'
assert export.digest(ROOT / DONOR) == DONOR_SHA
metadata = dict(schema='historical-recovery-donor-diagnostic-v1',
    checkpoint=str(checkpoint), checkpoint_sha256=export.digest(checkpoint),
    reference_checkpoint=str(reference), reference_sha256=export.digest(reference),
    donor_checkpoint=str(DONOR), donor_sha256=DONOR_SHA,
    applied_variants=['robust'], reference_variant_unchanged=True,
    current_decoder_retained=True, historical_policy_function_reproduced=False,
    eligible_for_checkpoint_mean_qualification=False, new_PPO=0,
    wrapper_sha256=export.digest(__file__))
native_load = export.load_mean
labelled_class = None


def attach_artifact_label(cls):
    key = '_declared_historical_donor_overrides'
    if 'evaluation_overrides' in cls.__dict__:
        raise ValueError('Existing metadata definition must be retained')

    def get(env):
        return env.__dict__.setdefault(key, {'historical_recovery_donor_probe': metadata})

    def set_value(env, value):
        env.__dict__[key] = dict(value, historical_recovery_donor_probe=metadata)

    cls.evaluation_overrides = property(get, set_value)


def load(path, **kwargs):
    global labelled_class
    mean, info = native_load(path, **kwargs)
    if Path(path).resolve() == checkpoint:
        runtime = apply(mean, ROOT)
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        attach_artifact_label(RobustGetupEnv)
        labelled_class = RobustGetupEnv
        output.mkdir(parents=True, exist_ok=True)
        (output / 'runtime_function_readback.json').write_text(json.dumps(
            dict(metadata, runtime_donor=runtime), indent=2) + '\n')
        for file in (Path(__file__).resolve(), Path(__file__).parent / 'historical_recovery_donor_probe001.py'):
            target = output / 'source' / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    return mean, info


export.load_mean = load
try:
    batch.main()
finally:
    export.load_mean = native_load
    if labelled_class is not None:
        del labelled_class.evaluation_overrides
