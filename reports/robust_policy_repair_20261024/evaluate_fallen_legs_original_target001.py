"""Labelled legs-only saved-mean diagnosis through the original native batch."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(Path(__file__).resolve().parent)]
import json
import shutil
import torch
import export_robust_getup as export
import evaluate_robust_getup_batch as batch
from fallen_output_partition_probe001 import partition

checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
R = ROOT / 'reports/robust_policy_repair_20261024'
endpoint = json.loads((R / 'fallen_round014_endpoint001.json').read_text())
source = ROOT / 'runs/robust_policy_repair_20261024/fallen_output_root001/model_initial.pt'
assert checkpoint == (ROOT / endpoint['checkpoint']).resolve()
assert export.digest(checkpoint) == endpoint['checkpoint_sha256']
assert export.digest(source) == endpoint['source_sha256']
metadata = dict(schema='fallen-legs-only-original-target-diagnostic-v1',
    checkpoint=str(checkpoint), checkpoint_sha256=export.digest(checkpoint),
    source=str(source), source_sha256=export.digest(source), retained_learned_rows='legs',
    gaussian_mean_changed=True, checkpoint_file_changed=False, sampling_std_changed=False,
    observer_changed=False, physics_changed=False, eligible_for_checkpoint_mean_qualification=False,
    applied_variants=['robust'], reference_variant_unchanged=True,
    source_clock=1501, learned_branch_clock=1509, new_PPO=0,
    wrapper_sha256=export.digest(__file__), probe_sha256=export.digest(R / 'fallen_output_partition_probe001.py'))
native_load = export.load_mean
labelled_class = None


def attach_artifact_label(cls):
    """Only this recorded metadata attribute changes; no physical state or input."""
    key = '_declared_mean_diagnostic_overrides'
    if 'evaluation_overrides' in cls.__dict__:
        raise ValueError('Do not replace an existing class-level metadata definition')

    def get(env):
        return env.__dict__.setdefault(key, {'fallen_legs_original_target_probe': metadata})

    def set_value(env, value):
        env.__dict__[key] = dict(value, fallen_legs_original_target_probe=metadata)

    cls.evaluation_overrides = property(get, set_value)


def label_native_env():
    global labelled_class
    from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
    attach_artifact_label(RobustGetupEnv)
    labelled_class = RobustGetupEnv


def load(path, **kwargs):
    mean, info = native_load(path, **kwargs)
    if Path(path).resolve() == checkpoint:
        with torch.random.fork_rng(devices=[]):
            baseline, _ = native_load(source)
        actual = partition(mean, baseline, 'legs')
        assert actual['final_layer_runtime_sha256'] == 'fada0ed894d2d818d4eb83a0b5ccb8e0c57f53a7050198e4dc946bad945a4ecd'
        label_native_env()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'runtime_function_readback.json').write_text(json.dumps(dict(metadata, runtime_partition=actual), indent=2) + '\n')
        for file in (Path(__file__).resolve(), R / 'fallen_output_partition_probe001.py'):
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
