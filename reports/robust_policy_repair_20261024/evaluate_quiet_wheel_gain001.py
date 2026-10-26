"""Evaluate an explicitly labelled quiet weight intervention, never qualify it."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(Path(__file__).resolve().parent)]
if '--sampled-policy' in sys.argv:
    raise ValueError('Saved-mean counterfactual diagnosis only')
checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
import export_robust_getup as export
import evaluate_gap_robustness as evaluate
from quiet_wheel_gain_probe001 import apply_gain, VERSION

assert checkpoint == ROOT / 'runs/robust_policy_repair_20261024/quiet_stance050003/model_probe.pt'
assert export.digest(checkpoint) == 'c025624014b5ff47d0967f47291d7c9aa16e2d4f89d4ba1de30d9a1c90deb5e9'
native_load, native_write = export.load_mean, evaluate.write_json
metadata = dict(schema=VERSION, source_checkpoint=str(checkpoint), source_sha256=export.digest(checkpoint),
    factor=3., changed_tensors=['quiet_velocity.' + x + '.weight' for x in ('low', 'mid', 'high')],
    changed_rows=[2, 3], changed_entries=12, gaussian_mean_changed=True,
    checkpoint_file_changed=False, sampling_std_changed=False, observer_changed=False,
    eligible_for_checkpoint_mean_qualification=False, new_PPO=0,
    rationale='Test whether stronger existing quiet wheel position feedback reduces long-lived drift under bias/delay. Preserve all leg weights, current.5 quiet authority, existing35D sensors and all physical guards.',
    wrapper_sha256=export.digest(Path(__file__)), probe_sha256=export.digest(Path(__file__).with_name('quiet_wheel_gain_probe001.py')))


def load(path, **kwargs):
    mean, info = native_load(path, **kwargs)
    if Path(path).resolve() == checkpoint:
        assert info['settings']['quiet_stance_retained_authority'] == .5
        records = apply_gain(mean, 3.)
        native_write(output / 'runtime_function_readback.json', dict(metadata, weights=records))
    return mean, info


def write(path, value):
    if Path(path).name in ('task_contract.json', 'progress.json'):
        value.setdefault('evaluation_overrides', {})['quiet_wheel_gain_probe'] = metadata
        value['qualification'] = None
    if Path(path).name == 'progress.json' and 'source_sha256' in value:
        for file in (Path(__file__).resolve(), Path(__file__).with_name('quiet_wheel_gain_probe001.py').resolve()):
            relative = file.relative_to(ROOT)
            target = output / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
            value['source_sha256'][str(relative)] = evaluate.digest(file)
    return native_write(path, value)


export.load_mean, evaluate.write_json = load, write
try:
    evaluate.main()
finally:
    export.load_mean, evaluate.write_json = native_load, native_write
