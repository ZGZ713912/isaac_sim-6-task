"""Native deterministic counterfactual; no saved checkpoint mean qualification."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(Path(__file__).resolve().parent)]
if '--sampled-policy' in sys.argv:
    raise ValueError('Deterministic diagnostic only')
checkpoint = Path(sys.argv[sys.argv.index('--checkpoint')+1]).resolve()
output = Path(sys.argv[sys.argv.index('--output')+1]).resolve()
import export_robust_getup as export
import evaluate_gap_robustness as evaluate
from motion_knee_damping_probe001 import MotionKneeDamping, VERSION

assert checkpoint == ROOT/'runs/robust_policy_repair_20261024/motion_damping080002/model_probe.pt'
assert export.digest(checkpoint) == '9a3610dc396c97d5bf3a0965f2be1217ac3e225e857604554c84b991d7f9eee9'
native_load, native_write = export.load_mean, evaluate.write_json
metadata = dict(schema=VERSION, source_checkpoint=str(checkpoint), source_sha256=export.digest(checkpoint),
    knee_increment_alpha=.6, common_rotation_retained=True,
    scope='Additional knee-only target increment damping on existing upright motion gate; scalar feasible correction preserves target error, action and paired-knee limits.',
    sensors='Existing35D/previous applied action only; no case ID, timer or simulator state',
    gaussian_mean_changed=True, checkpoint_file_changed=False, sampling_std_changed=False,
    observer_changed=False, eligible_for_checkpoint_mean_qualification=False,
    wrapper_sha256=export.digest(Path(__file__)),
    probe_sha256=export.digest(Path(__file__).with_name('motion_knee_damping_probe001.py')), new_PPO=0)


def load(path, **kwargs):
    mean, info = native_load(path, **kwargs)
    if Path(path).resolve() == checkpoint:
        mean = MotionKneeDamping(mean, alpha=.6).eval()
        native_write(output/'runtime_function_readback.json', metadata)
    return mean, info


def write(path, value):
    if Path(path).name in ('task_contract.json', 'progress.json'):
        value.setdefault('evaluation_overrides', {})['motion_knee_damping_probe'] = metadata
        value['qualification'] = None
    if Path(path).name == 'progress.json' and 'source_sha256' in value:
        for file in (Path(__file__).resolve(), Path(__file__).with_name('motion_knee_damping_probe001.py').resolve()):
            relative = file.relative_to(ROOT)
            target = output/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
            value['source_sha256'][str(relative)] = evaluate.digest(file)
    return native_write(path, value)


export.load_mean, evaluate.write_json = load, write
try:
    evaluate.main()
finally:
    export.load_mean, evaluate.write_json = native_load, native_write
