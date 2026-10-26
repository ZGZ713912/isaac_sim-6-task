"""Run one labelled native legs-only or wheels-only diagnostic, with no optimizer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(Path(__file__).resolve().parent)]
i = sys.argv.index('--retained-fallen-output-rows')
mode = sys.argv[i + 1]
del sys.argv[i:i + 2]
if mode not in ('legs', 'wheels') or '--sampled-policy' in sys.argv:
    raise ValueError('Only the predeclared deterministic output partition diagnostics')
checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
source = ROOT / 'runs/robust_policy_repair_20261024/fallen_output_root001/model_initial.pt'
import json
import torch
import export_robust_getup as export
import evaluate_gap_robustness as evaluate
from fallen_output_partition_probe001 import partition, VERSION

endpoint = json.loads((ROOT / 'reports/robust_policy_repair_20261024/fallen_round014_endpoint001.json').read_text())
assert checkpoint == (ROOT / endpoint['checkpoint']).resolve() and export.digest(checkpoint) == endpoint['checkpoint_sha256']
assert export.digest(source) == endpoint['source_sha256']
native_load, native_write = export.load_mean, evaluate.write_json
metadata = dict(schema=VERSION, retained_learned_rows=mode,
    checkpoint=str(checkpoint), checkpoint_sha256=export.digest(checkpoint),
    source=str(source), source_sha256=export.digest(source),
    runtime_weights_changed=True, checkpoint_file_changed=False, gaussian_mean_changed=True,
    sampling_std_changed=False, observer_changed=False, eligible_for_checkpoint_mean_qualification=False,
    wrapper_sha256=export.digest(Path(__file__)),
    probe_sha256=export.digest(Path(__file__).with_name('fallen_output_partition_probe001.py')), new_PPO=0)


def load(path, **kwargs):
    mean, info = native_load(path, **kwargs)
    if Path(path).resolve() == checkpoint:
        with torch.random.fork_rng(devices=[]):
            source_mean, _ = native_load(source)
        actual = partition(mean, source_mean, mode)
        native_write(output / 'runtime_function_readback.json', dict(metadata, **actual))
    return mean, info


def write(path, value):
    if Path(path).name in ('task_contract.json', 'progress.json'):
        value.setdefault('evaluation_overrides', {})['fallen_output_partition_probe'] = metadata
        value['qualification'] = None
    if Path(path).name == 'progress.json' and 'source_sha256' in value:
        for file in (Path(__file__).resolve(), Path(__file__).with_name('fallen_output_partition_probe001.py').resolve()):
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
