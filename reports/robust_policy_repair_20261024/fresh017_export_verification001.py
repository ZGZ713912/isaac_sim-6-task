"""Seal real native export parity and fail-closed corruption probes; no simulator."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
NATIVE = BASE / REL / 'fresh017_full_native_nominal001'
EXPORTED = BASE / REL / 'fresh017_export_diagnostic001'
TOOL = ROOT / 'scripts/verify_fresh_export_parity.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = REL / 'fresh017_export_diagnostic001_verified.json'
    if (ROOT / target).exists() or (BASE / target).exists():
        raise ValueError('Preserve the original export verification')
    parity_path = NATIVE / 'export_parity_inputs.npz'
    receipt_path = NATIVE / 'receipt.json'
    result_path = EXPORTED / 'result.json'
    result = json.loads(result_path.read_text())
    assert result['schema'] == 'fresh-diagnostic-mean-export-v1'
    checkpoint = Path(result['checkpoint'])
    assert sha(checkpoint) == result['checkpoint_sha256']
    assert sha(parity_path) == result['native_samples_sha256']
    assert sha(receipt_path) == result['native_receipt_sha256']
    assert sha(EXPORTED / 'diagnostic_mean.onnx') == result['onnx_sha256']
    assert all(sha(ROOT / name) == value for name, value in result['source_sha256'].items())
    assert result['diagnostic_only'] and not result['policy_behavior_qualified'] and not result['hardware_ready']
    assert result['metrics']['valid_native_vectors'] == 54535 and result['metrics']['native_cases'] == 198
    assert result['metrics']['onnx_vs_native_max_abs_raw_action'] <= result['numeric_absolute_tolerance_raw_action']
    assert result['RSL_metrics_read_back'] == 8 and result['optimizer_constructed'] is False
    receipt = json.loads(receipt_path.read_text())
    with np.load(parity_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    probes = []
    with tempfile.TemporaryDirectory(prefix='fresh-export-negative-') as temporary:
        scratch = Path(temporary)
        for name, expected_text in (
                ('wrong_checkpoint_reference', 'Native source must identify this exact checkpoint'),
                ('changed_native_mean', 'Archived CPU/native or inference parity failed'),
                ('nonfinite_native_observation', 'Need finite actual raw35D/mean6D samples')):
            native = scratch / name
            native.mkdir()
            source_receipt = dict(receipt)
            source_arrays = {key: value.copy() for key, value in arrays.items()}
            if name == 'wrong_checkpoint_reference':
                source_receipt['checkpoint_sha256'] = '0' * 64
            elif name == 'changed_native_mean':
                assert source_arrays['valid'][0, 0]
                source_arrays['mean_actions'][0, 0, 0] += 1.
            else:
                source_arrays['raw_observation'][0, 0, 0] = np.nan
            (native / 'receipt.json').write_text(json.dumps(source_receipt))
            np.savez_compressed(native / 'export_parity_inputs.npz', **source_arrays)
            output = scratch / (name + '_rejected_output')
            command = [sys.executable, str(TOOL), '--checkpoint', str(checkpoint),
                       '--native-directory', str(native), '--output', str(output)]
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, 'PYTHONPATH': 'src:scripts', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                timeout=60)
            assert process.returncode != 0 and expected_text in process.stderr, (name, process.stderr)
            assert not output.exists(), name
            probes.append(dict(name=name, rejected=True, output_created=False,
                expected_reason=expected_text, returncode=process.returncode,
                rejection_log_sha256=hashlib.sha256(process.stderr.encode()).hexdigest()))
    assert sha(checkpoint) == result['checkpoint_sha256'] and sha(parity_path) == result['native_samples_sha256']
    event, = EXPORTED.glob('events.out.tfevents.*')
    value = dict(schema='fresh-native-export-numerical-verification-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), all_numerical_checks_passed=True,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        native_directory=str(NATIVE), native_parity_samples_sha256=sha(parity_path),
        diagnostic_export_directory=str(EXPORTED), export_result=result,
        artifact_sha256={str(p):sha(p) for p in (result_path, EXPORTED/'diagnostic_mean.onnx', event)},
        corruption_probes=probes, original_files_kept=True,
        new_PPO=0, new_native_steps=0, optimizer_constructed=False,
        behavior_or_hardware_qualification=False, RMCS_modified=False,
        note='Numerical export evidence for an already failed stopped policy only; no final policy admission or installation.',
        producer_sha256=sha(Path(__file__)))
    for root in (ROOT, BASE):
        (root / target).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for path in (Path(__file__), TOOL):
        destination = BASE / path.relative_to(ROOT)
        destination.write_bytes(path.read_bytes())
    print(json.dumps(dict(numerical_pass=True, native_vectors=54535, corruption_rejections=len(probes),
                         onnx_max_abs_error=result['metrics']['onnx_vs_native_max_abs_raw_action'])))


if __name__ == '__main__':
    main()
