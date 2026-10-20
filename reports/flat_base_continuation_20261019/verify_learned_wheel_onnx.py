"""Verify the nonzero learned v6 head in ONNX on all recorded development cases."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
directory = HERE / 'learned_wheel1296_onnx_numeric001'
if directory.exists():
    raise SystemExit('Preserve previous diagnostic export outputs')
checkpoint = RUN / 'pilot027_from1280/model_001296.pt'
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
actor = mean_from_checkpoint(state, source)
seal_path = HERE / 'wheel_from1280_completion001.json'
seal = read(seal_path)
if not seal['passed'] or seal['checkpoint_sha256'] != sha(checkpoint):
    raise SystemExit('An authenticated completed development trial is required')
graph = actor.export_graph(source)
metadata = {p.key: p.value for p in graph.metadata_props}
metadata.update(diagnostic_only='true', behavior_qualified='false', physical_robot_tested='false',
    diagnostic_checkpoint_sha256=sha(checkpoint))
helper.set_model_props(graph, metadata)
onnx.checker.check_model(graph, full_check=True)
original = onnx.load(source)
initializers = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
checks = dict(original_initializers_retained=all(v.name in initializers and np.array_equal(
    numpy_helper.to_array(v), initializers[v.name]) for v in original.graph.initializer))
directory.mkdir()
model_path = directory / 'diagnostic_policy.onnx'
onnx.save(graph, model_path)
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
cases = {}
for cohort in ('wheel_from1280_basic001', 'wheel_from1280_default001', 'wheel_from1280_maximum001'):
    evaluation_path = RUN / cohort / 'evaluation.json'
    evaluation = read(evaluation_path)
    checks[cohort + '_sealed_identity'] = evaluation['checkpoint_sha256'] == sha(checkpoint)
    trace_path = RUN / cohort / 'traces.npz'
    with np.load(trace_path) as trace:
        for case in evaluation['result']['cases']:
            lanes = trace['groups'] == case
            obs = trace['observations'][:, lanes][trace['active'][:, lanes]]
            indexes = np.linspace(0, len(obs) - 1, min(512, len(obs)), dtype=np.int64)
            inputs = obs[indexes].copy()
            with torch.no_grad():
                tensor = torch.from_numpy(inputs)
                expected = actor(tensor).numpy()
                correction = actor.authority_actions(tensor).numpy()
                gate = actor.authority_gate(tensor).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
            error = float(np.max(np.abs(expected - actual)))
            key = cohort + '/' + case
            checks[key + '_torch_ORT_numerics'] = np.isfinite(actual).all().item() and error < 2e-5
            cases[key] = dict(sample_frames=len(inputs), full_native_active_frames=len(obs),
                torch_ORT_max_abs_error=error, new_head_max_abs_raw_action=float(abs(correction).max()),
                gate_positive_samples=int((gate > 0).sum()), gate_min=float(gate.min()), gate_max=float(gate.max()),
                evaluation_sha256=sha(evaluation_path), trace_sha256=sha(trace_path))
checks['all_fifty_two_case_views_covered'] = len(cases) == 52
checks['actual_learned_head_signal_exercised'] = max(v['new_head_max_abs_raw_action'] for v in cases.values()) > .01
checks['closed_and_active_gates_exercised'] = min(v['gate_min'] for v in cases.values()) == 0.
checks['full_gate_exercised'] = max(v['gate_max'] for v in cases.values()) == 1.
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
proof = dict(schema='flat-learned-wheel-head-ONNX-numerics-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks, cases=cases,
    original_onnx_sha256=sha(source), checkpoint_sha256=sha(checkpoint), diagnostic_onnx_sha256=sha(model_path),
    sealed_development_trial_sha256=sha(seal_path), verifier_sha256=sha(Path(__file__)),
    source_sha256={n: sha(ROOT / n) for n in (
        'src/wheeled_algo/command_residual.py', 'src/wheeled_algo/wheel_authority.py')},
    sampled_case_views=len(cases), sampled_native_inputs=sum(v['sample_frames'] for v in cases.values()),
    torch_ORT_max_abs_error=max(v['torch_ORT_max_abs_error'] for v in cases.values()),
    new_optimizer_updates=0, new_native_steps=0, heldout_seed_used=False,
    diagnostic_only=True, behavior_qualified=False, hardware_deployment_ready=False,
    scope='CPU graph parity for actual nonzero learned parameters. This diagnostic policy retains known behavior failures and is not a deployment release.')
(directory / 'verification.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: proof[k] for k in ('passed', 'sampled_case_views', 'sampled_native_inputs',
    'torch_ORT_max_abs_error', 'diagnostic_only', 'behavior_qualified')}, indent=2))
