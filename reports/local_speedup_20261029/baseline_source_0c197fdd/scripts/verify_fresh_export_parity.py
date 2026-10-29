#!/usr/bin/env python3
"""Export an unqualified diagnostic mean and compare actual native input/output pairs.

This numerical tool does not admit a policy, write an RMCS profile, or establish
behavior/hardware readiness. It reconstructs only the authenticated actor from
its archived source; it never constructs PPO or an optimizer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_fresh_full_suite import authenticate, digest, verify_actor_chain, write_json


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from tensordict import TensorDict
    from rsl_rl.utils import resolve_class

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--native-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new diagnostic output directory')
    torch.set_num_threads(1)
    checkpoint_sha = digest(args.checkpoint)
    bundle = authenticate(args.checkpoint)
    if bundle['infos']['actor_class'] != 'height_initialization_candidate:HeightInitializedModel':
        raise ValueError('This diagnostic currently covers the authenticated trainable three-height actor ABI')
    receipt_path = args.native_directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('checkpoint_sha256') != checkpoint_sha or receipt.get('new_PPO') != 0
            or receipt.get('optimizer_constructed') is not False or receipt.get('preflight_only') is not False
            or receipt.get('native_policy_steps', 0) <= 0):
        raise ValueError('Native source must identify this exact checkpoint and actual zero-PPO steps')
    source = args.native_directory / 'export_parity_inputs.npz'
    source_sha = digest(source)
    sealed = receipt.get('export_parity_inputs')
    if sealed is not None and (sealed.get('sha256') != source_sha
            or sealed.get('checkpoint_sha256') != checkpoint_sha):
        raise ValueError('Native parity samples differ from their recorded identity')
    with np.load(source, allow_pickle=False) as arrays:
        raw, expected, valid, steps = (arrays[k].copy() for k in
            ('raw_observation', 'mean_actions', 'valid', 'policy_step'))
    if (raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[-1] != 35
            or expected.dtype != np.float32 or expected.shape != raw.shape[:2] + (6,)
            or valid.dtype != np.bool_ or valid.shape != raw.shape[:2]
            or steps.shape != (len(raw),) or not np.array_equal(steps, np.arange(len(raw)) * 20)
            or not np.isfinite(raw).all() or not np.isfinite(expected).all() or not valid.any()
            or not valid.any(axis=0).all() or int(steps[-1]) >= receipt['native_policy_steps']):
        raise ValueError('Need finite actual raw35D/mean6D samples from all native cases')
    if sealed is not None and (sealed['valid_vectors'] != int(valid.sum())
            or sealed['measured_policy_instants'] != len(raw) or sealed['lanes'] != raw.shape[1]):
        raise ValueError('Native parity population differs from its receipt')
    sys.path[:0] = [str(bundle['archive'] / 'src'), str(bundle['archive'] / 'scripts')]
    actor_class, actor_cfg = resolve_class(deepcopy(bundle['agent']['actor']))
    verify_actor_chain(actor_class, bundle['archive'], bundle['actor_source_chain'])
    dummy = TensorDict({'policy': torch.zeros(1, 35)}, batch_size=[1])
    actor = actor_class(dummy, bundle['agent']['obs_groups'], 'actor', 6, **actor_cfg).cpu().eval()
    actor.load_state_dict(bundle['state']['actor_state_dict'], strict=True)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    inference = actor.as_onnx().cpu().eval()
    limit = 5e-5
    cpu_error = np.zeros(6, dtype=np.float64)
    inference_error = np.zeros(6, dtype=np.float64)
    observations, references = raw[valid], expected[valid]
    with torch.inference_mode():
        for start in range(0, len(observations), 2048):
            frame = torch.from_numpy(observations[start:start + 2048])
            actual = actor(TensorDict({'policy': frame}, batch_size=[len(frame)]))
            cpu_error = np.maximum(cpu_error, np.abs(actual.numpy() - references[start:start + len(frame)]).max(0))
            inference_error = np.maximum(inference_error, np.abs(inference(frame).numpy() - actual.numpy()).max(0))
        # Negative controls establish that this input set detects omitted trained
        # height/velocity handling and a second application of normalization.
        probe = torch.from_numpy(observations[:min(2048, len(observations))])
        exact = inference(probe)
        missing_wrapper = float((actor.mlp(actor.obs_normalizer(probe)) - exact).abs().max())
        double_normalization = float((inference(actor.obs_normalizer(probe)) - exact).abs().max())
    if max(cpu_error.max(), inference_error.max()) > limit:
        raise ValueError(f'Archived CPU/native or inference parity failed: {cpu_error}, {inference_error}')
    if missing_wrapper <= limit or double_normalization <= limit:
        raise ValueError('Native samples do not detect a material export omission')
    args.output.mkdir(parents=True)
    policy = args.output / 'diagnostic_mean.onnx'
    torch.onnx.export(inference, torch.zeros(1, 35), str(policy),
        input_names=['obs'], output_names=['actions'], opset_version=17, dynamo=False, dynamic_axes=None)
    graph = onnx.load(policy)
    for key, value in dict(fresh_diagnostic_only='true', fresh_checkpoint_sha256=checkpoint_sha,
                           fresh_normalization_embedded='true', fresh_behavior_qualified='false').items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph)
    onnx.save(graph, policy)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=['CPUExecutionProvider'])
    if (len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1
            or session.get_inputs()[0].name != 'obs' or session.get_inputs()[0].shape != [1, 35]
            or session.get_inputs()[0].type != 'tensor(float)'
            or session.get_outputs()[0].name != 'actions' or session.get_outputs()[0].shape != [1, 6]
            or session.get_outputs()[0].type != 'tensor(float)'):
        raise ValueError('Export changed the float32 single-frame35 to6 ABI')
    if any(key in session.get_modelmeta().custom_metadata_map for key in ('rmcs_obs_mean', 'rmcs_obs_std')):
        raise ValueError('Embedded normalization must not also request RMCS normalization')
    ort_error = np.zeros(6, dtype=np.float64)
    worst_sample = None
    for index, (frame, reference) in enumerate(zip(observations, references, strict=True)):
        actual = session.run(['actions'], {'obs': frame[None]})[0][0]
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite ONNX output')
        delta = np.abs(actual - reference)
        if float(delta.max()) > float(ort_error.max()):
            worst_sample = index
        ort_error = np.maximum(ort_error, delta)
    if ort_error.max() > limit:
        raise ValueError(f'ONNX/native parity failed: {ort_error}')
    if any(not torch.equal(before[k], v) for k, v in actor.state_dict().items()):
        raise ValueError('Numerical export modified actor state')
    if digest(args.checkpoint) != checkpoint_sha or digest(source) != source_sha:
        raise ValueError('Original checkpoint or native sample file changed')
    metrics = dict(valid_native_vectors=int(valid.sum()), sampled_policy_instants=len(raw), native_cases=raw.shape[1],
        cpu_vs_native_max_abs_raw_action=float(cpu_error.max()),
        inference_vs_cpu_max_abs_raw_action=float(inference_error.max()),
        onnx_vs_native_max_abs_raw_action=float(ort_error.max()),
        omitted_wrapper_negative_control_max_abs_raw_action=missing_wrapper,
        double_normalization_negative_control_max_abs_raw_action=double_normalization)
    from rsl_rl.utils.logger import Logger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    logger = Logger(log_dir=str(args.output), cfg=bundle['agent'], env_cfg=bundle['cfg'], num_envs=1,
        is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device='cpu')
    logger.init_logging_writer()
    for key, value in metrics.items():
        logger.writer.add_scalar('ExportParity/' + key, value, bundle['infos']['completed_updates'])
    logger.writer.flush()
    logger.writer.close()
    events = EventAccumulator(str(args.output), size_guidance={'scalars': 0}).Reload()
    for key, value in metrics.items():
        item, = events.Scalars('ExportParity/' + key)
        if item.step != bundle['infos']['completed_updates'] or not np.isclose(item.value, value, rtol=2e-6, atol=1e-8):
            raise ValueError('RSL export parity metric did not round-trip')
    result = dict(schema='fresh-diagnostic-mean-export-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
        completed_updates=bundle['infos']['completed_updates'], diagnostic_only=True,
        actor_class=bundle['infos']['actor_class'], source_chain=bundle['actor_source_chain'],
        archived_actor_sources_verified=True, all_actor_state_unchanged=True,
        native_receipt_sha256=digest(receipt_path), native_samples_sha256=source_sha,
        native_receipt_had_parity_hash=sealed is not None,
        sample_provenance_note='Original native receipt may predate failure-safe parity hash persistence; current file hash and all recorded input/output pairs are independently verified here.',
        input=dict(name='obs', shape=[1, 35], dtype='float32'),
        output=dict(name='actions', shape=[1, 6], dtype='float32', semantics='Unclipped deterministic mean in native policy motor order'),
        policy_hz=1. / bundle['cfg']['policy_dt'], physical_hz=1. / bundle['cfg']['physics_dt'],
        normalization_embedded_once=True, learned_height_terms_embedded=True,
        raw_wheel_velocity_increment_embedded=True, leg_velocity_separation_embedded=True,
        onnx_sha256=digest(policy), onnx_vs_native_max_abs_by_output=ort_error.tolist(),
        cpu_vs_native_max_abs_by_output=cpu_error.tolist(), numeric_absolute_tolerance_raw_action=limit,
        worst_valid_flat_sample_index=worst_sample, metrics=metrics, RSL_metrics_read_back=len(metrics),
        new_PPO=0, new_native_steps=0, optimizer_constructed=False, policy_behavior_qualified=False,
        hardware_ready=False, rmcs_modified=False, policy_installed=False,
        versions=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__), ROOT/'scripts/evaluate_fresh_full_suite.py')})
    write_json(args.output / 'result.json', result)
    print(json.dumps(dict(status='diagnostic_numeric_parity_passed', **metrics)))


if __name__ == '__main__':
    main()
