#!/usr/bin/env python3
"""Export a measured flat-speed checkpoint for explicit simulation preview."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_tasks.chassis.flat_base_continuation import digest,encode
    from export_flat_base import read_checkpoint_plan
    torch.set_num_threads(1)
    checkpoint=args.checkpoint.resolve()
    comparison=args.comparison.resolve()
    report=json.loads((comparison/'report.json').read_text())
    if (report['status']!='evaluated' or report['probe_mode']!='checkpoint_compare'
            or not report['full_envelope'] or report['replicas']<4
            or report['horizon_seconds']<32 or report.get('qualified') is not False):
        raise ValueError('Require completed full-duration, four-replica simulation comparison')
    sha=digest(checkpoint)
    matches=[i for i,a in report['checkpoint_comparison'].items()if a['sha256']==sha]
    if len(matches)!=1:
        raise ValueError('Exact checkpoint must have a unique arm in the native evidence')
    arm=matches[0]
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    info=state['infos'];plan=read_checkpoint_plan(ROOT,info)
    contract=checkpoint.parent/'config/contract.json'
    if digest(contract)!=info['contract_sha256']:
        raise ValueError('Checkpoint training contract changed')
    runtime=json.loads(contract.read_text())
    policy_hz=round(1/runtime['policy_dt']);physics_hz=round(1/runtime['physics_dt'])
    if report['policy_hz']!=policy_hz or report['physics_hz']!=physics_hz:
        raise ValueError('Preview requires native evidence at the checkpoint runtime frequency')
    source=ROOT/plan['actor_bundle']/'source.onnx'
    if digest(source)!=info['flat_base_source_onnx_sha256']:
        raise ValueError('Original actor source changed')
    actor=mean_from_checkpoint(state,source).eval()
    if report.get('reference_scope')=='straight_high_reverse_only':
        from wheeled_algo.reverse_reference_scope import ReverseReferenceScope
        parent_path=Path(report['checkpoint_comparison']['0']['path'])
        if digest(parent_path)!=report['checkpoint_comparison']['0']['sha256']:
            raise ValueError('Scoped reference base changed')
        original=mean_from_checkpoint(torch.load(parent_path,map_location='cpu',weights_only=True),source).eval()
        actor=ReverseReferenceScope(original,actor)
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    graph=actor.export_graph(source)
    metadata={p.key:p.value for p in graph.metadata_props}
    metadata.update(source_checkpoint_sha256=sha,preview_scope='simulation_unqualified',
        policy_hz=str(policy_hz),physics_hz=str(physics_hz),training_plan_sha256=info['plan_sha256'])
    onnx.helper.set_model_props(graph,metadata)
    onnx.checker.check_model(graph,full_check=True)
    model=args.output/'policy.onnx';onnx.save(graph,str(model))
    native_graph_exact=None
    if report.get('inference_backend')=='onnx_cpu_single':
        native=report['inference_models'][arm];native_path=Path(native['path'])
        if digest(native_path)!=native['sha256'] or native['batch_size']!=1:
            raise ValueError('Native single-frame inference model changed')
        native_graph_exact=onnx.load(str(native_path)).graph.SerializeToString()==graph.graph.SerializeToString()
        if not native_graph_exact:
            raise ValueError('Preview graph differs from the actual closed-loop inference graph')
    with np.load(comparison/'traces.npz')as trace:
        ids=np.flatnonzero(np.char.startswith(trace['groups'],f'g{arm}_'))
        values=trace['observations'][::10,ids,:].reshape(-1,35)
        active=trace['active'][::10,ids].reshape(-1).astype(bool)
        values=values[active]
        recorded=trace['actions'][::10,ids,:].reshape(-1,6)[active]
    if len(values)<1000:
        raise ValueError('Insufficient actual candidate observations')
    if len(values)>8192:
        sample=np.linspace(0,len(values)-1,8192,dtype=int)
        values=values[sample];recorded=recorded[sample]
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected=actor(torch.from_numpy(values.copy())).numpy()
    actual=session.run(None,{'obs':values})[0]
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
    native_single_max_error=None
    if native_graph_exact:
        single=np.concatenate([session.run(None,{'obs':v[None]})[0]for v in values])
        np.testing.assert_allclose(single,recorded,rtol=1e-6,atol=1e-6)
        native_single_max_error=float(np.abs(single-recorded).max())
    cases={n:v for n,v in report['result']['cases'].items()if n.startswith(f'g{arm}_')}
    proof=dict(schema='flat-speed-simulation-preview-v1',status='verified_export',
        model_sha256=digest(model),checkpoint=str(checkpoint),checkpoint_sha256=sha,
        contract=str(contract),contract_sha256=digest(contract),
        training_plan_file=info['plan_file'],training_plan_sha256=info['plan_sha256'],
        comparison=str(comparison/'report.json'),comparison_sha256=digest(comparison/'report.json'),
        traces_sha256=digest(comparison/'traces.npz'),comparison_arm=arm,
        verification_samples=len(values),pytorch_onnx_max_abs_error=float(np.abs(actual-expected).max()),
        native_single_frame_graph_exact=native_graph_exact,
        native_single_frame_action_max_abs_error=native_single_max_error,
        successful_updates_total=info['successful_updates_total'],actor_updates_total=info['actor_updates_total'],
        input_shape=['batch',35],output_shape=['batch',6],policy_hz=policy_hz,physics_hz=physics_hz,
        reference_scope=report.get('reference_scope','checkpoint'),
        full_episodes=sum(c['full_horizon_episodes']for c in cases.values()),
        requested_episodes=sum(c['requested_episodes']for c in cases.values()),
        cases_passing_all_checks=sum(c['passed']for c in cases.values()),total_cases=len(cases),
        behavior_qualified=False,run_scope='interactive_unscored',hardware_deployment_ready=False,
        exporter_sha256=digest(Path(__file__)))
    (args.output/'verification.json').write_bytes(encode(proof))
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
