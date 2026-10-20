#!/usr/bin/env python3
"""Preview an explicitly exported speed candidate with the existing WASD GUI."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def preview_session(policy,contract_path,contract):
    import numpy as np
    import onnxruntime as ort
    import torch
    from wheeled_tasks.chassis.flat_base_continuation import digest
    policy,contract_path=Path(policy),Path(contract_path)
    proof=json.loads((policy.parent/'verification.json').read_text())
    if (proof['schema']!='flat-speed-simulation-preview-v1' or proof['status']!='verified_export'
            or digest(policy)!=proof['model_sha256']
            or digest(Path(proof['checkpoint']))!=proof['checkpoint_sha256']
            or contract_path.resolve()!=Path(proof['contract']).resolve()
            or digest(contract_path)!=proof['contract_sha256']
            or digest(ROOT/proof['training_plan_file'])!=proof['training_plan_sha256']
            or digest(Path(proof['comparison']))!=proof['comparison_sha256']
            or proof['behavior_qualified'] is not False or proof['policy_hz'] not in (50,200)
            or proof['policy_hz']!=round(1/contract['policy_dt'])
            or proof['physics_hz']!=round(1/contract['physics_dt'])):
        raise ValueError('Speed preview model, checkpoint, plan or native evidence changed')
    state=torch.load(proof['checkpoint'],map_location='cpu',weights_only=True)
    if state['infos']['plan_sha256']!=proof['training_plan_sha256']:
        raise ValueError('Preview checkpoint belongs to a different plan')
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(policy),sess_options=options,providers=['CPUExecutionProvider'])
    inputs,outputs=session.get_inputs(),session.get_outputs()
    if (len(inputs)!=1 or inputs[0].name!='obs' or inputs[0].shape[-1]!=35
            or inputs[0].type!='tensor(float)' or len(outputs)!=1 or outputs[0].shape[-1]!=6):
        raise ValueError('Speed preview observation/action interface changed')
    probe=session.run(None,{'obs':np.zeros((1,35),dtype=np.float32)})[0]
    if probe.shape!=(1,6)or not np.isfinite(probe).all():
        raise ValueError('Speed preview inference failed')
    return session,dict(onnx_sha256=proof['model_sha256'],contract_sha256=proof['contract_sha256'],
        asset_manifest_sha256=contract['asset_manifest_sha256'],
        successful_updates_total=proof['successful_updates_total'],actor_updates_total=proof['actor_updates_total'],
        export_verification_sha256=digest(policy.parent/'verification.json'),
        run_scope='interactive_unscored',behavior_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--package',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    package=args.package.resolve()
    proof=json.loads((package/'verification.json').read_text())
    import play_flat_base as playback
    playback.MODEL=package/'policy.onnx'
    playback.CHECKPOINT=Path(proof['checkpoint'])
    playback.CONTRACT=Path(proof['contract'])
    playback.validated_flat_session=preview_session
    sys.argv=[sys.argv[0],*remaining]
    print(f"Speed candidate {proof['successful_updates_total']}: {proof['policy_hz']} Hz simulation preview; qualification=false",flush=True)
    return playback.main()


if __name__=='__main__':
    raise SystemExit(main())
