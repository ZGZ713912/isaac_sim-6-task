"""CPU diagnostic: separate changed actor weights from changed input statistics.

The swapped-statistic models never act in a simulator or enter training.
Only the two original checkpoints have actual trained actor/statistic pairs.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'

from copy import deepcopy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')
RUN=BASE/'runs/fresh_unified_20261027/pilot017_dense_pose_physical_speed_mixed1536'
NAME=Path(__file__).stem
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_fresh_full_suite import authenticate,digest,verify_actor_chain


def main():
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.utils import resolve_class
    from rsl_rl.utils.logger import Logger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    output=ROOT/REL/(NAME+'.json');event_dir=BASE/REL/(NAME+'_events')
    if output.exists() or event_dir.exists():raise ValueError('Retain every previous diagnostic')
    files=[RUN/'model_192.pt',RUN/'model_256.pt']
    identities={str(path):digest(path) for path in files}
    bundles=[authenticate(path) for path in files]
    bundle=bundles[0]
    assert all(b['archive']==bundle['archive'] and b['agent']==bundle['agent'] for b in bundles)
    sys.path[:0]=[str(bundle['archive']/'src'),str(bundle['archive']/'scripts')]
    actor_class,actor_cfg=resolve_class(bundle['agent']['actor'])
    verify_actor_chain(actor_class,bundle['archive'],bundle['actor_source_chain'])
    frame_file=ROOT/REL/'fresh017_height_feedback_read001.json'
    frame_report=json.loads(frame_file.read_text())
    reference=[row for row in frame_report['rows'] if row['completed_fresh_updates']==193]
    assert [row['height_m'] for row in reference]==[.23,.305,.42]
    base=torch.tensor([row['zero_motion_sensor_frame'] for row in reference],dtype=torch.float32)
    vectors=[base]
    for sign in (-1,1):
        frame=base.clone();frame[:,7]=math.sin(sign*.001);frame[:,9]=-math.cos(.001);vectors.append(frame)
    inputs=torch.cat(vectors)
    observation=TensorDict({'policy':inputs},batch_size=[len(inputs)])
    norm_keys=[key for key in bundle['state']['actor_state_dict'] if key.startswith('obs_normalizer.')]
    assert set(norm_keys)=={'obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std','obs_normalizer.count'}
    rows=[];metrics=[];statistics=[]
    for b in bundles:
        state=b['state']['actor_state_dict'];update=b['infos']['completed_updates']
        statistics.append(dict(completed_updates=update,
            mean=state['obs_normalizer._mean'].tolist(),std=state['obs_normalizer._std'].tolist(),
            count=int(state['obs_normalizer.count'])))
    for weights in bundles:
        for statistics_source in bundles:
            trained=weights['state']['actor_state_dict'];norm=statistics_source['state']['actor_state_dict']
            state={key:(norm[key] if key in norm_keys else value).clone() for key,value in trained.items()}
            assert all(torch.equal(state[k],v) for k,v in trained.items() if k not in norm_keys)
            actor=actor_class(observation,bundle['agent']['obs_groups'],'actor',6,**deepcopy(actor_cfg)).cpu().eval()
            actor.load_state_dict(state,strict=True)
            with torch.inference_mode():values=actor(observation).numpy().reshape(3,3,6)
            assert np.isfinite(values).all()
            assert all(torch.equal(state[k],v) for k,v in actor.state_dict().items())
            actor_update=weights['infos']['completed_updates'];norm_update=statistics_source['infos']['completed_updates']
            for index,height in enumerate((.23,.305,.42)):
                derivative=(values[2,index]-values[1,index])/.002
                row=dict(actor_updates=actor_update,normalizer_updates=norm_update,height_m=height,
                    actual_trained_pair=actor_update==norm_update,raw_mean=values[0,index].tolist(),
                    d_raw_action_d_gravity_x_angle_rad=derivative.tolist())
                rows.append(row)
                for label,array in (('mean_raw_action',values[0,index]),('d_raw_action_d_gravity_x_angle_rad',derivative)):
                    for channel,value in enumerate(array):
                        metrics.append((f'NormalizationAudit/norm{norm_update}/{height:.3f}m/{label}{channel}',actor_update,float(value)))
    logger=Logger(log_dir=str(event_dir),cfg=bundle['agent'],env_cfg=bundle['cfg'],num_envs=1,
        is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cpu')
    logger.init_logging_writer()
    try:
        for tag,step,value in metrics:logger.writer.add_scalar(tag,value,step)
        logger.writer.flush()
    finally:logger.writer.close()
    events=EventAccumulator(str(event_dir),size_guidance={'scalars':0}).Reload()
    for tag,step,value in metrics:
        found=[point for point in events.Scalars(tag) if point.step==step]
        assert len(found)==1 and math.isclose(found[0].value,value,rel_tol=2e-6,abs_tol=2e-8)
    assert all(digest(path)==sha for path,sha in identities.items())
    result=dict(schema='fresh017-normalization-counterfactual-CPU-audit-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),producer_sha256=digest(__file__),
        checkpoints_sha256=identities,input_frame_report_sha256=digest(frame_file),
        normalizer_statistics=statistics,rows=rows,RSL_values_read_back=len(metrics),
        RSL_events={p.name:digest(p) for p in event_dir.glob('events.out.tfevents.*')},
        new_native_steps=0,new_PPO=0,optimizer_constructed=False,checkpoint_parameters_modified=False,
        counterfactual_used_for_training=False,closed_loop_attribution_proved=False,behavior_qualified=False,
        scope='Synthetic static sensor frames. Only swap the four empirical-normalizer buffers in CPU diagnostic copies; original checkpoints retain all bytes. This quantifies input-statistic/weight effects at queried states, not a stable controller, a recovery trajectory, a teacher, or a new hardware policy.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (BASE/REL/output.name).write_bytes(output.read_bytes())
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(RSL_values_read_back=len(metrics),middle_height=[r for r in rows if r['height_m']==.305])))


if __name__=='__main__':main()
