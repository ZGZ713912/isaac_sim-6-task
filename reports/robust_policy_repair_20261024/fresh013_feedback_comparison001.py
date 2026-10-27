"""Read the partial standing reference on exactly the same synthetic sensor probes."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/scratch_standing_20261026/pilot013_gradual_admission_requested001'
NAME = 'fresh013_feedback_comparison001'
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_fresh_full_suite import authenticate,digest,verify_actor_chain


def main():
    import torch
    from rsl_rl.utils import resolve_class
    from rsl_rl.utils.logger import Logger
    from tensordict import TensorDict
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    output = BASE/REL/(NAME+'.json'); event_dir = BASE/REL/(NAME+'_events')
    if output.exists() or event_dir.exists():
        raise ValueError('Preserve earlier feedback observations')
    reference_file = ROOT/REL/'fresh016_height_feedback_read001.json'
    reference = json.loads(reference_file.read_text())
    template = [r for r in reference['rows'] if r['completed_fresh_updates']==193]
    assert [r['height_m'] for r in template]==[.23,.305,.42]
    base = torch.tensor([r['zero_motion_sensor_frame'] for r in template])
    inputs = [base]
    for feature,epsilon in (('gravity_x_angle_rad',.001),('gyro_y_rad_s',.02)):
        for sign in (-1,1):
            frame = base.clone()
            if feature=='gravity_x_angle_rad':
                frame[:,7]=math.sin(sign*epsilon);frame[:,9]=-math.cos(epsilon)
            else:
                frame[:,5]=.5*sign*epsilon
            inputs.append(frame)
    observation = TensorDict({'policy':torch.cat(inputs)},batch_size=[15])
    files = [RUN/'model_192.pt',RUN/'model_final.pt']
    hashes = {str(p):digest(p) for p in files}
    bundle = authenticate(files[0])
    assert digest(bundle['bank_file'])==reference['bank_sha256']
    sys.path[:0]=[str(bundle['archive']/'src'),str(bundle['archive']/'scripts')]
    cls,cfg = resolve_class(bundle['agent']['actor'])
    verify_actor_chain(cls,bundle['archive'],bundle['actor_source_chain'])
    rows,values = [],[]
    for file in files:
        current=authenticate(file);step=current['infos']['completed_updates']
        actor=cls(observation,bundle['agent']['obs_groups'],'actor',6,**deepcopy(cfg)).cpu().eval()
        actor.load_state_dict(current['state']['actor_state_dict'],strict=True)
        before={k:v.clone() for k,v in actor.state_dict().items()}
        with torch.inference_mode(): result=actor(observation).reshape(5,3,6)
        assert torch.isfinite(result).all() and all(torch.equal(before[k],v) for k,v in actor.state_dict().items())
        for i,height in enumerate((.23,.305,.42)):
            derivatives={feature:((result[2*j+2,i]-result[2*j+1,i])/(2*epsilon)).tolist()
                for j,(feature,epsilon) in enumerate((('gravity_x_angle_rad',.001),('gyro_y_rad_s',.02)))}
            rows.append(dict(checkpoint=file.name,completed_fresh_updates=step,height_m=height,
                raw_mean=result[0,i].tolist(),finite_difference_raw_action_per_physical_input=derivatives))
            for feature,derivative in derivatives.items():
                for j,value in enumerate(derivative):
                    values.append((f'FeedbackProbe13/{height:.3f}m/d_raw_action{j}_d_{feature}',step,value))
    logger=Logger(log_dir=str(event_dir),cfg=bundle['agent'],env_cfg=bundle['cfg'],num_envs=1,
        is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cpu')
    logger.init_logging_writer()
    try:
        for tag,step,value in values:logger.writer.add_scalar(tag,value,step)
        logger.writer.flush()
    finally:logger.writer.close()
    events=EventAccumulator(str(event_dir),size_guidance={'scalars':0}).Reload()
    for tag,step,value in values:
        found=[r for r in events.Scalars(tag) if r.step==step]
        assert len(found)==1 and math.isclose(found[0].value,value,rel_tol=2e-6,abs_tol=2e-8)
    assert all(digest(path)==expected for path,expected in hashes.items())
    report=dict(schema='fresh013-vs016-synthetic-feedback-comparison-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),producer_sha256=digest(__file__),
        exact_input_reference=str(reference_file),exact_input_reference_sha256=digest(reference_file),
        checkpoints_sha256=hashes,source_hashes=bundle['infos']['source_hashes'],rows=rows,
        RSL_events=dict(directory=str(event_dir),scalar_values_read_back=len(values),
            files_sha256={p.name:digest(p) for p in event_dir.glob('events.out.tfevents.*')}),
        comparison_limit='Same synthetic three-height reset sensor frames, not the trajectories of either actor.013 trained only305mm; architectures, seed, population and episode protocol differ. A derivative difference does not identify a unique causal share or prove a control sign.',
        new_PPO=0,new_native_steps=0,optimizer_constructed=False,old_policy_used_as_teacher=False,
        closed_loop_stability_proved=False,hardware_ready=False)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(output,ROOT/REL/output.name);shutil.copyfile(Path(__file__),BASE/REL/Path(__file__).name)
    print(json.dumps(dict(rows=len(rows),RSL_values=len(values),middle=[r for r in rows if r['height_m']==.305])))


if __name__=='__main__':main()
