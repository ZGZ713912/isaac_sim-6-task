"""CPU-only response probes of frozen checkpoints, never a closed-loop score."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/fresh_unified_20261027/pilot017_dense_pose_physical_speed_mixed1536'
NAME = 'fresh017_height_feedback_read001'
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_fresh_full_suite import authenticate, digest, verify_actor_chain


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import numpy as np
    import torch
    from rsl_rl.utils import resolve_class
    from rsl_rl.utils.logger import Logger
    from tensordict import TensorDict
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    output = BASE/REL/(NAME+'.json')
    event_dir = BASE/REL/(NAME+'_events')
    if output.exists() or event_dir.exists():
        raise ValueError('Preserve earlier feedback probes')
    files = [RUN/name for name in ('model_initial.pt','model_64.pt','model_128.pt','model_192.pt')]
    identities = {str(path):digest(path) for path in files}
    bundle = authenticate(files[0])
    sys.path[:0] = [str(bundle['archive']/'src'), str(bundle['archive']/'scripts')]
    from wheeled_tasks.chassis.scut_observation import build_scut35
    from wheeled_tasks.chassis.v5_control import V5Control
    source = bundle['archive']/'src/wheeled_tasks/chassis/scut_observation.py'
    if Path(sys.modules[build_scut35.__module__].__file__).resolve() != source:
        raise ValueError('Sensor ABI must come from the frozen learner source')
    manifest_file = ROOT/bundle['cfg']['asset_directory']/'manifest.json'
    if digest(manifest_file) != bundle['cfg']['asset_manifest_sha256']:
        raise ValueError('Mechanical manifest changed')
    manifest = read(manifest_file)
    nominal = torch.tensor([manifest['nominal_joint_pos'][key] for key in V5Control.ACTIVE])
    mapping = bundle['bank']['diagnostic_height_starts']['mapping']
    q = torch.tensor([[bundle['bank']['poses'][mapping[label]]['full_joint_positions'][key]
        for key in V5Control.ACTIVE] for label in ('stand_low','stand','stand_high')])
    previous = torch.atan2((q-nominal).sin(), (q-nominal).cos())/.25
    previous[:, [2,5]] = 0.
    commands = torch.tensor([[0.,0.,height] for height in (.23,.305,.42)])
    base_frame = build_scut35(torch.zeros(3,3), torch.tensor([[0.,0.,-1.]]).repeat(3,1),
        commands, q, torch.zeros_like(q), previous, nominal,
        torch.zeros(3,dtype=torch.bool), torch.zeros(3), torch.zeros(3))
    vectors = [base_frame]
    probes = [('gravity_x_angle_rad',.001), ('gyro_y_rad_s',.02),
        ('symmetric_wheel_velocity_rad_s',.1), ('leg_velocity_control0_rad_s',.1)]
    for feature, epsilon in probes:
        for sign in (-1,1):
            frame = base_frame.clone()
            if feature == 'gravity_x_angle_rad':
                frame[:,7] = math.sin(sign*epsilon)
                frame[:,9] = -math.cos(epsilon)
            elif feature == 'gyro_y_rad_s':
                frame[:,5] = .5*sign*epsilon
            elif feature == 'symmetric_wheel_velocity_rad_s':
                frame[:,20:22] = .1*sign*epsilon
            else:
                frame[:,16] = .1*sign*epsilon
            vectors.append(frame)
    inputs = torch.cat(vectors)
    observation = TensorDict({'policy':inputs}, batch_size=[len(inputs)])
    rows, values = [], []
    model_class, actor_cfg = resolve_class(bundle['agent']['actor'])
    verify_actor_chain(model_class, bundle['archive'], bundle['actor_source_chain'])
    for file in files:
        current = authenticate(file)
        if current['archive'] != bundle['archive'] or current['agent'] != bundle['agent']:
            raise ValueError('Do not mix actors or training sources in the response probe')
        actor = model_class(observation, bundle['agent']['obs_groups'], 'actor', 6,
            **deepcopy(actor_cfg)).cpu().eval()
        actor.load_state_dict(current['state']['actor_state_dict'],strict=True)
        before = {key: value.clone() for key,value in actor.state_dict().items()}
        with torch.inference_mode():
            result = actor(observation).numpy().reshape(len(vectors),3,6)
        if not np.isfinite(result).all() or any(not torch.equal(before[key],value) for key,value in actor.state_dict().items()):
            raise ValueError('Probing must retain the exact finite actor state')
        update = current['infos']['completed_updates']
        for height_index,height in enumerate((.23,.305,.42)):
            derivatives = {feature:((result[2*i+2,height_index]-result[2*i+1,height_index])/(2*epsilon)).tolist()
                for i,(feature,epsilon) in enumerate(probes)}
            if any(value != 0. for value in derivatives['leg_velocity_control0_rad_s'][:4]):
                raise ValueError('Declared separation of direct leg-speed feedback changed')
            row = dict(checkpoint=file.name, completed_fresh_updates=update, height_m=height,
                zero_motion_sensor_frame=inputs[height_index].tolist(), raw_mean=result[0,height_index].tolist(),
                finite_difference_raw_action_per_physical_input=derivatives,
                actual_height_coefficients=actor.height_leg_coefficients.detach().numpy().tolist())
            rows.append(row)
            for j,value in enumerate(row['raw_mean']):
                values.append((f'FeedbackProbe/{height:.3f}m/mean_raw_action{j}',update,value))
            for feature,derivative in derivatives.items():
                for j,value in enumerate(derivative):
                    values.append((f'FeedbackProbe/{height:.3f}m/d_raw_action{j}_d_{feature}',update,value))
    logger = Logger(log_dir=str(event_dir),cfg=bundle['agent'],env_cfg=bundle['cfg'],num_envs=1,
        is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cpu')
    logger.init_logging_writer()
    try:
        for tag,step,value in values:
            logger.writer.add_scalar(tag,value,step)
        logger.writer.flush()
    finally:
        logger.writer.close()
    events = EventAccumulator(str(event_dir),size_guidance={'scalars':0}).Reload()
    for tag,step,value in values:
        found = [r for r in events.Scalars(tag) if r.step==step]
        if len(found)!=1 or not math.isclose(found[0].value,value,rel_tol=2e-6,abs_tol=2e-8):
            raise ValueError('RSL feedback probe readback failed: '+tag)
    if any(digest(path)!=expected for path,expected in identities.items()):
        raise ValueError('A historical checkpoint changed during the CPU-only read')
    report = dict(schema='fresh017-height-sensor-response-probe-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), producer_sha256=digest(__file__),
        checkpoints_sha256=identities, source_archive=str(bundle['archive']),
        source_sha256=current['infos']['source_hashes'], bank_sha256=digest(bundle['bank_file']),
        manifest_sha256=digest(manifest_file), input_scope='Synthetic one-feature perturbations of validated three-height reset sensor frames, with measured-position previous actions; no delayed/noisy native frame replay or physical trajectory',
        policy_order=['L_hip','L_knee','R_hip','R_knee','L_wheel','R_wheel'],
        gravity_sign='Positive projected-gravity x angle; no wheel-to-vehicle control sign inferred',
        rows=rows, RSL_events=dict(directory=str(event_dir),scalar_values_read_back=len(values),
            files_sha256={p.name:digest(p) for p in event_dir.glob('events.out.tfevents.*')}),
        new_native_steps=0,new_PPO=0,optimizer_constructed=False,parameters_modified=False,
        closed_loop_stability_proved=False,causal_attribution_proved=False,hardware_ready=False)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(output,ROOT/REL/output.name)
    shutil.copyfile(Path(__file__),BASE/REL/Path(__file__).name)
    print(json.dumps(dict(rows=len(rows),RSL_values=len(values),latest=[r for r in rows if r['completed_fresh_updates']==193])))


if __name__ == '__main__':
    main()
