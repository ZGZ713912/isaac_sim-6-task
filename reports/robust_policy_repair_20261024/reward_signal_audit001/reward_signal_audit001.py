"""Compare reward functions on authenticated closed-loop states; no new policy or physics."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
QUEUE=REPORT/'requested_v2_evaluation_queue002'
OUT=REPORT/'reward_signal_audit001'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assert not OUT.exists()
queue=read(QUEUE/'receipt.json')
assert queue['status']=='completed_unqualified' and queue['active_child'] is None
assert not Path('/proc/'+str(queue['pid'])).exists()
run=Path(queue['checkpoint']).parent
assert sha(queue['checkpoint'])==queue['checkpoint_sha256']
old_path=run/'source_snapshot/src/wheeled_tasks/chassis/precise_standing_reward.py'
assert sha(old_path)==read(run/'source_hashes.json')['src/wheeled_tasks/chassis/precise_standing_reward.py']
candidate_path=WORK/'scripts/standing_reward_candidates.py'
old=load('frozen_gaussian_reward',old_path)
new=load('unadopted_long_tail',candidate_path)
OUT.mkdir()
agent=read(run/'agent_config.json')
logger=Logger(log_dir=str(OUT),cfg=agent,env_cfg=read(run/'config/contract.json'),num_envs=7,
    is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cpu')
logger.init_logging_writer()
rows=[]
expected={}
for profile in ('nominal','requested'):
    folder=QUEUE/('eval_'+profile)
    result=read(folder/'evaluation/result.json')
    assert sha(folder/'evaluation/native_trace.npz')==result['trace_sha256']
    assert sha(folder/'evaluation/feedback_trace.npz')==result['feedback_trace']['sha256']
    lanes=[i for i,c in enumerate(result['cases']) if c['height_m']==.305]
    assert len(lanes)==7 and all(result['cases'][i]['full_horizon'] for i in lanes)
    with np.load(folder/'evaluation/native_trace.npz') as z, np.load(folder/'evaluation/feedback_trace.npz') as f:
        assert np.array_equal(z['valid'],f['valid']) and z['valid'][:,lanes].all()
        height=torch.tensor(z['position'][:,lanes,2].reshape(-1),dtype=torch.float64)
        gravity=torch.tensor(z['gravity'][:,lanes].reshape(-1,3),dtype=torch.float64)
        velocity=torch.tensor(z['velocity'][:,lanes].reshape(-1,3),dtype=torch.float64)
        omega=torch.tensor(f['omega'][:,lanes].reshape(-1,3),dtype=torch.float64)
    target=torch.full_like(height,.305)
    gaussian=old.scores(height,gravity,velocity,omega,target).reshape(-1,7)
    tail=new.long_tail_scores(height,gravity,velocity,omega,target).reshape(-1,7)
    speed=velocity[:,:2].square().sum(-1).sqrt()
    s=speed.square()/.03**2+(omega[:,2]/.2).square()
    old_stationary=torch.exp(-s)
    amplitude=12.*((1.-gravity[:,2])*.5).clamp(0,1)*torch.exp(
        -gravity[:,:2].square().sum(-1)/math.sin(math.radians(5))**2)*torch.exp(-((height-target)/.015).square())
    old_derivative=(amplitude*old_stationary*2*speed/.03**2).reshape(-1,7)
    tail_derivative=(amplitude/(1+s).square()*2*speed/.03**2).reshape(-1,7)
    stationary=old_stationary.reshape(-1,7)
    rows_profile=[]
    for label,start in (('whole_0_to_8s',0),('declared_3_to_8s',600)):
        metrics=dict(observed_samples=int(gaussian[start:].numel()),
            gaussian_density_mean_per_s=float(gaussian[start:].mean()),
            long_tail_density_mean_per_s=float(tail[start:].mean()),
            gaussian_stationary_fraction_below_1e_6=float((stationary[start:]<1e-6).double().mean()),
            gaussian_speed_partial_abs_mean=float(old_derivative[start:].mean()),
            long_tail_speed_partial_abs_mean=float(tail_derivative[start:].mean()))
        rows_profile.append(dict(window=label,**metrics))
        for key,value in metrics.items():
            tag=f'RewardSignalAudit/{profile}/{label}/{key}'
            logger.writer.add_scalar(tag,value,512)
            expected[tag]=value
    rows.append(dict(profile=profile,case_indices=lanes,windows=rows_profile,
        native_trace_sha256=result['trace_sha256'],feedback_trace_sha256=result['feedback_trace']['sha256']))
logger.writer.flush()
logger.writer.close()
events=EventAccumulator(str(OUT),size_guidance={'scalars':0}).Reload()
assert set(events.Tags()['scalars'])==set(expected)
for tag,value in expected.items():
    points=events.Scalars(tag)
    assert len(points)==1 and points[0].step==512 and math.isclose(points[0].value,value,rel_tol=2e-6,abs_tol=2e-8)
result=dict(schema='standing-reward-signal-replay-audit-v1',checkpoint=queue['checkpoint'],
    checkpoint_sha256=queue['checkpoint_sha256'],profiles=rows,
    gaussian_source_sha256=sha(old_path),candidate_source_sha256=sha(candidate_path),
    producer_sha256=sha(Path(__file__)),all_RSL_scalars_readback_verified=len(expected),
    new_native_steps=0,new_PPO=0,optimizer_constructed=False,candidate_adopted=False,
    scope='All seven middle-height cases, with whole0..8s and a declared3..8s window. Speed partials describe only the mathematical reward function, not a PPO gradient or a unique cause. Re-evaluating recorded states does not demonstrate improved behavior.',
    qualification=False,hardware_ready=False)
(OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
(OUT/'standing_reward_candidates.py').write_bytes(candidate_path.read_bytes())
print(json.dumps(result))
