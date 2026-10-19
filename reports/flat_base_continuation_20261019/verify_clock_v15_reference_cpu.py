"""Paired policy-clock command math at200/1000Hz; no physics or policy fitting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TREE = HERE / 'clock_v15_source_candidate'
sys.path.insert(0, str(TREE / 'src'))
from wheeled_tasks.chassis.flat_command_reference import install_training_reference

settings = json.loads((HERE / 'clock_candidate_v15.json').read_text())['training_command_reference']
states, records = {}, {}
rng = torch.get_rng_state().clone()
for dt in (.005,.001):
    env = SimpleNamespace(num_envs=3,device='cpu',policy_dt=.02,references=None,
        cfg=dict(flat_base_continuation=True,evaluation_exact_cases=False,physics_dt=dt,command_slew=True),
        commands=torch.tensor([[0.,0.,.23],[-.03,0.,.305],[-.03,0.,.305]]),
        command_target=torch.tensor([[0.,0.],[-3.,0.],[-3.,0.]]),
        episode_length_buf=torch.zeros(3,dtype=torch.long),command_acceleration=torch.ones(3))
    env.skills = SimpleNamespace(command_base=torch.tensor([[0.,0.,.23],[-3.,0.,.305],[-3.,0.,.305]]),
        batches=[(torch.tensor([0]),dict(kind='stand',group_name='stand')),
            (torch.tensor([1,2]),dict(kind='backward',group_name='backward'))],yaw=lambda:torch.zeros(3))
    env.reset = lambda ids: env.episode_length_buf.__setitem__(ids,0)
    env.get_observations = lambda: env.commands.clone()
    states[dt] = install_training_reference(env,deepcopy(settings))
    trajectory, repeated = [], []
    for tick in range(12):
        if tick == 6:
            env.command_target[1] = torch.tensor([0.,-4.])
        if tick == 8:
            env.reset(torch.tensor([0]))
        trajectory.append(env.get_observations())
        repeated.append(torch.equal(trajectory[-1],env.get_observations()))
        env.episode_length_buf += 1
    records[dt] = dict(trajectory=torch.stack(trajectory),repeated_query_exact=all(repeated))
checks = dict(all_commands_bit_exact_across_physics_clocks=torch.equal(records[.005]['trajectory'],records[.001]['trajectory']),
    repeated_query_never_advances_policy_reference=all(r['repeated_query_exact'] for r in records.values()),
    metadata_reports_actual_feedback_clock=states[.005]['physics_hz']==200 and states[.001]['physics_hz']==1000,
    policy_clock_remains50=states[.005]['policy_hz']==states[.001]['policy_hz']==50,
    no_command_rng_consumption=torch.equal(rng,torch.get_rng_state()),
    same_direct_and_ramped_population=states[.005]['population']==states[.001]['population'],
    legacy_lane_unchanged=all(torch.equal(r['trajectory'][:,2],r['trajectory'][:1,2].expand(12,-1)) for r in records.values()),
    direct_planar_request_visible_first_tick=all(r['trajectory'][0,1,0]==-3. for r in records.values()),
    sixth_tick_reaches_low_height=all(abs(float(r['trajectory'][5,0,2])-.23)<1e-7 for r in records.values()))
report = dict(schema='flat-clock-v15-command-reference-pair-v1',checks={k:bool(v) for k,v in checks.items()},
    passed=all(checks.values()),source_sha256=hashlib.sha256((TREE/'src/wheeled_tasks/chassis/flat_command_reference.py').read_bytes()).hexdigest(),
    optimizer_updates_performed=0,native_environment_steps=0,source_adopted=False,behavior_qualified=False)
out = HERE / 'clock_candidate_v15_reference_cpu001.json'
if out.exists():
    raise SystemExit('Preserve the completed command-clock pairing review')
out.write_text(json.dumps(report,indent=2) + '\n')
print(json.dumps(report,indent=2))
if not report['passed']:
    raise SystemExit(1)
