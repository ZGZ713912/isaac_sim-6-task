"""Reproduce the real domain sampler's partial-reset clock leak without physics."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
OUT=REPORT/'sensor_reset_isolation_cpu001.result.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists()
from wheeled_tasks.chassis.robust_getup import RobustDomain
from wheeled_tasks.chassis.randomization_study import resolve
spec=importlib.util.spec_from_file_location('sensor_candidate',ROOT/'scripts/selective_sensor_refresh_candidate.py')
candidate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
contract=ROOT/'contracts/wheel_leg_randomization_study_v2.json'
task=resolve(json.loads(contract.read_text()),'requested',1.)['task']
task.update(nominal_fraction=0.,sensor_delay_policy_ticks=[1,1],sensor_noise_scale=0.)
native=RobustDomain(6,'cpu',task,271502,.001)
fixed=candidate.SelectiveSensorDomain(6,'cpu',task,271502,.001)
zero=torch.zeros(6,18);zero[:,5]=-1.
moving=zero.clone();moving[:,6]=.123
rows=[]
for domain in (native,fixed):
    domain.reset(torch.arange(6),zero,tick=0)
    domain.observe(zero,0)
    before=domain.observe(moving,1).clone()
    cursor=domain.sensor_cursor
    untouched_history=domain.sensor_history[:,1:].clone()
    domain.reset(torch.tensor([0]),zero[:1],tick=1)
    after=domain.observe(moving,1).clone()
    rows.append(dict(sensor_cursor_before=cursor,sensor_cursor_after=domain.sensor_cursor,
        untouched_lane_encoder_before=before[1:,6].tolist(),
        untouched_lane_encoder_after=after[1:,6].tolist(),
        untouched_lane_delta_rad=(after[1:,6]-before[1:,6]).tolist(),
        untouched_observation_unchanged=torch.equal(before[1:],after[1:]),
        untouched_history_unchanged=torch.equal(untouched_history,domain.sensor_history[:,1:])))
names=('enabled','strength','kp','kd','spring','torque_delay','sensor_delay','motor_alpha','bias','sigma')
checks=dict(legacy_same_tick_delay_collapses=all(abs(v-.123)<1e-7 for v in rows[0]['untouched_lane_delta_rad']),
    legacy_global_cursor_advances=rows[0]['sensor_cursor_after']==(rows[0]['sensor_cursor_before']+1)%3,
    candidate_keeps_untouched_sample=rows[1]['untouched_observation_unchanged'],
    candidate_keeps_untouched_history=rows[1]['untouched_history_unchanged'],
    candidate_does_not_advance_global_cursor=rows[1]['sensor_cursor_after']==rows[1]['sensor_cursor_before'],
    actual_sampled_domain_parameters_paired=all(torch.equal(getattr(native,n),getattr(fixed,n)) for n in names),
    native_domain_RNG_consumption_paired=torch.equal(native.rng.get_state(),fixed.rng.get_state()))
assert all(checks.values())
files=['src/wheeled_tasks/chassis/robust_getup.py','src/wheeled_tasks/chassis/grounded_standing_reset.py',
       'src/wheeled_tasks/chassis/standing_failure_termination.py','scripts/selective_sensor_refresh_candidate.py',
       'tests/test_selective_sensor_refresh_candidate.py']
result=dict(schema='partial-reset-sensor-clock-leak-CPU-proof-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
    producer_sha256=sha(__file__),source_sha256={name:sha(ROOT/name) for name in files},
    study=str(contract),study_sha256=sha(contract),base_profile='requested',
    explicit_CPU_isolation_overrides=dict(nominal_fraction=0.,sensor_delay_policy_ticks=[1,1],sensor_noise_scale=0.),
    resets=dict(policy_tick=1,reset_lane_ids=[0],untouched_lane_ids=[1,2,3,4,5],true_joint_sensor_delta_rad=.123),
    legacy=rows[0],candidate=rows[1],checks=checks,all_checks_pass=True,
    CPU_test_receipt=str(REPORT/'sensor_and_admission_candidates_CPU002.xml'),
    source_bug_confirmed=True,causal_training_improvement_proven=False,
    candidate_adopted=False,new_PPO=0,native_physics_steps=0,hardware_ready=False,
    scope='Exact local RobustDomain sampler only. No live source or randomization-width mutation; native behavior and fresh training are still required.')
write(OUT,result)
(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(checks=checks,legacy=rows[0],candidate=rows[1])))
