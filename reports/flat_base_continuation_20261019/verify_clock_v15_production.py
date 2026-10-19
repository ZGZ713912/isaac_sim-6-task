"""Installed-source CPU contract checks; no native steps or optimizer updates."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
if not (HERE/'clock_v15_install001.json').exists():
    raise SystemExit('The clock source has not been installed')
queue = json.loads((RUN/'feedback1024_queue001.json').read_text())
if queue['status'] != 'complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for the preceding native queue to finish and exit')
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from wheeled_tasks.chassis.flat_base_continuation import read_plan,write_bundle
from wheeled_tasks.chassis.flat_clock_transition import validate_flat_clock_transition
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.onnx_actor_import import recover_actor
from train_chassis import preflight

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
candidate = json.loads((HERE/'clock_candidate_v15_source002.json').read_text())
old = read_plan(ROOT,'contracts/flat_base_finetune_v14.json')
new = read_plan(ROOT,'contracts/flat_base_finetune_v15.json')
parent_path = RUN/'pilot019/model_001024.pt'
state = torch.load(parent_path,map_location='cpu',weights_only=True)
transition = validate_flat_clock_transition(ROOT,old,new,state['infos']['control_math_sha256'])
config_dir = RUN/'clock_v15_production_preflight001/config'
config,seed = write_bundle(ROOT,config_dir,512,plan_file='contracts/flat_base_finetune_v15.json')
_,manifest = preflight(config_dir/'contract.json')
control = load_chassis_control(ROOT/new['control_file'])
static = design_preflight(config,control,ROOT,stage='speed',role='train',num_envs=512,seed=seed)
source = ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
mean = mean_from_checkpoint(state,source)
original,_ = recover_actor(source)
identity = mean.verify_original(original)
log = HERE/'clock_v15_production_tests001.log'
support = json.loads((HERE/'clock_candidate_v15_support001.json').read_text())
checks = dict(installed_five_sources_exact_candidate=all(sha(ROOT/n)==v for n,v in candidate['candidate_source_sha256'].items()),
    installed_plan_exact_candidate=sha(ROOT/'contracts/flat_base_finetune_v15.json')==candidate['candidate_plan_sha256'],
    static512_worker_design_passed=static['status']=='static_passed',
    exact_actual_parent_control_in_transition=transition['parent_control_sha256']==state['infos']['control_math_sha256'],
    policy_clock50_feedback1000=transition['new']==dict(physics_dt=.001,policy_dt=.02,pc_control_dt=.001,decimation=20),
    all_non_clock_plan_fields_retained=all(old[k]==new[k] for k in old if k not in
        {'schema','physics_dt','control_file','control_sha256','training_scope'}),
    same2252_feedback_parameters_trainable=sum(p.numel() for p in mean.parameters() if p.requires_grad)==2252,
    original_source_tensors_verified=identity['original_feedback']=='exact_source_tensors',
    all63_installed_regression_tests_passed=log.exists() and '63 passed' in log.read_text(),
    three_support_sources_unchanged=all(sha(ROOT/n)==v for n,v in support['unchanged_dependency_sha256'].items()),
    actor_count_and_all_consumption_preserved=state['infos']['successful_updates_total']==1024 and
        state['infos']['actor_updates_total']==864 and state['infos']['consumed_optimizer_updates_total']==1025)
receipt = dict(schema='flat-clock-v15-installed-production-cpu-v1',checks=checks,passed=all(checks.values()),
    static_preflight=static,transition=transition,parent_checkpoint_sha256=sha(parent_path),
    installed_source_sha256=candidate['candidate_source_sha256'],tests_log_sha256=sha(log) if log.exists() else None,
    production_tests_passed=63 if checks['all63_installed_regression_tests_passed'] else 0,
    native_environment_steps=0,optimizer_updates_performed=0,behavior_qualified=False)
output = HERE/'clock_v15_production_cpu001.json'
if output.exists():
    raise SystemExit('Preserve the completed production CPU receipt')
output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=receipt['passed'],checks=checks,transition=transition),indent=2))
if not receipt['passed']:
    raise SystemExit(1)
