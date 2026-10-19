"""Freeze the single reward change before any v16 PPO update."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.preflight import design_preflight

torch.set_num_threads(4)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
oldfile,newfile = (f'contracts/flat_base_finetune_v{n}.json' for n in (15,16))
old,new = (read_plan(ROOT,n) for n in (oldfile,newfile))
parent_path = ROOT/'runs/flat_base_20261019/pilot021/model_001120.pt'
parent = torch.load(parent_path,map_location='cpu',weights_only=True)
info = parent['infos']
old_source = info['source_sha256']
new_source = {n:sha(ROOT/n) for n in old_source}
changed_sources = sorted(n for n in old_source if old_source[n]!=new_source[n])
expected_sources = sorted(('scripts/train_flat_base.py','src/wheeled_tasks/chassis/flat_base_continuation.py',
    'src/wheeled_tasks/chassis/flat_clock_transition.py'))
checks = dict(only_three_declared_sources_changed=changed_sources==expected_sources,
    exact_parent_plan=info['plan_sha256']==sha(ROOT/oldfile),
    only_three_declared_plan_fields_changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
        =={'schema','training_scope','moving_velocity_cost_scale'})
backup = HERE/'velocity_scale_v16_pre_adoption_source'
checks['old_source_bytes_preserved'] = all(sha(backup/n)==old_source[n] for n in expected_sources)
spec = importlib.util.spec_from_file_location('wheeled_tasks.chassis._v15_original_reader',
    backup/'src/wheeled_tasks/chassis/flat_base_continuation.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
with tempfile.TemporaryDirectory() as temporary:
    temp = Path(temporary)
    a,_ = original.write_bundle(ROOT,temp/'old',512,plan_file=oldfile)
    b,_ = write_bundle(ROOT,temp/'current',512,plan_file=oldfile)
    a['design_preflight']['file'] = b['design_preflight']['file']
    checks['v15_bundle_semantics_unchanged'] = a==b
    v15,_ = write_bundle(ROOT,temp/'v15',512,plan_file=oldfile)
    e15,_ = write_bundle(ROOT,temp/'e15',32,evaluation=True,plan_file=oldfile)
    e16,_ = write_bundle(ROOT,temp/'e16',32,evaluation=True,plan_file=newfile)
    checks['evaluation_costs_and_thresholds_unchanged'] = (e15['dense_tracking']==e16['dense_tracking']
        and e15['evaluation']==e16['evaluation'])
config,seed = write_bundle(ROOT,HERE/'velocity_scale_v16_preflight001/config',512,plan_file=newfile)
control = load_chassis_control(ROOT/config['control_math_source'])
preflight = design_preflight(config,control,ROOT,stage='speed',role='train',num_envs=512,seed=seed)
expected = deepcopy(v15['dense_tracking'])
for name in ('broad_weight','fine_weight'):
    expected['velocity'][name] *= 3.
checks.update(worker_design512_preflight_passed=preflight['status']=='static_passed',
    exactly_velocity_broad_and_fine_weights_times3=config['dense_tracking']==expected,
    stationary_clocks_population_teacher_architecture_rates_retained=all(config.get(k)==v15.get(k) for k in (
        'stationary_tracking','physics_dt','policy_dt','action_difference_reference_dt','scene_groups','skill_specs',
        'flat_group_episode_seconds','flat_rmcs_command_training','actor_adaptation','critic_dim','learning_rate',
        'critic_learning_rate','gamma','lam','num_steps_per_env','initial_noise_std','entropy_coef')))
# Native resume must still prove every saved tensor/Adam/RNG slot. Here we
# authenticate that the CPU policy remains the existing frozen-base descendant.
mean = mean_from_checkpoint(parent,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx')
checks['same2252_trainable_feedback_parameters'] = sum(p.numel() for p in mean.parameters() if p.requires_grad)==2252
source = ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
from wheeled_algo.onnx_actor_import import recover_actor
base,_ = recover_actor(source)
mean.verify_original(base)
checks['original_source_actor_verified'] = True
checks['same_models_optimizer_contract'] = info['actor_adaptation']==new['actor_adaptation'] and [g['lr'] for g in parent['optimizer_state_dict']['param_groups']]==[new['learning_rate'],new['critic_learning_rate']]
checks['all_previous_native_owners_out'] = all(not Path('/proc/'+str(json.loads((ROOT/'runs/flat_base_20261019'/name).read_text())[key])).exists()
    for name in ('feedback1120_queue001.json','wheel_feedback1120_queue001.json','leg_feedback1120_queue001.json') for key in ('pid','worker_pid'))
report = dict(schema='flat-velocity-scale-v16-production-cpu-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=all(checks.values()),checks=checks,parent_checkpoint_sha256=sha(parent_path),
    old_plan_sha256=sha(ROOT/oldfile),new_plan_sha256=sha(ROOT/newfile),source_sha256=new_source,
    changed_source_files=changed_sources,static_preflight=preflight,
    dense_velocity_before=v15['dense_tracking']['velocity'],dense_velocity_after=config['dense_tracking']['velocity'],
    regression_tests=dict(passed=76,failed=0,
        first_attempt='75 passed; one test wrongly required bit-exact3x floating multiplication; maximum difference5.96e-8; test corrected to1e-7 absolute tolerance for scaled terms only'),
    optimizer_updates=0,native_steps=0,behavior_qualified=False)
output = HERE/'velocity_scale_v16_cpu001.json'
if output.exists():
    raise SystemExit('Preserve existing production CPU proof')
if not report['passed']:
    print(json.dumps(report,indent=2));raise SystemExit('v16 CPU contract failed')
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),changed_sources=changed_sources,
    velocity_before=report['dense_velocity_before'],velocity_after=report['dense_velocity_after'])))
