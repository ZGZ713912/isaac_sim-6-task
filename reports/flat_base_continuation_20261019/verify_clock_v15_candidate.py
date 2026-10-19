"""CPU contract and rejection checks for the isolated1000Hz training candidate."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TREE = HERE / 'clock_v15_source_candidate'
sys.path[:0] = [str(TREE / 'src'), str(ROOT / 'scripts')]
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.flat_clock_transition import validate_flat_clock, validate_flat_clock_transition
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
from train_chassis import preflight

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
candidate_path = HERE / 'clock_candidate_v15.json'
old = read_plan(ROOT, 'contracts/flat_base_finetune_v14.json')
new = read_plan(ROOT, candidate_path)
transition = validate_flat_clock_transition(ROOT, old, new, old['control_sha256'])
output_dir = ROOT / 'runs/flat_base_20261019/clock_v15_candidate_preflight002'
config, seed = write_bundle(ROOT, output_dir / 'config', 512, plan_file=candidate_path)
_, manifest = preflight(output_dir / 'config/contract.json')
control = load_chassis_control(ROOT / config['control_math_source'])
static = design_preflight(config, control, ROOT, stage='speed', role='train', num_envs=512, seed=seed)
checks = dict(candidate_static_preflight=static['status'] == 'static_passed',
    policy50_physics1000_feedback1000=transition['new'] == dict(physics_dt=.001,policy_dt=.02,pc_control_dt=.001,decimation=20),
    all_other_plan_fields_retained=all(new[k] == old[k] for k in old if k not in
        {'schema','physics_dt','control_file','control_sha256','training_scope'}),
    no_policy_time_discount_or_rollout_change=all(old[k] == new[k] for k in ('policy_dt','gamma','lam','num_steps_per_env')),
    original200hz_plans_still_valid=all(read_plan(ROOT,f'contracts/flat_base_finetune_v{i}.json')['physics_dt'] == .005 for i in range(1,15)))


def rejects(call):
    try:
        call()
    except ValueError:
        return True
    return False


for name, changes in (
        ('bad_physics', {'physics_dt': .002}),
        ('bad_policy', {'policy_dt': .005}),
        ('old_schema_cannot_enable_1khz', {'schema': old['schema']}),
        ('unregistered_control', {'control_file': old['control_file']}),
        ('bad_control_digest', {'control_sha256': '0' * 64})):
    bad = deepcopy(new)
    bad.update(changes)
    checks['reject_' + name] = rejects(lambda: validate_flat_clock(ROOT,bad))
for name, key, value in (
        ('changed_reward','stationary_tracking',{**new['stationary_tracking'],'position_weight':2.}),
        ('changed_sampling','train_groups',new['train_groups'][:-1]),
        ('changed_gamma','gamma',.995),
        ('changed_rollout','num_steps_per_env',48),
        ('changed_rate','learning_rate',1e-4)):
    bad = deepcopy(new)
    bad[key] = value
    checks['reject_' + name] = rejects(lambda: validate_flat_clock_transition(ROOT,old,bad,old['control_sha256']))
checks['reject_wrong_parent_control'] = rejects(lambda: validate_flat_clock_transition(ROOT,old,new,'0'*64))
checks['reject_unchanged_clock_as_migration'] = rejects(lambda: validate_flat_clock_transition(ROOT,old,old,old['control_sha256']))
report = dict(schema='flat-clock-v15-candidate-cpu-review-v1', checks=checks,passed=all(checks.values()),
    static_preflight=static, timing_transition=transition,candidate_plan_sha256=sha(candidate_path),
    source_adopted=False,native_environment_steps=0,optimizer_updates_performed=0,behavior_qualified=False)
out = HERE / 'clock_candidate_v15_cpu002.json'
if out.exists():
    raise SystemExit('Preserve the existing candidate review')
out.write_text(json.dumps(report,indent=2,allow_nan=False) + '\n')
print(json.dumps(dict(passed=report['passed'],checks=checks,timing_transition=transition),indent=2))
if not report['passed']:
    raise SystemExit(1)
