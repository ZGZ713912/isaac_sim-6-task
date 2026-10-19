"""Static population study only; does not change or run the active trainer."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from wheeled_tasks.chassis.flat_base_continuation import read_plan,write_bundle
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.task import choose_scene_groups
from train_chassis import preflight

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
folder = HERE/'population_coverage_candidate001'
if folder.exists():
    raise SystemExit('Preserve completed static population study')
folder.mkdir()
parent_file = 'contracts/flat_base_finetune_v16.json'
parent = read_plan(ROOT,parent_file)
rows, baseline = {}, None
for count in (512,1024,2048,4096):
    candidate = deepcopy(parent)
    if count != 512:
        candidate.update(allowed_num_envs=[count],default_num_envs=count)
    changed = sorted(key for key in parent if candidate[key] != parent[key])
    assert changed == ([] if count == 512 else ['allowed_num_envs','default_num_envs'])
    path = folder/f'plan_{count}.json'
    path.write_text(json.dumps(candidate,indent=2)+'\n')
    config, seed = write_bundle(ROOT,folder/f'envs_{count}/config',count,plan_file=str(path.relative_to(ROOT)))
    _, manifest = preflight(folder/f'envs_{count}/config/contract.json')
    control = load_chassis_control(ROOT/config['control_math_source'])
    static = design_preflight(config,control,ROOT,stage='speed',role='train',num_envs=count,seed=seed)
    ignored = {'target_num_envs','ppo_minibatch_samples','flat_base_continuation','design_preflight'}
    unchanged = {key:value for key,value in config.items() if key not in ignored}
    if baseline is None: baseline = unchanged
    groups = Counter(name for name,_ in choose_scene_groups(config['scene_groups'],count))
    population = {name:dict(total=n,direct=(n if name.startswith('stand_') else (n+1)//2),
        ramped=(0 if name.startswith('stand_') else n//2)) for name,n in groups.items()}
    checks = dict(exact_only_two_population_plan_fields=changed == ([] if count == 512 else ['allowed_num_envs','default_num_envs']),
        unchanged_worker_fields_except_population_and_bundle_identity=unchanged == baseline,
        all25_original_groups_present=len(groups) == 25 and set(groups) == {g['name'] for g in parent['train_groups']},
        all_environments_accounted=sum(groups.values()) == count,
        direct_and_legacy_partition=all(row['direct']+row['ramped'] == row['total'] for row in population.values()),
        exact_policy_and_physics=config['policy_dt'] == .02 and config['physics_dt'] == .001,
        exact_rollout_and_minibatch=config['num_steps_per_env'] == 96 and config['ppo_minibatch_samples'] == count*96//4,
        CPU_design_preflight=static['status'] == 'static_passed' and static['num_envs'] == count)
    if not all(checks.values()):
        print(json.dumps(checks,indent=2));raise SystemExit('Static population derivation differs')
    rows[str(count)] = dict(checks=checks,changed_plan_fields=changed,plan_sha256=sha(path),
        worker_contract_sha256=sha(folder/f'envs_{count}/config/contract.json'),
        population=population,training_frames_per_update=count*96,minibatch_samples=count*96//4,
        static_preflight=static)
queue = read(ROOT/'runs/flat_base_20261019/feedback1248_height_coverage_queue001.json')
preserved = len(queue['source_sha256']) == 128 and all(sha(ROOT/name) == digest for name,digest in queue['source_sha256'].items())
assert preserved
report = dict(schema='flat-population-static-coverage-study-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=True,parent_plan_file=parent_file,parent_plan_sha256=sha(ROOT/parent_file),
    reader_sha256=sha(Path(__file__)),all128_live_execution_sources_unchanged=preserved,populations=rows,
    candidate_adopted=False,native_steps=0,optimizer_updates=0,
    scope='CPU population/configuration comparison only; no throughput, memory capacity or behavior qualification.',
    before_any_adoption='Need authenticated population-changing full-state resume, exact cumulative transition accounting, '
        'and measured native capacity after all current owners exit; current trainer rejects a changed num_envs.')
(folder/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:dict(checks=len(row['checks']),frames=row['training_frames_per_update'],
    edge3=row['population']['backward_edge3'],low_backward=row['population']['backward_low']) for key,row in rows.items()}))
