"""Authenticated local pilot population and evidence-bound 5000-update stage."""
from copy import deepcopy
import json
from pathlib import Path

SCHEMA = 'flat-base-finetune-plan-v21'
BASELINE_SHA256 = 'e55569a9b6bdc5f60cabfda232c12ac25cb3ec589de9b15fc9fddc398dc6d576'


def make_campaign_plan(base, spec):
    if base['schema'] != 'flat-base-finetune-plan-v20':
        raise ValueError('Campaign requires the frozen transient objective')
    if spec['stage'] not in ('pilot', 'long5000') or spec['num_envs'] not in (128, 512, 1024):
        raise ValueError('Undeclared local campaign population or stage')
    plan = deepcopy(base)
    plan.update(schema=SCHEMA, allowed_num_envs=[spec['num_envs']], default_num_envs=spec['num_envs'],
        first_block_updates=64 if spec['stage'] == 'pilot' else 5000,
        additional_critic_warmup_updates=0, checkpoint_interval=8 if spec['stage'] == 'pilot' else 25,
        transient_campaign=deepcopy(spec))
    return plan


def validate_admission(root, files, checkpoint_sha):
    """Two complete four-replica seeds; replay decisions are recomputed."""
    from .flat_base_continuation import digest
    import sys
    sys.path.insert(0, str(Path(root)/'scripts'))
    from flat_transient_metrics import RETENTION, STEADY, TRANSIENT, assess_response
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError('5000 updates require two matched response assessments')
    seeds = set()
    for item in files:
        path = Path(root)/item['file']
        if digest(path) != item['sha256']:
            raise ValueError('Response assessment changed')
        analysis = json.loads(path.read_text())
        native_path = path.parent/'report.json'
        native = json.loads(native_path.read_text())
        arms = {a['role']: a for a in native['checkpoint_comparison'].values()}
        expected_conditions = {f'h{height:g}_mu{mu:g}_{name}' for height in (.305, .42)
            for mu in (.2, .5) for name in RETENTION + STEADY + TRANSIENT}
        coverage = all(set(analysis['summaries'].get(role, {})) == expected_conditions
            and all(row['requested'] == 4 for row in analysis['summaries'][role].values())
            for role in ('parent', 'trained'))
        if (analysis.get('schema') != 'flat-transient-native-response-v1'
                or analysis['native_report_sha256'] != digest(native_path)
                or native['status'] != 'evaluated' or native['replicas'] != 4
                or native['policy_hz'] != 200 or native['physics_hz'] != 1000
                or native['heights'] != [.305, .42] or native['ground_friction'] != [.2, .5]
                or set(native['case_names']) != set(RETENTION + STEADY + TRANSIENT)
                or native['horizon_seconds'] != 16. or arms['trained']['sha256'] != checkpoint_sha
                or arms['parent']['sha256'] != BASELINE_SHA256 or not coverage
                or analysis['checkpoint_comparison'] != native['checkpoint_comparison']
                or native['code_sha256']['scripts/flat_transient_metrics.py'] != digest(Path(root)/'scripts/flat_transient_metrics.py')
                or not assess_response(native, analysis['summaries'])['admitted']):
            raise ValueError('Native response evidence does not admit this exact checkpoint')
        seeds.add(native['seed'])
    if len(seeds) != 2:
        raise ValueError('Response confirmation requires distinct reset seeds')
    return dict(admitted=True, seeds=sorted(seeds), checkpoint_sha256=checkpoint_sha)


def validate_campaign_plan(root, plan):
    from .flat_base_continuation import digest, read_plan
    root = Path(root)
    spec = plan.get('transient_campaign', {})
    required = {'base_plan_file', 'base_plan_sha256', 'checkpoint', 'checkpoint_sha256',
                'parent_plan_file', 'parent_plan_sha256', 'num_envs', 'stage',
                'lineage_start_successful_updates', 'short_update_ceiling', 'admission'}
    if set(spec) != required or spec['lineage_start_successful_updates'] != 1346 or spec['short_update_ceiling'] != 512:
        raise ValueError('Local campaign budget or lineage changed')
    for name in ('base_plan', 'parent_plan'):
        if digest(root/spec[name+'_file']) != spec[name+'_sha256']:
            raise ValueError('Campaign predecessor plan changed')
    if digest(root/spec['checkpoint']) != spec['checkpoint_sha256']:
        raise ValueError('Campaign learning parent changed')
    base = read_plan(root, spec['base_plan_file'])
    if plan != make_campaign_plan(base, spec):
        raise ValueError('Campaign changed objective, physics or an undeclared field')
    if spec['stage'] == 'pilot':
        if spec['admission']:
            raise ValueError('Pilot cannot claim long-stage admission')
    else:
        validate_admission(root, spec['admission'], spec['checkpoint_sha256'])
    return spec
