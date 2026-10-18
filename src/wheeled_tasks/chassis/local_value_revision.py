"""A bounded same-parent value-clipping arm with completed control consumption.

The control binding is constructed from actual finalized artifacts. No target
plan can be generated or accepted while that control/evaluation is incomplete.
Once written, the target plan pins every derived digest as well as this recipe.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import local_scratch as local


CONTROL_DIR = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot019')
CONTROL_EVALUATION = CONTROL_DIR.parent / 'eval019/evaluation.json'
CONSUMPTION_FILE = 'reports/local_training_20261017/value_clip_ab001_consumption.json'
CONTROL_REPORT = 'reports/local_training_20261017/pilot019.json'
BASE = {
    'schema': 'v6-local-value-revision-v1',
    'source_plan_file': 'contracts/v6_local_scratch_200hz_v8.json',
    'source_plan_sha256': '3e42fb67b805a15306188468a138912138ed9630b3481b5a5c13c1757eece920',
    'checkpoint_file': str(CONTROL_DIR.parent / 'pilot018/model_final.pt'),
    'checkpoint_sha256': '3cbd7aabf95d500d6ffa089728b28ad1a92c8e812be9d503d9c6bdf152ae2b0e',
    'source_contract_sha256': '36ede273b3bbaeb231d2440809641b9a747cbdde105142352c14d6239417d32c',
    'completion_sha256': '907d9743d91485eb517901a7079a831c2599f31c30fb16c889da06db07659d5d',
    'seal_sha256': '9708b32c471dca8a43741fe96a20d75d77349a4cc9103f3aa076f3defa58ad34',
    'learning_updates': 2945, 'prior_consumption': 1344,
    'lineage_update_ceiling': 18656, 'first_block_update_ceiling': 128,
    'training_transitions': 182648832,
    'source_use_clipped_value_loss': True, 'target_use_clipped_value_loss': False,
}


def value_clipping_setting(config):
    """Legacy absence keeps the native True default; only named v9 accepts False."""
    if 'use_clipped_value_loss' not in config:
        return True
    value = config['use_clipped_value_loss']
    marker = config.get('local_scratch')
    if (type(value) is not bool or value is not False
            or not isinstance(marker, dict) or marker.get('plan_file') != local.VALUE_PLAN_FILE):
        raise ValueError('Value clipping override requires the exact named local v9 boolean False')
    return value


def revision_binding(root):
    """Read only real completed control records, never caller-supplied clock/SHA overrides."""
    root = Path(root)
    decision_path = root / 'reports/local_training_20261017/value_clip_ab001_plan.json'
    if (local.digest(decision_path) != '1723fb55d7b5fdcf5dff427fd9d6a1e455e2a09372fbffd6c69d03228cb49d21'
            or local.digest(root / local.RETURN_PLAN_FILE) != BASE['source_plan_sha256']):
        raise ValueError('Value clipping source plan or independent AB decision changed')
    decision = json.loads(decision_path.read_text())
    if (decision['parent_checkpoint_sha256'] != BASE['checkpoint_sha256']
            or decision['parent_lineage'] != BASE['learning_updates']):
        raise ValueError('Value clipping common-parent decision changed')
    for row in decision['evidence']:
        if local.digest(root / row['file']) != row['sha256']:
            raise ValueError('Value clipping evidence changed: ' + row['file'])
    receipt_path = root / CONSUMPTION_FILE
    if not receipt_path.is_file():
        raise ValueError('Value clipping requires actual completed/evaluated control consumption')
    receipt = json.loads(receipt_path.read_text())
    completion = json.loads((CONTROL_DIR / 'completion.json').read_text())
    seal = json.loads((CONTROL_DIR / 'model_final.checkpoint.json').read_text())
    evaluation = json.loads(CONTROL_EVALUATION.read_text())
    original = json.loads((CONTROL_DIR.parent / 'eval018/evaluation.json').read_text())
    checkpoint = CONTROL_DIR / 'model_final.pt'
    checkpoint_sha = local.digest(checkpoint)
    source = json.loads((CONTROL_DIR / 'contract.json').read_text())
    count = local.validate_parent(checkpoint, source)
    if (count != 3073 or completion['status'] != 'completed'
            or completion['parent_updates'] != 2945 or completion['successful_updates'] != 128
            or completion['attempted_updates'] != 128 or completion['discarded_updates_charged'] != 0
            or completion['parent_checkpoint_sha256'] != BASE['checkpoint_sha256']
            or completion['prior_local_experiment_consumption'] != 1216
            or completion['parent_training_transitions'] != BASE['training_transitions']
            or completion['checkpoint_sha256'] != checkpoint_sha
            or seal['checkpoint_sha256'] != checkpoint_sha or seal['sealed'] is not True
            or local.digest(CONTROL_DIR / 'contract.json') != BASE['source_contract_sha256']):
        raise ValueError('Value clipping control did not complete the exact same-parent 128 updates')
    candidate = evaluation['candidates'][-1]
    if (evaluation['status'] != 'evaluated' or evaluation['settings'] != original['settings']
            or evaluation['episodes_per_case'] != 4 or evaluation['evaluated_case_names'] != original['evaluated_case_names']
            or len(evaluation['candidates']) != 3 or len(evaluation['evaluated_case_names']) != 8
            or evaluation['candidates'][:2] != [original['candidates'][0], original['candidates'][-1]]
            or candidate['checkpoint'] != str(checkpoint) or candidate['checkpoint_sha256'] != checkpoint_sha
            or candidate['checkpoint_updates'] != 3073 or candidate['passed'] is not False):
        raise ValueError('Value clipping requires the original evaluated/rejected control and exact old candidates')
    live_stat = Path('/proc/1039116/stat')
    if live_stat.exists() and int(live_stat.read_text().rsplit(')', 1)[1].split()[19]) == 8023354:
        raise ValueError('Value clipping cannot fork while the control learner is alive')
    control = dict(checkpoint_file=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        completion_file=str(CONTROL_DIR / 'completion.json'),
        completion_sha256=local.digest(CONTROL_DIR / 'completion.json'),
        seal_file=str(CONTROL_DIR / 'model_final.checkpoint.json'),
        seal_sha256=local.digest(CONTROL_DIR / 'model_final.checkpoint.json'),
        evaluation_file=str(CONTROL_EVALUATION), evaluation_sha256=local.digest(CONTROL_EVALUATION),
        learning_updates=3073, additional_consumption=128)
    old_receipt = 'reports/local_training_20261017/entropy_ab001_consumption.json'
    expected = dict(schema='v6-local-branch-consumption-v1', status='behavior_rejected',
        consumption=1344, common_parent=dict(checkpoint=BASE['checkpoint_file'],
            sha256=BASE['checkpoint_sha256'], learning_updates=2945),
        components=[dict(file=old_receipt, sha256=local.digest(root / old_receipt), consumption=1216),
            dict(file=CONTROL_REPORT, sha256=local.digest(root / CONTROL_REPORT),
                phase_consumption=128, **control)],
        total_all_experiments_before_treatment=4289, lineage_update_ceiling=18656,
        total_update_ceiling=20000, transition_ceiling=3271557120,
        expected_total_after_treatment128=4417, expected_remaining_after_treatment128=15583,
        no_refund=True)
    # Timestamp and explanatory prose have no accounting authority.
    actual = {k: v for k, v in receipt.items() if k not in ('recorded_at', 'note')}
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('Value clipping control consumption would refund or duplicate learning history')
    result = deepcopy(BASE)
    result.update(control=control,
        evidence=[*deepcopy(decision['evidence']),
            dict(file='reports/local_training_20261017/value_clip_ab001_plan.json', sha256=local.digest(decision_path)),
            dict(file=CONTROL_REPORT, sha256=local.digest(root / CONTROL_REPORT)),
            dict(file=CONSUMPTION_FILE, sha256=local.digest(receipt_path))])
    return result


def expected_plan(root):
    revision = revision_binding(root)
    plan = deepcopy(local.read_plan(root, local.RETURN_PLAN_FILE))
    plan.update(initialization='full_state_same2945_value_clip_ab_treatment',
        use_clipped_value_loss=False, local_value_revision=revision,
        prior_experiment_consumption=1344, prior_experiment_receipt_file=CONSUMPTION_FILE,
        prior_experiment_receipt_sha256=revision['evidence'][-1]['sha256'], update_ceiling=18656)
    return plan


def validate_plan(plan, root):
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected_plan(root), sort_keys=True, allow_nan=False):
        raise ValueError('Local value revision differs from its exact same-parent derivation')


def validate_contracts(source, target, root):
    if (source.get('local_scratch', {}).get('plan_file') != local.RETURN_PLAN_FILE
            or target.get('local_scratch', {}).get('plan_file') != local.VALUE_PLAN_FILE):
        raise ValueError('Local value revision requires exact local v8 -> v9 contracts')
    budget = local.validate_config(target, root)
    _, original = local.build_bundle(root, Path(source['design_preflight']['file']).parent,
        source['target_num_envs'], plan_file=local.RETURN_PLAN_FILE)
    if original['worker'] != source:
        raise ValueError('Local value source is not its independent v8 derivation')
    allowed = {'use_clipped_value_loss', 'local_scratch', 'design_preflight',
        'budget_spec_file', 'budget_spec_sha256'}
    if {k: v for k, v in source.items() if k not in allowed} != {
            k: v for k, v in target.items() if k not in allowed}:
        raise ValueError('Local value revision changes an undeclared contract field')
    if (value_clipping_setting(source) is not True or value_clipping_setting(target) is not False
            or source['target_num_envs'] != 512 or target['num_steps_per_env'] != 384
            or target['lam'] != 1. or target['gamma'] != .9974905699336811
            or target['entropy_coef'] != .0005 or target['ppo_minibatch_samples'] != 49152
            or budget['inherited_consumption'] != 1344 or budget['update_ceiling'] != 18656
            or budget['transition_ceiling'] != 3271557120):
        raise ValueError('Local value revision changed native timing, clipping or consumed budget')
    return dict(schema=BASE['schema'], changed_contract_fields=sorted(allowed),
        source_use_clipped_value_loss=True, target_use_clipped_value_loss=False,
        consumption='all_old_lineages_entropy_control_and_value_control_preserved_no_refund',
        additional_critic_warmup=0)


def validate_parent(checkpoint, target, root):
    from .full_curriculum import checkpoint_contract_path
    checkpoint = Path(checkpoint).resolve(strict=True)
    source_path = checkpoint_contract_path(checkpoint)
    source = json.loads(source_path.read_text())
    result = validate_contracts(source, target, root)
    count = local.validate_parent(checkpoint, source)
    if (checkpoint != Path(BASE['checkpoint_file']) or count != 2945
            or local.digest(checkpoint) != BASE['checkpoint_sha256']
            or local.digest(source_path) != BASE['source_contract_sha256']
            or local.digest(checkpoint.parent / 'completion.json') != BASE['completion_sha256']
            or local.digest(checkpoint.with_suffix('.checkpoint.json')) != BASE['seal_sha256']):
        raise ValueError('Local value arm requires the exact finalized pilot018-2945, never the control endpoint')
    result.update(parent_updates=count, inherited_consumption=1344,
        total_prior_consumption=4289, lineage_update_ceiling=18656,
        parent_training_transitions=BASE['training_transitions'])
    return result
