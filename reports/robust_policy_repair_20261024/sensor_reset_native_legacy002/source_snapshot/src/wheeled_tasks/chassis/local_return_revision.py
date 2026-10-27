"""Exact full-state continuation into one bounded longer-return experiment."""
from copy import deepcopy
import json
from pathlib import Path

from . import local_scratch as local


REVISION = {'schema': 'v6-local-return-revision-v1',
 'source_plan_file': 'contracts/v6_local_scratch_200hz_v7.json',
 'source_plan_sha256': '4066923002b9f694c41a3048f8820f3eb3494a26469f47ee1e0ebb770eb93689',
 'checkpoint_file': '/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/local_200hz_20261018/pilot015/model_final.pt',
 'checkpoint_sha256': '988af5220e5de9cb3a00024c04c0eaec38f1570bf15805b03618b82187ed8479',
 'source_contract_sha256': '533382b32cbec3a6be0f36d72bfb94197f981c2f2f985b2800da17b5987cf7c9',
 'completion_sha256': 'ba34070e1964c99867263f26b3c59331ae1e983e5578d20a85fd2045f97821ed',
 'seal_sha256': '7090e2a7dc2d97d995f144535d00d1a797f77edc36204cc77b85348338055524',
 'learning_updates': 2688,
 'prior_consumption': 1216,
 'lineage_update_ceiling': 18784,
 'first_block_update_ceiling': 128,
 'training_transitions': 132120576,
 'source_num_steps_per_env': 96,
 'target_num_steps_per_env': 384,
 'source_lam': 0.9872585449014338,
 'target_lam': 1.0,
 'evidence': [{'file': 'reports/local_training_20261017/pilot015.json',
               'sha256': 'fd4b3893a8598fc545db8c46a079ff4e40918f359c3a46300118fef64411c944'},
              {'file': 'reports/local_training_20261017/entropy_ab001_credit_followup.json',
               'sha256': '74f56ece5782dd08056278a584bafadf0c3ab8f3e76d1537769b3b6034603c21'},
              {'file': 'reports/local_training_20261017/entropy_ab001_birth_credit.json',
               'sha256': '6032a0c460dd02bb656d7878769cbfe593fb7be8d779a5902efe9ae2b537d095'},
              {'file': 'reports/local_training_20261017/action_horizon_diagnosis001.json',
               'sha256': '999b31c5f7a4c2df6c8bf828fcf629ec5893c2af4a6e2ec994c53625242729de'},
              {'file': 'reports/local_training_20261017/return_sensitivity001.json',
               'sha256': '5359be9c9dd0495e71b656a0de5ab5fb278483a7ce6c6de325ab2f9ec87ef12e'},
              {'file': 'docs/V6_LOCAL_RETURN_HORIZON_PLAN_20261018.md',
               'sha256': '91d6d411258f2eb1266245ae9c41f6f0930c8b704da3d0b21708e9871135d314'}]}


def validate_plan(plan, root):
    root = Path(root)
    if local.digest(root / local.ENTROPY_AB_PLAN_FILE) != REVISION['source_plan_sha256']:
        raise ValueError('Local return revision source v7 plan changed')
    expected = deepcopy(local.read_plan(root, local.ENTROPY_AB_PLAN_FILE))
    expected.update(initialization='full_state_2688_return_window_trial',
        local_return_revision=deepcopy(REVISION))
    expected['ppo_timing'].update(num_steps_per_env=384, lam=1.)
    if plan != expected:
        raise ValueError('Local return revision differs from its exact window derivation')
    for row in REVISION['evidence']:
        if local.digest(root / row['file']) != row['sha256']:
            raise ValueError('Local return revision evidence changed: ' + row['file'])


def validate_contracts(source, target, root):
    if (source.get('local_scratch', {}).get('plan_file') != local.ENTROPY_AB_PLAN_FILE
            or target.get('local_scratch', {}).get('plan_file') != local.RETURN_PLAN_FILE):
        raise ValueError('Local return revision requires exact local v7 -> v8 contracts')
    budget = local.validate_config(target, root)
    _, original = local.build_bundle(root, Path(source['design_preflight']['file']).parent,
        source['target_num_envs'], plan_file=local.ENTROPY_AB_PLAN_FILE)
    if original['worker'] != source:
        raise ValueError('Local return source is not its independent v7 derivation')
    allowed = {'num_steps_per_env', 'lam', 'ppo_minibatch_samples', 'local_scratch',
        'design_preflight', 'budget_spec_file', 'budget_spec_sha256'}
    if {k: v for k, v in source.items() if k not in allowed} != {
            k: v for k, v in target.items() if k not in allowed}:
        raise ValueError('Local return revision changes an undeclared contract field')
    expected_transitions = REVISION['training_transitions'] + (
        REVISION['lineage_update_ceiling'] - REVISION['learning_updates']) * 512 * 384
    if (source['num_steps_per_env'] != 96 or source['lam'] != REVISION['source_lam']
            or target['num_steps_per_env'] != 384 or target['lam'] != 1.
            or target['gamma'] != .9974905699336811 or target['entropy_coef'] != .0005
            or source['target_num_envs'] != 512 or target['ppo_minibatch_samples'] != 49152
            or budget['inherited_consumption'] != 1216 or budget['update_ceiling'] != 18784
            or budget['transition_ceiling'] != expected_transitions):
        raise ValueError('Local return revision changed timing, population or consumed budget')
    return dict(schema=REVISION['schema'], changed_contract_fields=sorted(allowed),
        source_num_steps_per_env=96, target_num_steps_per_env=384,
        source_lam=source['lam'], target_lam=1., gamma=target['gamma'],
        consumption='all_old_lineages_and_both_entropy_arms_preserved_no_refund',
        additional_critic_warmup=0)


def validate_parent(checkpoint, target, root):
    from .full_curriculum import checkpoint_contract_path
    checkpoint = Path(checkpoint).resolve(strict=True)
    source_path = checkpoint_contract_path(checkpoint)
    source = json.loads(source_path.read_text())
    result = validate_contracts(source, target, root)
    count = local.validate_parent(checkpoint, source)
    if (checkpoint != Path(REVISION['checkpoint_file'])
            or local.digest(checkpoint) != REVISION['checkpoint_sha256']
            or local.digest(source_path) != REVISION['source_contract_sha256']
            or local.digest(checkpoint.parent / 'completion.json') != REVISION['completion_sha256']
            or local.digest(checkpoint.with_suffix('.checkpoint.json')) != REVISION['seal_sha256']
            or count != 2688):
        raise ValueError('Local return revision requires exact finalized pilot015-2688')
    completion = json.loads((checkpoint.parent / 'completion.json').read_text())
    seal = json.loads(checkpoint.with_suffix('.checkpoint.json').read_text())
    if (completion['status'] != 'completed' or seal.get('global_actor_update') != count
            or completion['parent_training_transitions'] + completion['successful_updates']
                * source['target_num_envs'] * source['num_steps_per_env'] != REVISION['training_transitions']):
        raise ValueError('Local return revision parent learning/exposure clocks changed')
    result.update(parent_updates=count, inherited_consumption=1216,
        total_prior_consumption=3904, lineage_update_ceiling=18784,
        parent_training_transitions=REVISION['training_transitions'])
    return result
