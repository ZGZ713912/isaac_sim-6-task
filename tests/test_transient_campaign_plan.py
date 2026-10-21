"""Smaller populations keep the objective; 5000 updates require behavior proof."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from wheeled_tasks.chassis.flat_base_continuation import digest, read_plan
from wheeled_tasks.chassis.transient_campaign_plan import make_campaign_plan, validate_admission
from train_flat_base import restored_training_frames, validate_population_transition


def campaign():
    base_file = 'reports/flat_transient_response_20261021/candidate003/plan.json'
    base = read_plan(ROOT, base_file)
    checkpoint = base['transient_response']['checkpoint']
    spec = dict(base_plan_file=base_file, base_plan_sha256=digest(ROOT/base_file),
        checkpoint=checkpoint, checkpoint_sha256=digest(ROOT/checkpoint),
        parent_plan_file=base_file, parent_plan_sha256=digest(ROOT/base_file),
        num_envs=512, stage='pilot', lineage_start_successful_updates=1346,
        short_update_ceiling=512, admission=[])
    return base, make_campaign_plan(base, spec)


def test_population_plan_changes_no_objective_physics_sampling_or_actor(tmp_path):
    base, plan = campaign()
    changed = {k for k in base.keys() | plan.keys() if base.get(k) != plan.get(k)}
    assert changed == {'schema', 'allowed_num_envs', 'default_num_envs', 'first_block_updates',
                       'additional_critic_warmup_updates', 'checkpoint_interval', 'transient_campaign'}
    assert (plan['policy_dt'], plan['physics_dt'], plan['num_steps_per_env']) == (.005, .001, 384)
    assert plan['first_block_updates'] == 64 and plan['additional_critic_warmup_updates'] == 0
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    assert read_plan(ROOT, path) == plan


@pytest.mark.parametrize('field,value', [('policy_dt', .02), ('learning_rate', .001),
    ('num_steps_per_env', 96), ('first_block_updates', 5000), ('checkpoint_interval', 100)])
def test_undeclared_changes_cannot_hide_in_population_migration(tmp_path, field, value):
    _, plan = campaign()
    plan[field] = value
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        read_plan(ROOT, path)


def test_population_receipt_keeps_all_inherited_exposure_and_has_no_extra_warmup():
    old, new = campaign()
    info = dict(num_envs=4096, successful_updates_total=1354,
        training_transitions_total=121896960,
        source_sha256={'scripts/train_flat_base.py': 'old', 'actor.py': 'actor'})
    expected = dict(schema='flat-training-population-transition-v1', old_num_envs=4096,
        new_num_envs=512, num_steps_per_env=384, inherited_successful_updates=1354,
        inherited_successful_training_frames=121896960, additional_critic_warmup_updates=0)
    source = {'scripts/train_flat_base.py': 'new', 'actor.py': 'actor',
              'src/wheeled_tasks/chassis/transient_campaign_plan.py': 'new'}
    assert validate_population_transition(info, old, new, 512, dict(population_transition=expected), source) == expected
    bad = deepcopy(expected)
    bad['inherited_successful_training_frames'] = 1354*512*384
    with pytest.raises(ValueError, match='exact inherited exposure'):
        validate_population_transition(info, old, new, 512, dict(population_transition=bad), source)
    with pytest.raises(ValueError, match='reviewed trainer'):
        validate_population_transition(info, old, new, 512, dict(population_transition=expected), {**source, 'actor.py': 'changed'})
    assert restored_training_frames(info, old) == 121896960


def test_long_stage_rejects_loss_only_or_missing_native_confirmation(tmp_path):
    with pytest.raises(ValueError, match='two matched'):
        validate_admission(ROOT, [], 'unused')
    _, plan = campaign()
    plan = make_campaign_plan(read_plan(ROOT, plan['transient_campaign']['base_plan_file']),
                              {**plan['transient_campaign'], 'stage': 'long5000'})
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='two matched'):
        read_plan(ROOT, path)


def confirmation(tmp_path, seed):
    from flat_transient_metrics import RETENTION, STEADY, TRANSIENT, assess_response
    from wheeled_tasks.chassis.transient_campaign_plan import BASELINE_SHA256
    folder = tmp_path/str(seed)
    folder.mkdir()
    native = dict(status='evaluated', replicas=4, policy_hz=200, physics_hz=1000,
        heights=[.305, .42], ground_friction=[.2, .5], horizon_seconds=16., seed=seed,
        case_names=list(RETENTION+STEADY+TRANSIENT),
        code_sha256={'scripts/flat_transient_metrics.py': digest(ROOT/'scripts/flat_transient_metrics.py')},
        checkpoint_comparison={'0': dict(role='parent', sha256=BASELINE_SHA256),
                               '2': dict(role='trained', sha256='trained')}, result=dict(cases={}))
    summaries = dict(parent={}, trained={})
    for height in (.305, .42):
        for mu in (.2, .5):
            for name in RETENTION+STEADY+TRANSIENT:
                condition = f'h{height:g}_mu{mu:g}_{name}'
                for index, role, response in [(0, 'parent', 1.), (2, 'trained', .5)]:
                    native['result']['cases'][f'g{index}_{condition}'] = dict(
                        closure_gap_max_m=.001, tilt_max_deg=3., checks=dict(survival=True),
                        vx_mae_m_s=.05, yaw_mae_rad_s=.05, height_mae_m=.001)
                    summaries[role][condition] = dict(case_name=name, complete=4, requested=4,
                        censored_stops=0, censored_launches=0, final_planar_m_s=.01,
                        final_yaw_abs_rad_s=.01, settle_time_s=response, response90_s=response,
                        stop_distance_m=response)
    (folder/'report.json').write_text(json.dumps(native))
    analysis = dict(schema='flat-transient-native-response-v1',
        native_report_sha256=digest(folder/'report.json'),
        checkpoint_comparison=native['checkpoint_comparison'], summaries=summaries,
        admission=assess_response(native, summaries))
    path = folder/'response.json'
    path.write_text(json.dumps(analysis))
    return dict(file=str(path), sha256=digest(path))


def test_long_admission_recomputes_complete_paired_results_and_rejects_duplicate_seed(tmp_path):
    first, second = confirmation(tmp_path, 1), confirmation(tmp_path, 2)
    assert validate_admission(ROOT, [first, second], 'trained')['seeds'] == [1, 2]
    with pytest.raises(ValueError, match='distinct reset'):
        validate_admission(ROOT, [first, first], 'trained')
    with pytest.raises(ValueError, match='exact checkpoint'):
        validate_admission(ROOT, [first, second], 'different')
    path = Path(second['file'])
    analysis = json.loads(path.read_text())
    # A stale admitted=true field cannot hide an incomplete physical episode.
    next(iter(analysis['summaries']['trained'].values()))['complete'] = 3
    path.write_text(json.dumps(analysis))
    second['sha256'] = digest(path)
    with pytest.raises(ValueError, match='exact checkpoint'):
        validate_admission(ROOT, [first, second], 'trained')


def test_diagnostic_inference_override_cannot_certify_sealed_checkpoint(tmp_path):
    first, second = confirmation(tmp_path, 1), confirmation(tmp_path, 2)
    path = Path(first['file'])
    native_path = path.parent/'report.json'
    native = json.loads(native_path.read_text())
    native['diagnostic_inference_override'] = {'relaxation': .25}
    native_path.write_text(json.dumps(native))
    analysis = json.loads(path.read_text())
    analysis['native_report_sha256'] = digest(native_path)
    path.write_text(json.dumps(analysis))
    first['sha256'] = digest(path)
    with pytest.raises(ValueError, match='exact checkpoint'):
        validate_admission(ROOT, [first, second], 'trained')


def test_extended_course_contains_full_speed_braking_without_changing_other_horizons(tmp_path):
    from wheeled_tasks.chassis.transient_dwell_plan import make_dwell_plan, EVIDENCE
    base, campaign_plan = campaign()
    plan = make_dwell_plan(base, campaign_plan['transient_campaign'], digest(ROOT/EVIDENCE))
    by_name = {g['name']: g for g in base['train_groups']}
    for group in plan['train_groups']:
        skill = group['skill']
        name = group['name']
        assert group['fraction'] == by_name[name]['fraction']
        if skill['kind'] == 'start_stop' or 'yaw_start_stop_segment_seconds' in skill:
            assert skill.get('segment_seconds', skill.get('yaw_start_stop_segment_seconds')) == 8.
            assert plan['group_episode_seconds'][name] == 32.
        else:
            assert group == by_name[name]
            assert plan['group_episode_seconds'][name] == base.get('group_episode_seconds', {}).get(name, base['episode_seconds'])
    for field in ('policy_dt', 'physics_dt', 'gamma', 'lam', 'num_steps_per_env', 'learning_rate',
                  'transient_response', 'contact_material', 'motion_trust_region'):
        assert plan[field] == base[field]
    path = tmp_path/'dwell.json'
    path.write_text(json.dumps(plan))
    assert read_plan(ROOT, path) == plan
    plan['train_groups'][0]['fraction'] += .01
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='undeclared setting'):
        read_plan(ROOT, path)
