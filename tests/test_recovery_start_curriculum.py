"""Backward start difficulty never consumes holdouts or unlocks capabilities."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.cold_evaluation_lifecycle import reset_retired_fixed_cases
from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
from wheeled_tasks.chassis.multi_pose_recovery import FAMILIES, LEG_CLASSES, ResetSampler
from wheeled_tasks.chassis.prone_motion import validate as validate_plan
from wheeled_tasks.chassis.recovery_start_curriculum import select_start

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/'contracts/wheel_leg_self_righting_near_start_v1.json').read_text())
BANK = json.loads((ROOT/PLAN['pose_bank']['file']).read_text())


def sampler(seed=7):
    return ResetSampler(BANK, PLAN['multi_pose_recovery'], seed)


def test_training_pools_are_nested_native_admitted_static_and_disjoint_from_holdouts():
    c = sampler().start_curriculum
    assert list(map(len, c.pools)) == [1,4,7,19]
    for limit, pool in zip(c.config['belly_orbit_limits_deg'], c.pools):
        for index in pool:
            p = BANK['poses'][index]
            r = p['recovery']
            assert p['admitted_as_training_reset'] and r['native_passive_admitted'] and r['cad_surface_screen_passed']
            assert r['split'] == 'train' and not r['dynamic'] and r['family'] == 'belly_down'
            assert max(map(abs,r['orbital_pair_deg'])) <= limit
    assert all(set(a) <= set(b) for a,b in zip(c.pools,c.pools[1:]))


def test_training_retains_all_seven_families_and_four_leg_classes_from_first_stage():
    s = sampler()
    rows = [s.sample_start(10000) for _ in range(5000)]
    full = [BANK['poses'][i]['recovery'] for i,phase in rows if phase == -1]
    assert .07 < len(full)/len(rows) < .13
    assert {r['family'] for r in full} == set(FAMILIES)
    assert {r['leg_class'] for r in full} == set(LEG_CLASSES)
    assert all(BANK['poses'][i]['recovery']['split'] == 'train' for i,_ in rows)
    assert s.start_curriculum.phase == 0  # PPO counters never advance it.


def test_gate_needs_64_actual_episodes_and_75_percent_and_rejects_old_birth_credit():
    c = sampler().start_curriculum
    c.observe([0]*63,[True]*63)
    assert c.phase == 0
    c.observe([0],[True])
    assert c.phase == 1 and c.advancements == [dict(from_phase=0,trials=64,successes=64)]
    c.observe([0]*64+[-1]*64,[True]*128)
    assert c.phase == 1 and c.trials[1] == 0
    c.observe([1]*64,[True]*47+[False]*17)
    assert c.phase == 1
    c.observe([1]*64,[True]*48+[False]*16)
    assert c.phase == 2


def test_one_interval_cannot_advance_multiple_stages_or_receive_future_credit():
    c = sampler().start_curriculum
    c.observe([0]*64+[1]*64,[True]*128)
    assert c.phase == 1 and c.trials == [64,0,0,0,0]
    for phase in range(1,4):
        c.observe([phase]*64,[True]*64)
    assert c.phase == 4
    marker = object()
    assert c.sample(lambda:marker) == (marker,4)


def test_checkpoint_preserves_both_sampler_streams_phase_history_and_future_choices():
    s = sampler()
    for _ in range(31):s.sample_start(250)
    s.start_curriculum.observe([0]*64,[True]*64)
    state = s.state_dict()
    ResetSampler.validate_state(state)
    expected = [s.sample_start(99999) for _ in range(300)]
    resumed = sampler(999)
    resumed.load_state_dict(state)
    assert [resumed.sample_start(99999) for _ in range(300)] == expected
    assert resumed.start_curriculum.phase == 1
    broken = deepcopy(state)
    broken.pop('start_curriculum')
    with pytest.raises(ValueError):resumed.load_state_dict(broken)
    legacy = ResetSampler(BANK, {k:v for k,v in PLAN['multi_pose_recovery'].items() if k != 'start_curriculum'},7)
    with pytest.raises(ValueError):legacy.load_state_dict(state)


def test_playback_uses_original_fallen_sampler_without_consuming_training_curriculum_rng():
    s = sampler()
    snapshot = s.state_dict()
    start_before = s.start_curriculum.state_dict()
    actual = [select_start(s,250,training=False) for _ in range(100)]
    expected = sampler()
    expected.load_state_dict(snapshot)
    assert actual == [(expected.sample(250),-1) for _ in range(100)]
    assert s.start_curriculum.state_dict() == start_before


@pytest.mark.parametrize('mutation', ['phase','window','advancement','rng','unadvanced','future','missing_totals'])
def test_corrupt_curriculum_checkpoint_is_rejected(mutation):
    c = sampler().start_curriculum
    state = c.state_dict()
    if mutation == 'phase':state['phase'] = 1
    if mutation == 'window':state['window'] = [True]*65
    if mutation == 'advancement':state.update(phase=1,advancements=[dict(from_phase=0,trials=64,successes=47)])
    if mutation == 'rng':state['rng'] = (2,)
    if mutation == 'unadvanced':state.update(window=[True]*64,trials=[64,0,0,0,0],successes=[64,0,0,0,0])
    if mutation == 'future':state['trials'][1] = 1
    if mutation == 'missing_totals':state.update(phase=1,advancements=[dict(from_phase=0,trials=64,successes=64)])
    with pytest.raises(ValueError):c.load_state_dict(state)


def test_retiring_cold_lanes_preserves_curriculum_progress_and_random_stream():
    s = sampler()
    before = s.start_curriculum.state_dict()
    env = SimpleNamespace(fixed_cases=[{}],finished=torch.tensor([True,False]),
        generator=torch.Generator(),domain=SimpleNamespace(rng=torch.Generator()),
        recovery_sampler=s,prone_reset_generation=7,cold_evaluation_reset_events=0)
    def reset(ids):
        s.sample_start(5000)
        s.start_curriculum.observe([0]*64,[True]*64)
        env.finished[ids] = False
    env.reset = reset
    assert reset_retired_fixed_cases(env) == 1
    assert s.start_curriculum.state_dict() == before
    assert env.finished.tolist() == [True,False]


def test_formal_candidate_preserves_plant_rewards_full_evaluation_and_capability_gates():
    validate_plan(PLAN)
    baseline = json.loads((ROOT/'contracts/wheel_leg_self_righting_recovery_focus_v1.json').read_text())
    changed = {k for k in baseline if baseline[k] != PLAN[k]}
    assert changed == {'reference','multi_pose_recovery'}
    old_course = GlobalCurriculum(baseline, sampler().mastery)
    new_course = GlobalCurriculum(PLAN, sampler().mastery)
    assert old_course.jobs(250) == new_course.jobs(250)
    assert not new_course.recovery.unlocked and not new_course.domains_admitted
    invalid = deepcopy(PLAN)
    invalid['global_curriculum'].pop('recovery_focus')
    with pytest.raises(ValueError):validate_plan(invalid)
