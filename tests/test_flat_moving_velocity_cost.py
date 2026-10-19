"""A reward-only pilot must strengthen tracking without changing standing."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.rewards import apply_dense_tracking

ROOT = Path(__file__).resolve().parents[1]


def bundles(tmp_path, evaluation=False):
    return [write_bundle(ROOT, tmp_path / f'v{n}', 32 if evaluation else 512,
        evaluation=evaluation, plan_file=f'contracts/flat_base_finetune_v{n}.json')[0] for n in (15,16)]


def test_cost_only_plan_keeps_clocks_population_architecture_and_targets():
    old,new = [read_plan(ROOT,f'contracts/flat_base_finetune_v{n}.json') for n in (15,16)]
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    assert changed == {'schema','training_scope','moving_velocity_cost_scale'}
    assert new['moving_velocity_cost_scale'] == 3.


def test_scaled_training_design_matches_worker_and_rejects_unscaled_cost(tmp_path):
    old,new = bundles(tmp_path)
    control = load_chassis_control(ROOT/new['control_math_source'])
    assert design_preflight(new,control,ROOT,stage='speed',role='train',num_envs=512,seed=191019)['status']=='static_passed'
    assert (new['physics_dt'],new['policy_dt'],new['gamma'],new['lam']) == (.001,.02,old['gamma'],old['lam'])
    expected = deepcopy(old['dense_tracking'])
    for name in ('broad_weight','fine_weight'):
        expected['velocity'][name] *= 3.
    assert new['dense_tracking'] == expected
    for name in ('stationary_tracking','scene_groups','skill_specs','flat_rmcs_command_training',
                 'actor_adaptation','learning_rate','critic_learning_rate','initial_noise_std'):
        assert new[name] == old[name]
    bad = deepcopy(new)
    bad['dense_tracking'] = old['dense_tracking']
    with pytest.raises(ValueError,match='Preflight rejected'):
        design_preflight(bad,control,ROOT,stage='speed',role='train',num_envs=512,seed=191019)


def test_actual_cost_and_gradient_scale_both_signs_but_preserve_quiet_and_other_terms(tmp_path):
    old,new = bundles(tmp_path)
    velocity = torch.tensor([[.3,0.,0.],[-.3,0.,0.],[.1,0.,0.],[.01,0.,0.]],requires_grad=True)
    commands = torch.tensor([[.5,0.,.305],[-.5,0.,.305],[0.,4.,.305],[0.,0.,.305]])
    omega = torch.zeros(4,3)
    gravity = torch.tensor([[0.,0.,-1.]]).expand(4,-1)
    height,support = torch.full((4,),.305),torch.ones(4)
    terms = []
    for config in (old,new):
        result = {}
        apply_dense_tracking(result,velocity,omega,gravity,height,commands,support,
            torch.ones(4,dtype=torch.bool),None,config['dense_tracking'])
        terms.append(result)
    a,b = terms
    selected = ('dense_velocity_broad','dense_velocity_precision')
    for name in a:
        if name in selected:
            torch.testing.assert_close(b[name],3*a[name],rtol=2e-7,atol=1e-7)
        else:
            torch.testing.assert_close(b[name],a[name],rtol=0,atol=0)
    assert all(a[name][3].item()==b[name][3].item()==0. for name in selected)
    torch.testing.assert_close(a[selected[0]][0],a[selected[0]][1],rtol=0,atol=0)
    first = torch.autograd.grad(sum(a[k].sum() for k in selected),velocity,retain_graph=True)[0]
    second = torch.autograd.grad(sum(b[k].sum() for k in selected),velocity)[0]
    torch.testing.assert_close(second,3*first,rtol=1e-6,atol=1e-6)
    assert first[0,0] > 0 and first[1,0] < 0 and first[3,0] == 0


def test_evaluation_reward_and_acceptance_remain_identical(tmp_path):
    old,new = bundles(tmp_path,evaluation=True)
    assert new['dense_tracking'] == old['dense_tracking']
    assert new['evaluation'] == old['evaluation']
    assert (new['physics_dt'],new['policy_dt']) == (old['physics_dt'],old['policy_dt'])


@pytest.mark.parametrize('scale',[True,0.,1.,-1.,10.01,float('nan'),float('inf'),None])
def test_invalid_scales_are_rejected(tmp_path,scale):
    plan = read_plan(ROOT,'contracts/flat_base_finetune_v16.json')
    plan['moving_velocity_cost_scale'] = scale
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='moving velocity cost scale'):
        read_plan(ROOT,path)


def test_historical_plan_cannot_silently_enable_new_cost(tmp_path):
    plan = read_plan(ROOT,'contracts/flat_base_finetune_v15.json')
    plan['moving_velocity_cost_scale'] = 3.
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='explicit v16'):
        read_plan(ROOT,path)
