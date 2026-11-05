"""A geometric lift gate must not change control, capture or acceptance."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis import recovery_reward
from wheeled_tasks.chassis.multi_pose_recovery import recovery_terms
from wheeled_tasks.chassis.prone_motion import validate

ROOT = Path(__file__).resolve().parents[1]


def state(count=3):
    return dict(gravity=torch.tensor([[0., 0., -1.]]*count), height=torch.full((count,), .22),
        omega=torch.full((count, 3), .2), velocity=torch.full((count, 3), .1),
        recovery_support_error_m=torch.full((count,), .03),
        recovery_wheel_load_fraction=torch.zeros(count, 2),
        recovery_chassis_load_fraction=torch.full((count,), .5),
        torque=torch.full((count, 6), 3.), action_delta=torch.full((count,), .1))


def test_only_the_versioned_lift_gate_changes_in_the_complete_training_plan():
    old=json.loads((ROOT/'contracts/wheel_leg_self_righting_encoder_increment_range_v1.json').read_text())
    new=json.loads((ROOT/'contracts/wheel_leg_self_righting_geometric_lift_v1.json').read_text())
    assert validate(new) == new
    assert validate(old) == old
    assert recovery_reward.validate(new['multi_pose_recovery']['dense_reward']) == recovery_reward.geometric_lift_contract()
    adjusted=deepcopy(new)
    adjusted['multi_pose_recovery']['dense_reward']=old['multi_pose_recovery']['dense_reward']
    assert adjusted == old
    adjusted=deepcopy(new['multi_pose_recovery']['dense_reward'])
    adjusted.pop('lift_gate');adjusted['schema']=old['multi_pose_recovery']['dense_reward']['schema']
    assert adjusted == old['multi_pose_recovery']['dense_reward']


def test_unloaded_and_single_wheel_lift_get_equal_geometric_guidance_without_capture_credit():
    s=state();s['recovery_wheel_load_fraction'][:]=torch.tensor([[0.,0.],[.5,0.],[.5,.5]])
    old=recovery_reward.dense_terms(s,recovery_reward.contract())
    new=recovery_reward.dense_terms(s,recovery_reward.geometric_lift_contract())
    assert old['recovery_lift'][:2].count_nonzero() == 0
    for key in ('recovery_lift','recovery_lift_error'):
        assert new[key][0] > 0 and torch.equal(new[key],new[key][0].expand(3))
    assert new['recovery_support_capture'][:2].count_nonzero() == 0
    assert new['recovery_support_capture'][2] > 0
    for key in old.keys()-{'recovery_lift','recovery_lift_error'}:
        assert torch.equal(new[key],old[key])


def test_lift_height_gradient_exists_before_wheel_load_without_a_target_state_accumulator():
    s=state(1);s['height'].requires_grad_()
    before={k:v.clone() for k,v in s.items()}
    terms=recovery_reward.dense_terms(s,recovery_reward.geometric_lift_contract())
    (terms['recovery_lift']+terms['recovery_lift_error']).sum().backward()
    assert torch.isfinite(s['height'].grad).all() and s['height'].grad.item() > 0
    assert all(torch.equal(s[k],v) for k,v in before.items())


@pytest.mark.parametrize('gz',[0.,.5,1.])
def test_side_and_inverted_states_get_no_lift_credit(gz):
    s=state();s['gravity'][:,2]=gz
    terms=recovery_reward.dense_terms(s,recovery_reward.geometric_lift_contract())
    assert terms['recovery_lift'].count_nonzero() == 0
    assert terms['recovery_lift_error'].count_nonzero() == 0


def test_full_bilateral_load_is_exactly_the_original_reward():
    s=state(31);s['recovery_wheel_load_fraction'][:]=.5
    s['height']=torch.linspace(.1,.6,31)
    a=recovery_reward.dense_terms(s,recovery_reward.contract())
    b=recovery_reward.dense_terms(s,recovery_reward.geometric_lift_contract())
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_legacy_formula_is_bit_identical_for_seeded_states():
    gen=torch.Generator().manual_seed(61);s=state(1024)
    for key in s:s[key]=torch.rand(s[key].shape,generator=gen)
    s['gravity'][:,2]=2.*s['gravity'][:,2]-1.
    c=recovery_reward.contract();up=(-s['gravity'][:,2]).clamp(0.,1.)
    # Match the original multiplication order as well as the equation.
    alignment=1./(1.+(s['recovery_support_error_m']/.12).square())
    load=(s['recovery_wheel_load_fraction']/.15).clamp(0.,1.).amin(-1)
    capture=up.square()*alignment*load
    error=s['height']-.305;body_rate=s['omega'][:,:2].square().sum(-1)
    reference=math.log1p(((.173-.305)/.08)**2)
    expected=dict(recovery_orientation=2.*(-s['gravity'][:,2]).clamp(-1.,1.),
        recovery_support_capture=2.*capture,
        recovery_lift=4.*capture*torch.exp(-(error/.06).square()),
        recovery_lift_error=2.*capture*(reference-torch.log1p((error/.08).square())),
        recovery_phase_body_rate=-(.01+.12*capture)*body_rate,
        recovery_capture_speed=-.2*capture*s['velocity'][:,:2].square().sum(-1),
        recovery_capture_body_contact=-4.*capture*s['recovery_chassis_load_fraction'])
    actual=recovery_reward.dense_terms(s,c)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k],expected[k]) for k in actual)


def test_original_time_effort_action_and_impact_densities_are_preserved():
    p=json.loads((ROOT/'contracts/wheel_leg_self_righting_geometric_lift_v1.json').read_text())
    s=state();density=torch.tensor([.1,.2,.3]);c=p['multi_pose_recovery']
    actual=recovery_terms(s,c,density)
    assert torch.equal(actual['rise_time'],-torch.ones(3))
    assert torch.equal(actual['impact'],-density)
    assert torch.allclose(actual['effort'],torch.full((3,),-.000025*9))
    assert torch.allclose(actual['action_rate'],torch.full((3,),-.0001))


@pytest.mark.parametrize('change',[lambda c:c.pop('lift_gate'),lambda c:c.update(lift_gate='actual_height_done'),
    lambda c:c.update(lift_weight=8.),lambda c:c.update(observation_additions=True)])
def test_partial_or_silently_changed_contract_is_rejected(change):
    c=recovery_reward.geometric_lift_contract();change(c)
    with pytest.raises(ValueError):recovery_reward.validate(c)
