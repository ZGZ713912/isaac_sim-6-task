from copy import deepcopy

import pytest
import torch
from torch import nn

from wheeled_algo.getup_symmetry import reflect_actions,reflect_observation
from wheeled_algo.motion_target_damping import (
    VERSION,MODULE,MotionTargetDampedGapActor,damp_motion_targets,
    motion_damping_gate,validate_motion_damping_checkpoint,
)
from wheeled_algo.paired_recovery import wrap
from wheeled_algo.periodic_entry_actor import PeriodicEntryGapActor


def sample(n=12):
    obs = torch.zeros(n,35)
    obs[:,0],obs[:,2],obs[:,3],obs[:,9] = .5,-1.,.305*5.,-1.
    action = torch.zeros(n,6)
    action[:,:4] = .08
    action[:,4:] = torch.randn(n,2)
    return obs,action,torch.tensor([16.]*4+[9.]*2)


def test_alpha_one_is_exact_identity_and_has_no_new_state():
    obs,a,bounds = sample()
    assert damp_motion_targets(a,obs,1.,bounds) is a
    kw = dict(upright_entry_recovery=True,canonical_recovery=True,smooth_knee_targets=True,
              recovery_leg_step=.3,final_target_winding_version='paired-v1',
              cyclic_leg_targets=True,leg_action_clip=16.)
    old = PeriodicEntryGapActor(nn.Linear(35,6),**kw)
    new = MotionTargetDampedGapActor(deepcopy(old.parent),motion_leg_alpha=1.,**kw)
    new.load_state_dict(old.state_dict(),strict=True)
    assert torch.equal(old(obs),new(obs))
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())


def test_active_motion_small_target_step_is_damped_and_wheels_are_exact():
    obs,a,bounds = sample()
    out = damp_motion_targets(a,obs,.8,bounds)
    torch.testing.assert_close(out[:,:4],.8*a[:,:4],rtol=0.,atol=1e-7)
    assert torch.equal(out[:,4:],a[:,4:])


@pytest.mark.parametrize('mode',['quiet','fallen','entry'])
def test_inactive_paths_are_bit_exact(mode):
    obs,a,bounds = sample()
    if mode=='quiet':
        obs[:,0:3] = 0.
        obs[:,4:7] = 3.
    elif mode=='fallen':
        obs[:,9] = 1.
    else:
        obs[:,10:14] = torch.tensor([-2.5,-2.5,2.5,2.5])
    assert torch.equal(motion_damping_gate(obs),torch.zeros(len(obs),1))
    assert torch.equal(a,damp_motion_targets(a,obs,.8,bounds))


def test_stale_previous_target_cannot_bypass_upright_error_limit():
    obs,a,bounds = sample(200)
    obs[:,22:26] = torch.randn(200,4)*30.
    a[:,:4] = torch.empty(200,4).uniform_(-.45,.45)
    out = damp_motion_targets(a,obs,.8,bounds)
    assert wrap(out[:,:4]*.25-obs[:,10:14]).abs().max()<=.120001
    assert torch.equal(out[:,4:],a[:,4:])


def test_previous_action_periodic_representation_and_reflection():
    obs,a,bounds = sample()
    obs[:,10:14] = torch.tensor([-.03,.04,-.02,.05])
    obs[:,22:26] = torch.tensor([.1,-.1,.2,-.2])
    shifted = obs.clone()
    shifted[:,22:26] += torch.tensor([2.,-2.,4.,-4.])*torch.pi/.25
    first = damp_motion_targets(a,obs,.8,bounds)
    second = damp_motion_targets(a,shifted,.8,bounds)
    torch.testing.assert_close(first,second,rtol=0.,atol=2e-6)
    reflected = reflect_actions(damp_motion_targets(reflect_actions(a),reflect_observation(obs),.8,bounds))
    torch.testing.assert_close(first,reflected,rtol=0.,atol=1e-7)


def test_actor_and_previous_target_gradients_are_finite():
    obs,a,bounds = sample()
    obs.requires_grad_()
    a.requires_grad_()
    damp_motion_targets(a,obs,.8,bounds).square().sum().backward()
    assert torch.isfinite(obs.grad).all() and torch.isfinite(a.grad).all()
    assert obs.grad[:,22:26].abs().sum()>0 and a.grad[:,:4].abs().sum()>0


@pytest.mark.parametrize('alpha',[True,0.,1.01,float('nan'),float('inf')])
def test_invalid_alpha_is_rejected(alpha):
    obs,a,bounds = sample()
    with pytest.raises(ValueError,match='alpha'):
        damp_motion_targets(a,obs,alpha,bounds)


def states():
    source = dict(infos=dict(completed_updates=1501,settings=dict(gap_entry_encoder_version='periodic-chart-v1'),
                            source_sha256={},rng=torch.arange(4)),
                  actor_state_dict={'weight':torch.ones(2)},critic_state_dict={'weight':torch.ones(1)},
                  optimizer_state_dict={'moments':torch.ones(2)})
    state = deepcopy(source)
    state['infos']['settings'].update(motion_leg_damping_version=VERSION,motion_leg_damping_alpha=.8)
    state['infos']['source_sha256'][MODULE] = 'authenticated-by-factory'
    state['infos']['motion_leg_damping_migration'] = dict(schema='motion-leg-damping-migration-v1',
        new_PPO=0,learning_clock=1501)
    state['infos']['hardware_ready'] = False
    return state,source


def test_explicit_migration_preserves_all_saved_state():
    validate_motion_damping_checkpoint(*states())


@pytest.mark.parametrize('tamper',['actor','critic','optimizer','rng','clock','settings'])
def test_migration_rejects_unrelated_changes(tamper):
    state,source = states()
    if tamper in ('actor','critic','optimizer'):
        next(iter(state[tamper+'_state_dict'].values())).add_(1.)
    elif tamper=='rng':
        state['infos']['rng'][0]+=1
    elif tamper=='clock':
        state['infos']['completed_updates']+=1
    else:
        state['infos']['settings']['unrelated']=True
    with pytest.raises(ValueError):
        validate_motion_damping_checkpoint(state,source)
