"""Action meaning, fixed input units, fresh state and usable PPO gradients."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from train_prone_motion import configuration, learner_config, source_hashes
from wheeled_algo.prone_initialization import CONTRACT, initialize_prone_fresh, validate
from wheeled_algo.prone_observation import (GatedObservationNormalization,
    PhysicalPolicyPrefixNormalization, PhysicalScaleProneMotionModel, ProneMotionModel)
from wheeled_tasks.chassis.prone_motion import validate as validate_plan

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'contracts/wheel_leg_prone_motion_v4_physical_interface_v1.json'
PLAN = json.loads(PATH.read_text())
OLD = json.loads((ROOT/'contracts/wheel_leg_prone_motion_v4_shape_priority_v1.json').read_text())


def algorithm(seed=107, plan=PLAN):
    from rsl_rl.algorithms import PPO
    torch.manual_seed(seed)
    obs=TensorDict(dict(policy=torch.zeros(32,35),critic=torch.zeros(32,83)),batch_size=[32])
    cfg=learner_config(plan,'cpu');cfg['multi_gpu']=None
    alg=PPO.construct_algorithm(obs,SimpleNamespace(num_envs=32,num_actions=6),cfg,'cpu')
    receipt=initialize_prone_fresh(alg,torch.tensor([.3,-.06,-.3,.06,0.,0.]),plan)
    return alg,receipt,obs


def test_only_explicit_learning_interface_changes_contract():
    same=deepcopy(PLAN);assert same.pop('learning_interface')==CONTRACT and same==OLD
    assert validate_plan(PLAN)==PLAN and validate_plan(OLD)==OLD
    assert PLAN['multi_pose_recovery']['dense_reward']['schema']=='wheel-leg-standard-getup-standing-v4'
    assert PLAN['closure_guard']['mode']=='record_only'


@pytest.mark.parametrize('key,value', [('runtime_encoder_skip',True),('leg_position_scale_rad',1.),
    ('action_semantics','relative'),('observation_normalization','empirical'),('extra',1)])
def test_partial_or_changed_interface_fails(key,value):
    bad=deepcopy(CONTRACT);bad[key]=value
    with pytest.raises(ValueError,match='complete versioned'):validate(bad)


@pytest.mark.parametrize('dim',[35,83])
def test_policy_units_do_not_depend_on_curriculum_statistics(dim):
    n=PhysicalPolicyPrefixNormalization(dim)
    training=torch.zeros(32,dim);training[:,3]=1.6
    for _ in range(5):n.update(training)
    x=torch.randn(32,dim);x[:,3]=2.1;x[:,10]=math.pi;x[:,22]=16.
    assert torch.equal(n(x)[:,:35],x[:,:35])
    if dim>35:
        torch.testing.assert_close(n(x)[:,35:],x[:,35:]/.01)
    assert n.count==160
    n.eval();before=n.count.clone();n.update(x);assert torch.equal(n.count,before)


def test_legacy_normalization_and_checkpoint_model_are_preserved():
    legacy=GatedObservationNormalization(35);legacy.update(torch.zeros(32,35))
    x=torch.zeros(1,35);x[:,0]=.2
    assert legacy(x)[0,0]==20.  # Actual RSL epsilon0.01, not a division by zero.
    alg,_,_=algorithm(plan=OLD)
    assert isinstance(alg.actor,ProneMotionModel) and not isinstance(alg.actor,PhysicalScaleProneMotionModel)


def test_initial_absolute_leg_mean_holds_distant_poses_and_stiffens_standing_without_a_runtime_skip():
    alg,receipt,obs=algorithm()
    obs['policy']=torch.randn(32,35)*5
    center=torch.tensor([.3,-.06,-.3,.06])*.25
    delta=torch.linspace(-2.8,2.8,128).reshape(32,4)
    delta=torch.where(delta.abs()<.6,torch.full_like(delta,.6),delta)
    obs['policy'][:,10:14]=delta+center
    before=alg.actor(obs)
    torch.testing.assert_close(.25*before[:,:4],obs['policy'][:,10:14],rtol=0.,atol=1e-6)
    obs['policy'][:,10:14]=center+torch.linspace(-.08,.08,128).reshape(32,4)
    torch.testing.assert_close(.25*alg.actor(obs)[:,:4],center.expand(32,4),rtol=0.,atol=.004)
    x=obs['policy'].clone().requires_grad_()
    local=TensorDict(dict(policy=x,critic=obs['critic']),batch_size=[32])
    derivative=torch.autograd.grad((.25*alg.actor(local)[:,:4]).sum(),x)[0][:,10:14]
    assert derivative.abs().max()<.1  # Retain at least90% of the position stiffness.
    assert receipt['checkpoint_loaded'] is False and receipt['teacher_loaded'] is False
    assert receipt['runtime_action_decoder_changed'] is False and not alg.optimizer.state
    assert all(p.requires_grad for m in (alg.actor,alg.critic) for p in m.parameters())
    with torch.no_grad():
        for layer in alg.actor.mlp.modules():
            if isinstance(layer,torch.nn.Linear):layer.weight.zero_();layer.bias.zero_()
    # Decoder receives zero absolute offsets for every encoder. A relative
    # runtime decoder/skip would still follow the supplied encoder here.
    assert torch.equal(alg.actor(obs),torch.zeros(32,6))


def test_fresh_identity_is_reproducible_seed_specific_and_checkpoint_roundtrips():
    a,ra,obs=algorithm();b,rb,_=algorithm();c,rc,_=algorithm(108)
    assert ra==rb and ra['actor_tensor_sha256']!=rc['actor_tensor_sha256']
    assert ra['critic_tensor_sha256']!=rc['critic_tensor_sha256']
    b.actor.load_state_dict(a.actor.state_dict(),strict=True)
    assert torch.equal(a.actor(obs),b.actor(obs))
    legacy,_,_=algorithm(plan=OLD)
    with pytest.raises(RuntimeError):b.actor.load_state_dict(legacy.actor.state_dict(),strict=True)


def test_ppo_likelihood_and_all_six_outputs_can_train_from_the_new_origin():
    alg,_,obs=algorithm()
    obs['policy']=torch.randn(32,35);obs['critic']=torch.randn(32,83)
    mean=alg.actor(obs);alg.actor.distribution.update(mean)
    action=mean.detach()+torch.tensor([.1,-.2,.3,-.15,.2,-.1])
    loss=-(alg.actor.distribution.log_prob(action)*torch.linspace(-1.,1.,32)).mean()+alg.critic(obs).square().mean()
    before=mean.detach().clone();loss.backward()
    last=[x for x in alg.actor.mlp.modules() if isinstance(x,torch.nn.Linear)][-1]
    assert (last.weight.grad.abs().sum(-1)>0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in alg.actor.parameters())
    alg.optimizer.step();after=alg.actor(obs).detach()
    assert ((after-before).abs().sum(0)>0).all() and alg.optimizer.state


def test_interface_is_serialized_in_native_contract_and_frozen_source(tmp_path):
    configuration(tmp_path,PLAN)
    assert json.loads((tmp_path/'task_contract.json').read_text())['learning_interface']==CONTRACT
    hashes=source_hashes(PATH,PLAN)
    assert all(name in hashes for name in ['src/wheeled_algo/prone_initialization.py',
        'tests/test_prone_learning_interface.py','docs/STANDARD_GETUP_LEARNING_INTERFACE_20261103.md',str(PATH.relative_to(ROOT))])
    bad=deepcopy(PLAN);bad['ppo']=dict(hidden_dims=[3,8],epochs=3,mini_batches=4)
    with pytest.raises(ValueError,match='first16'):learner_config(bad,'cpu')
