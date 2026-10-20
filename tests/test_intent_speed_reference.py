from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import onnxruntime as ort
import pytest
import torch

from wheeled_algo.intent_speed_reference import (
    IntentSpeedReferenceActor, configure_intent_speed_training, extend_intent_speed_parent)
from wheeled_algo.static_grip import KIND as PARENT_KIND
from test_static_grip_adaptation import means, observations
from test_guarded_feedback_residual import rsl_models, write_source


def child(parent):
    return IntentSpeedReferenceActor(parent,'a'*64,[.15,.3,.02,.02],[-.04,-.01],.5)


def test_low_exact_and_pure_high_initialization_matches_the_native_teacher(means):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from flat_speed_prior import traction_reference_actions
    parent=means[1];mean=child(parent)
    low=observations();low[:,0].clamp_(-.75,.75);low[:,2].clamp_(-1.5,1.5)
    assert torch.equal(mean(low),parent(low))
    high=observations().repeat(6,1)
    count=len(high)//6
    high[:,0]=torch.tensor([2.,-2.,3.,-3.,0.,0.]).repeat_interleave(count)
    high[:,2]=torch.tensor([0.,0.,0.,0.,4.,-4.]).repeat_interleave(count)
    high[:,3]=2.1
    with torch.no_grad():
        torch.testing.assert_close(mean(high),traction_reference_actions(high,parent,torch.tensor(0.)),atol=3e-6,rtol=3e-6)


def test_residual_retains_the_external_goal_after_conditioning(means):
    mean=child(means[1]);obs=torch.zeros(2,35)
    obs[:,0]=torch.tensor([-2.,-3.]);obs[:,9]=-1.;obs[:,3]=2.1
    assert torch.equal(mean(obs)[0],mean(obs)[1])
    with torch.no_grad():mean.speed_residual.weight[4,64]=1.
    assert mean(obs)[0,4]!=mean(obs)[1,4]


@pytest.mark.parametrize('enabled',[False,True])
def test_projected_native_ppo_preserves_frozen_weights_and_actor_freezes(means,enabled):
    from rsl_rl.storage import RolloutStorage
    from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
    mean=child(means[1]);obs,actor,critic=rsl_models(mean)
    obs['policy'][:,0]=-3.;obs['policy'][:,9]=-1.;obs['policy'][:,3]=2.1
    alg=TeacherAnchoredPPO(actor,critic,RolloutStorage('rl',8,4,obs,[6]),num_learning_epochs=2,
        num_mini_batches=2,learning_rate=1e-4,schedule='fixed',use_clipped_value_loss=False)
    alg.configure_teacher(deepcopy(means[1]),coefficient=0.,action_scales=[.025]*4+[.04]*2,min_vx=.75,min_yaw=1.5)
    scope=configure_intent_speed_training(actor,enabled)
    assert scope['trainable_parameters']==(421 if enabled else 0)
    with torch.no_grad():
        mean.reference_margin.fill_(.001);mean.reference_lean_gain.fill_(.05);mean.reference_encoder_gain.fill_(.7)
    before=deepcopy(actor.state_dict())
    with torch.no_grad():
        for _ in range(4):
            actions=alg.act(obs);alg.process_env_step(obs,-(actions-.1).square().sum(-1),torch.zeros(8),{})
        alg.compute_returns(obs)
    assert all(np.isfinite(x)for x in alg.update().values())
    assert all(torch.equal(before[n],actor.state_dict()[n])for n in scope['frozen_state_names'])
    if enabled:
        assert mean.reference_encoder_gain<=.6 and mean.reference_lean_gain.abs().max()<=.04
        assert (mean.reference_margin>=torch.tensor([.05,.05,.005,.005])).all()
    else:
        assert all(torch.equal(v,actor.state_dict()[n])for n,v in before.items())


def test_real_adam_migration_preserves_existing_owners_and_counters(means):
    _,actor,critic=rsl_models(means[1]);old_names=list(dict(actor.named_parameters()))
    opt=torch.optim.Adam([dict(params=actor.parameters(),lr=1e-4),dict(params=critic.parameters(),lr=5e-4)])
    for p in (*actor.parameters(),*critic.parameters()):p.grad=torch.ones_like(p)
    opt.step();opt.zero_grad(set_to_none=True)
    state=dict(actor_state_dict=deepcopy(actor.state_dict()),critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=deepcopy(opt.state_dict()),torch_rng_state=torch.get_rng_state().clone(),
        infos=dict(actor_adaptation=PARENT_KIND,successful_updates_total=1338))
    new=deepcopy(actor);new.mlp=child(actor.mlp)
    new_opt=torch.optim.Adam([dict(params=new.parameters(),lr=1e-4),dict(params=critic.parameters(),lr=5e-4)])
    migrated,proof=extend_intent_speed_parent(state,new,critic,new_opt,old_names)
    new.load_state_dict(migrated['actor_state_dict'],strict=True);new_opt.load_state_dict(migrated['optimizer_state_dict'])
    ids=dict(zip(dict(new.named_parameters()),migrated['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    old_ids=dict(zip(old_names,state['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    for n,index in old_ids.items():
        assert torch.equal(state['actor_state_dict'][n],migrated['actor_state_dict'][n])
        for key,value in state['optimizer_state_dict']['state'][index].items():
            assert torch.equal(value,migrated['optimizer_state_dict']['state'][ids[n]][key])
    assert ids['mlp.reference_encoder_gain']not in migrated['optimizer_state_dict']['state']
    assert proof['added_parameters']==421 and migrated['infos']==state['infos']
    assert torch.equal(migrated['torch_rng_state'],state['torch_rng_state'])


def test_onnx_dynamic_batches_match_nonzero_head(means,tmp_path):
    mean=child(means[1])
    with torch.no_grad():mean.speed_residual.weight.normal_(0,.001)
    source=tmp_path/'source.onnx';write_source(source,SimpleNamespace(mlp=mean.command.base))
    graph=mean.export_graph(source)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    runtime=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    for obs in (observations(),torch.zeros(1,35)):
        np.testing.assert_allclose(runtime.run(None,{'obs':obs.numpy()})[0],mean(obs).detach().numpy(),atol=4e-6,rtol=4e-6)
