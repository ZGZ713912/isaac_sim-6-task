from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch

from wheeled_algo.intent_speed_reference import IntentSpeedReferenceActor,KIND as PARENT_KIND
from wheeled_algo.motion_state_residual import MotionStateResidualActor,configure_motion_state_training,extend_motion_state_parent
from test_static_grip_adaptation import means,observations
from test_guarded_feedback_residual import rsl_models,write_source


def pair(parent):
    args=(parent,'a'*64,[.15,.3,.02,.02],[-.04,-.01],.5)
    return IntentSpeedReferenceActor(*args),MotionStateResidualActor(*args)


def test_zero_extension_identity_and_moving_stop_is_trainable(means):
    parent,child=pair(means[1]);obs=observations()
    assert torch.equal(parent(obs),child(obs))
    stop=torch.zeros(3,35);stop[:,9]=-1.;stop[:,3]=1.525
    stop[0,20]=5.;stop[0,21]=-5.;stop[1,6]=2.
    assert child.motion_gate(stop).flatten().tolist()==[1.,1.,0.]
    with torch.no_grad():child.motion_residual.bias.fill_(.01)
    assert not torch.equal(parent(stop[:2]),child(stop[:2]))
    assert torch.equal(parent(stop[2:]),child(stop[2:]))
    low=observations();low[:,0]=.5;low[:,2]=0.
    assert torch.equal(parent(low),child(low))
    low[:,0]=0.;low[:,2]=1.
    assert torch.equal(parent(low),child(low))


@pytest.mark.parametrize('enabled',[False,True])
def test_native_ppo_can_update_a_moving_stop_without_mutating_v10(means,enabled):
    from rsl_rl.storage import RolloutStorage
    from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
    parent,child=pair(means[1]);obs,actor,critic=rsl_models(child)
    obs['policy'][:,0]=0.;obs['policy'][:,2]=0.;obs['policy'][:,20]=5.;obs['policy'][:,21]=-5.
    alg=TeacherAnchoredPPO(actor,critic,RolloutStorage('rl',8,4,obs,[6]),num_learning_epochs=2,
        num_mini_batches=2,learning_rate=1e-3,schedule='fixed',use_clipped_value_loss=False)
    alg.configure_teacher(parent,coefficient=0.,action_scales=[.025]*4+[.04]*2,min_vx=.75,min_yaw=1.5)
    scope=configure_motion_state_training(actor,enabled)
    assert scope['trainable_parameters']==(414 if enabled else 0)
    before=deepcopy(actor.state_dict())
    with torch.no_grad():
        for _ in range(4):
            actions=alg.act(obs);alg.process_env_step(obs,-(actions-.1).square().sum(-1),torch.zeros(8),{})
        alg.compute_returns(obs)
    assert all(np.isfinite(v)for v in alg.update().values())
    assert all(torch.equal(v,actor.state_dict()[n])for n,v in before.items()if n in scope['frozen_state_names'])
    changed={n for n,v in before.items()if not torch.equal(v,actor.state_dict()[n])}
    assert changed==({'mlp.motion_residual.weight','mlp.motion_residual.bias'}if enabled else set())


def test_adam_extension_keeps_v10_parameter_owners(means):
    parent,child=pair(means[1]);_,actor,critic=rsl_models(parent)
    old_names=list(dict(actor.named_parameters()))
    opt=torch.optim.Adam([dict(params=actor.parameters(),lr=1e-4),dict(params=critic.parameters(),lr=5e-4)])
    for p in (*actor.parameters(),*critic.parameters()):p.grad=torch.ones_like(p)
    opt.step();opt.zero_grad(set_to_none=True)
    state=dict(actor_state_dict=deepcopy(actor.state_dict()),critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=deepcopy(opt.state_dict()),torch_rng_state=torch.get_rng_state(),
        infos=dict(actor_adaptation=PARENT_KIND,successful_updates_total=1338))
    new=deepcopy(actor);new.mlp=child
    new_opt=torch.optim.Adam([dict(params=new.parameters(),lr=1e-3),dict(params=critic.parameters(),lr=5e-4)])
    migrated,proof=extend_motion_state_parent(state,new,critic,new_opt,old_names)
    new_ids=dict(zip(dict(new.named_parameters()),migrated['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    old_ids=dict(zip(old_names,state['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    for n,i in old_ids.items():
        assert torch.equal(state['actor_state_dict'][n],migrated['actor_state_dict'][n])
        for k,v in state['optimizer_state_dict']['state'][i].items():assert torch.equal(v,migrated['optimizer_state_dict']['state'][new_ids[n]][k])
    assert new_ids['mlp.motion_residual.weight']not in migrated['optimizer_state_dict']['state']
    assert proof['initial_mean_is_identity']and proof['added_parameters']==414


def test_motion_state_onnx_with_learned_head(means,tmp_path):
    _,mean=pair(means[1])
    with torch.no_grad():mean.motion_residual.weight.normal_(0,.001)
    source=tmp_path/'source.onnx';write_source(source,SimpleNamespace(mlp=mean.command.base))
    graph=mean.export_graph(source)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    runtime=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    for obs in (observations(),torch.zeros(1,35)):
        np.testing.assert_allclose(runtime.run(None,{'obs':obs.numpy()})[0],mean(obs).detach().numpy(),atol=5e-6,rtol=5e-6)


def test_yaw_stop_profile_contains_both_signs_and_two_stops():
    from wheeled_tasks.chassis.skill_commands import profile_command
    t=torch.tensor([0.,2.99,3.,5.99,6.,8.99,9.,11.99])
    profile=dict(kind='rotate',yaw_start_stop_segment_seconds=3.)
    actual=profile_command(t,[0.,4.,.305],profile)
    assert actual[:,1].tolist()==[4.,4.,0.,0.,-4.,-4.,0.,0.]
    assert actual[:,0].count_nonzero()==0


def test_yaw_stop_profile_reaches_the_runtime_command_target():
    from wheeled_tasks.chassis.skill_commands import SkillCommands
    commands=SkillCommands.__new__(SkillCommands)
    commands.env=SimpleNamespace(episode_length_buf=torch.tensor([0,150,300,450]),policy_dt=.02,
        command_target=torch.zeros(4,2))
    commands.batches=[(torch.arange(4),dict(kind='rotate',command=[0.,4.,.305],yaw_start_stop_segment_seconds=3.))]
    commands.start_stop_delay=torch.zeros(4)
    commands.command_base=torch.tensor([[0.,4.,.305]]*4)
    commands.heading=None
    commands.yaw=lambda:torch.zeros(4)
    commands.apply_command_sequence=lambda:None
    commands.apply_large_stair_commands=lambda:None
    commands.update()
    assert commands.env.command_target[:,1].tolist()==[4.,0.,-4.,0.]
