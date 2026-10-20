from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from wheeled_algo.speed_reference import SpeedReferenceActor, configure_speed_reference_training
from test_static_grip_adaptation import means, observations
from test_guarded_feedback_residual import write_source
from test_guarded_feedback_residual import rsl_models
from wheeled_algo.speed_reference import extend_speed_reference_parent
from wheeled_algo.static_grip import KIND as PARENT_KIND
from wheeled_algo.speed_reference import KIND


def test_small_adapter_has_gradients_without_changing_low_command_function(means):
    parent = means[1]
    child = SpeedReferenceActor(parent, 'a'*64, [.3]*4, [.01]*2)
    actor = nn.Module(); actor.mlp = child
    actor.register_parameter('std', nn.Parameter(torch.full((6,), .04)))
    proof = configure_speed_reference_training(actor, True)
    assert proof['trainable_names'] == ['mlp.reference_margin', 'mlp.reference_lean_gain',
                                      'mlp.speed_residual.weight', 'mlp.speed_residual.bias']
    assert proof['trainable_parameters'] == 396
    low = observations(); low[:, 0].clamp_(-.75, .75); low[:, 2].clamp_(-1.5, 1.5)
    assert torch.equal(parent(low), child(low))
    obs = torch.zeros(4,35); obs[:,0]=torch.tensor([3.,-3.,0.,0.]); obs[:,2]=torch.tensor([0.,0.,4.,-4.])
    obs[:, 9] = -1.
    obs[:2,20:22]=torch.tensor([[1.,-1.],[-1.,1.]])
    before = deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    (child(obs)-child(obs).detach()-.02).square().sum().backward()
    assert torch.all(child.reference_margin.grad.abs() > 0.)
    assert torch.all(child.reference_lean_gain.grad.abs() > 0.)
    optimizer.step()
    assert all(torch.equal(actor.state_dict()[n], before[n]) for n in proof['frozen_state_names'])
    assert torch.equal(parent(low), child(low))
    assert configure_speed_reference_training(actor, False)['trainable_parameters'] == 0
    assert all(p.grad is None for p in actor.parameters())


def test_export_dynamic_batch_matches_and_checks_frozen_parent(means, tmp_path):
    parent = means[1]; child = SpeedReferenceActor(parent, 'a'*64, [.15,.3,.6,.3], [.01,.02])
    source=tmp_path/'source.onnx'; original=SimpleNamespace(mlp=parent.command.base)
    write_source(source, original)
    graph=child.export_graph(source)
    options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
    runtime=ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    for obs in (observations(), torch.zeros(1,35)):
        with torch.no_grad(): expected=child(obs).numpy()
        actual=runtime.run(None, {'obs':obs.numpy()})[0]
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6)
    with torch.no_grad(): child.static_feedback[2].bias.add_(.001)
    with pytest.raises(ValueError,match='Frozen speed-reference parent'):
        child.verify_original(original)


def test_real_rsl_adam_is_remapped_by_owner_and_new_slots_are_empty(means):
    _, actor, critic = rsl_models(means[1])
    old_names = list(dict(actor.named_parameters()))
    optimizer = torch.optim.Adam([dict(params=actor.parameters(),lr=1e-4),dict(params=critic.parameters(),lr=5e-4)])
    for parameter in (*actor.parameters(), *critic.parameters()):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    parent = dict(actor_state_dict=deepcopy(actor.state_dict()), critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=deepcopy(optimizer.state_dict()), torch_rng_state=torch.get_rng_state().clone(),
        infos=dict(actor_adaptation=PARENT_KIND,successful_updates_total=1338,consumed_optimizer_updates_total=1356))
    child = deepcopy(actor); child.mlp = SpeedReferenceActor(actor.mlp,'a'*64,[.15,.3,.05,.05],[.01,.01])
    new_optimizer = torch.optim.Adam([dict(params=child.parameters(),lr=1e-4),dict(params=critic.parameters(),lr=5e-4)])
    state, proof = extend_speed_reference_parent(parent,child,critic,new_optimizer,old_names)
    child.load_state_dict(state['actor_state_dict'], strict=True)
    new_optimizer.load_state_dict(state['optimizer_state_dict'])
    names = list(dict(child.named_parameters()))
    old_ids = dict(zip(old_names,parent['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    new_ids = dict(zip(names,state['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
    for name,index in old_ids.items():
        assert torch.equal(state['actor_state_dict'][name],parent['actor_state_dict'][name])
        for key,value in parent['optimizer_state_dict']['state'][index].items():
            assert torch.equal(state['optimizer_state_dict']['state'][new_ids[name]][key],value)
    assert new_ids['mlp.reference_margin'] not in state['optimizer_state_dict']['state']
    assert new_ids['mlp.reference_lean_gain'] not in state['optimizer_state_dict']['state']
    assert state['infos'] == parent['infos']
    assert torch.equal(state['torch_rng_state'],parent['torch_rng_state'])
    assert proof['added_parameters'] == 396


@pytest.mark.parametrize('enabled',[False, True])
def test_native_cpu_ppo_has_finite_loss_and_preserves_parent(means, enabled):
    from rsl_rl.storage import RolloutStorage
    from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
    mean=SpeedReferenceActor(means[1],'a'*64,[.15,.3,.05,.05],[-.005,-.005])
    obs,actor,critic=rsl_models(mean)
    obs['policy'][:,0]=torch.tensor([3.,-3.,0.,0.]*2)
    obs['policy'][:,2]=torch.tensor([0.,0.,4.,-4.]*2)
    obs['policy'][:,6:10]=torch.tensor([0.,0.,0.,-1.])
    obs['policy'][:,20:22]=0.
    alg=TeacherAnchoredPPO(actor,critic,RolloutStorage('rl',8,4,obs,[6]),num_learning_epochs=2,
        num_mini_batches=2,learning_rate=1e-4,schedule='fixed',use_clipped_value_loss=False)
    alg.configure_teacher(deepcopy(means[1]),coefficient=0.,action_scales=[.025]*4+[.04]*2,min_vx=.75,min_yaw=1.5)
    scope=configure_speed_reference_training(actor,enabled)
    before=deepcopy(actor.state_dict()); values=deepcopy(critic.state_dict())
    with torch.no_grad():
        for _ in range(4):
            actions=alg.act(obs)
            alg.process_env_step(obs,-(actions-.1).square().sum(-1),torch.zeros(8),{})
        alg.compute_returns(obs)
    losses=alg.update()
    assert all(np.isfinite(x) for x in losses.values())
    assert all(torch.equal(before[n],actor.state_dict()[n]) for n in scope['frozen_state_names'])
    assert any(not torch.equal(v,critic.state_dict()[n]) for n,v in values.items())
    changed=[n for n,v in before.items() if not torch.equal(v,actor.state_dict()[n])]
    assert bool(changed)==enabled
    assert set(changed)<= {'mlp.reference_margin','mlp.reference_lean_gain','mlp.speed_residual.weight','mlp.speed_residual.bias'}


@pytest.mark.parametrize('evaluation',[False,True])
def test_plan_reaches_independent_static_preflight_and_preserves_materials(tmp_path,evaluation):
    from wheeled_tasks.chassis.flat_base_continuation import read_plan,write_bundle
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    root=Path(__file__).resolve().parents[1]
    plan=read_plan(root,'contracts/flat_base_finetune_ground_0p2_0p5_v1.json')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'plan-only-authentication-fixture')
    plan.update(actor_adaptation=KIND,speed_reference=dict(checkpoint=str(parent),
        sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),margin_m_s=[.15,.3,.05,.05],lean_gain_rad_per_m_s=[-.005,-.005]))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    count=32 if evaluation else 4096
    config,seed=write_bundle(root,tmp_path/'config',count,evaluation=evaluation,plan_file=path)
    assert config['contact_material']['ground_static_friction_range']==[.2,.5]
    assert config['policy_dt']==.02 and config['physics_dt']==.001
    result=design_preflight(config,load_chassis_control(root/config['control_math_source']),root,
        stage='speed',role='quick' if evaluation else 'train',seed=seed,num_envs=count)
    assert result['status']=='static_passed'
    assert config.get('speed_reference')==(None if evaluation else plan['speed_reference'])
