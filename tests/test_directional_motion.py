from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from test_static_grip_adaptation import means, observations
from test_guarded_feedback_residual import write_source, rsl_models
from wheeled_algo.directional_motion import DirectionalMotionActor, configure_directional_training, extend_directional_parent
from wheeled_algo.motion_trust_region import MotionTrustRegion
from wheeled_algo.static_grip import KIND as PARENT_KIND


def test_low_commands_and_quiet_states_retain_parent_and_features_are_bounded(means):
    parent = means[1]
    child = DirectionalMotionActor(parent, 'a' * 64)
    obs = observations()
    obs[:, 0] = .5
    obs[:, 2] = 0.
    with torch.no_grad():
        child.motion_banks.weight.normal_(0, .01)
        child.motion_banks.bias.fill_(.01)
    assert torch.equal(parent(obs), child(obs))
    quiet = torch.zeros(7, 35)
    quiet[:, 9] = -1.
    assert torch.equal(parent(quiet), child(quiet))
    assert child.bounded_features(observations() * 1000).abs().max() <= 1.


def test_gradient_is_confined_to_selected_direction_and_moving_stop(means):
    child = DirectionalMotionActor(means[1], 'a' * 64)
    actor = nn.Module()
    actor.mlp = child
    actor.std = nn.Parameter(torch.ones(6))
    proof = configure_directional_training(actor, True)
    assert proof['trainable_parameters'] == 1656
    obs = torch.zeros(8, 35)
    obs[:, 0] = -3.
    obs[:, 9] = -1.
    obs[:, 3] = 2.1
    child(obs).sum().backward()
    gradient = child.motion_banks.bias.grad.reshape(4, 6)
    assert gradient[1].count_nonzero() == 6
    assert gradient[[0, 2, 3]].count_nonzero() == 0
    child.zero_grad(set_to_none=True)
    obs[:, 0] = 0.
    obs[:, 20], obs[:, 21] = -5., 5.
    child(obs).sum().backward()
    assert child.motion_banks.bias.grad.reshape(4, 6)[1].count_nonzero() == 6
    assert all(p.grad is None for n, p in actor.named_parameters() if not n.startswith('mlp.motion_banks.'))


def test_adam_extension_preserves_all_existing_slots_and_rng(means):
    parent = means[1]
    _, actor, critic = rsl_models(parent)
    old_names = list(dict(actor.named_parameters()))
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1e-4), dict(params=critic.parameters(), lr=5e-4)])
    for p in (*actor.parameters(), *critic.parameters()):
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = dict(actor_state_dict=deepcopy(actor.state_dict()), critic_state_dict=deepcopy(critic.state_dict()),
        optimizer_state_dict=deepcopy(optimizer.state_dict()), torch_rng_state=torch.get_rng_state(),
        infos=dict(actor_adaptation=PARENT_KIND, successful_updates_total=1338))
    new = deepcopy(actor)
    new.mlp = DirectionalMotionActor(parent, 'a' * 64)
    new_optimizer = torch.optim.Adam([dict(params=new.parameters(), lr=1e-4), dict(params=critic.parameters(), lr=5e-4)])
    migrated, proof = extend_directional_parent(state, new, critic, new_optimizer, old_names)
    old_ids = dict(zip(old_names, state['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
    new_ids = dict(zip(dict(new.named_parameters()), migrated['optimizer_state_dict']['param_groups'][0]['params'], strict=True))
    for name, i in old_ids.items():
        assert torch.equal(state['actor_state_dict'][name], migrated['actor_state_dict'][name])
        for key, value in state['optimizer_state_dict']['state'][i].items():
            assert torch.equal(value, migrated['optimizer_state_dict']['state'][new_ids[name]][key])
    assert new_ids['mlp.motion_banks.weight'] not in migrated['optimizer_state_dict']['state']
    assert torch.equal(state['torch_rng_state'], migrated['torch_rng_state'])
    assert proof['added_parameters'] == 1656


def test_learned_mean_exports_at_single_and_dynamic_batch(means, tmp_path):
    child = DirectionalMotionActor(means[1], 'a' * 64)
    with torch.no_grad():
        child.motion_banks.weight.normal_(0, .001)
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=child.command.base))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(child.export_graph(source).SerializeToString(), sess_options=options,
        providers=['CPUExecutionProvider'])
    for obs in (observations(), torch.zeros(1, 35)):
        actual = runtime.run(None, {'obs': obs.numpy()})[0]
        np.testing.assert_allclose(actual, child(obs).detach().numpy(), atol=2e-5, rtol=2e-5)


def test_trust_region_bounds_mean_change_without_rng_or_critic_change(means):
    _, actor, critic = rsl_models(DirectionalMotionActor(means[1], 'a' * 64))
    configure_directional_training(actor, True)
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=.1), dict(params=critic.parameters(), lr=.001)])
    obs = observations()
    old_mean = actor.mlp(obs).detach()
    std = torch.full_like(old_mean, .01)
    guard = MotionTrustRegion(actor, optimizer, .02)
    for p in actor.mlp.motion_banks.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    before_critic = deepcopy(critic.state_dict())
    rng = torch.get_rng_state()
    result = guard.project(obs, (old_mean, std))
    assert result['backtracks'] > 0 and result['scale'] < 1.
    assert result['kl'] <= .02
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, critic.state_dict()[n]) for n, v in before_critic.items())


def test_trust_region_rejects_trainable_std(means):
    _, actor, _ = rsl_models(means[1])
    with pytest.raises(ValueError, match='frozen exploration'):
        MotionTrustRegion(actor, torch.optim.Adam(actor.parameters()), .02)


def test_native_ppo_with_guard_updates_actor_and_critic_with_frozen_std():
    from test_anchored_ppo import make_algorithm, configure
    from wheeled_algo.anchored_ppo import TeacherAnchoredPPO
    alg, obs = make_algorithm(TeacherAnchoredPPO)
    configure(alg, 0.)
    alg.actor.distribution.requires_grad_(False)
    alg.configure_motion_trust_region(max_kl=.02, max_backtracks=12)
    with torch.no_grad():
        for _ in range(4):
            actions = alg.act(obs)
            alg.process_env_step(obs, -actions.square().sum(-1), torch.zeros(8), {})
        alg.compute_returns(obs)
    actor_before = deepcopy(alg.actor.state_dict())
    critic_before = deepcopy(alg.critic.state_dict())
    values = alg.update()
    assert all(torch.isfinite(torch.tensor(x)) for x in values.values())
    assert values['motion_trust_kl'] <= .02
    assert any(not torch.equal(v, alg.actor.state_dict()[n]) for n, v in actor_before.items())
    assert any(not torch.equal(v, alg.critic.state_dict()[n]) for n, v in critic_before.items())
    assert all(torch.equal(v, alg.actor.state_dict()[n]) for n, v in actor_before.items() if n.startswith('distribution.'))


def test_directional_plan_refuses_clock_material_or_budget_tampering(tmp_path):
    import json
    from pathlib import Path
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, digest, encode
    from wheeled_tasks.chassis.directional_motion_plan import make_directional_plan
    root = Path(__file__).resolve().parents[1]
    parent_file = 'reports/flat_base_speed_20261020/policy200_capacity002/initial_state.pt'
    parent = root / parent_file
    info = torch.load(parent, map_location='cpu', weights_only=True)['infos']
    base = read_plan(root, info['plan_file'])
    plan = make_directional_plan(root, base, parent_file, digest(parent), info['plan_file'], info['plan_sha256'])
    path = tmp_path / 'plan.json'
    path.write_bytes(encode(plan))
    assert read_plan(root, path)['policy_dt'] == .005
    for name, value in (('gamma', .99), ('first_block_updates', 999), ('contact_material', {'enabled': False})):
        broken = deepcopy(plan)
        broken[name] = value
        path.write_bytes(encode(broken))
        with pytest.raises(ValueError, match='undeclared'):
            read_plan(root, path)


def test_composed_runtime_keeps_parent_onnx_exact_outside_motion_gate(means, tmp_path):
    from wheeled_algo.directional_runtime import PreservedDirectionalRuntime
    parent = means[1]
    child = DirectionalMotionActor(parent, 'a' * 64)
    with torch.no_grad():
        child.motion_banks.weight.normal_(0, .001)
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=parent.command.base))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    original = ort.InferenceSession(parent.export_graph(source).SerializeToString(), sess_options=options,
        providers=['CPUExecutionProvider'])
    composed = PreservedDirectionalRuntime(parent, child)
    runtime = ort.InferenceSession(composed.export_graph(source).SerializeToString(), sess_options=options,
        providers=['CPUExecutionProvider'])
    obs = observations()
    actual = runtime.run(None, {'obs': obs.numpy()})[0]
    baseline = original.run(None, {'obs': obs.numpy()})[0]
    off = child.motion_gate(obs).flatten() == 0
    np.testing.assert_array_equal(actual[off], baseline[off])
    np.testing.assert_allclose(actual, composed(obs).detach().numpy(), atol=2e-5, rtol=2e-5)


def test_concurrent_single_frame_arms_match_sequential_bitwise(means, tmp_path):
    from wheeled_algo.single_frame_inference import SingleFrameInference
    parent = means[1]
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=parent.command.base))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    graph = parent.export_graph(source).SerializeToString()
    sessions = {i: ort.InferenceSession(graph, sess_options=options, providers=['CPUExecutionProvider']) for i in range(4)}
    frames = {i: observations()[:32].numpy() for i in range(4)}
    serial = SingleFrameInference(sessions)
    concurrent = SingleFrameInference(sessions, parallel=True)
    try:
        expected, actual = serial.run(frames), concurrent.run(frames)
        for index in sessions:
            np.testing.assert_array_equal(expected[index], actual[index])
        with pytest.raises(ValueError, match='Each independent'):
            concurrent.run({0: frames[0]})
    finally:
        serial.close()
        concurrent.close()
