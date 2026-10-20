"""Motion adaptation must preserve the trained static head and full checkpoint contract."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from test_static_grip_adaptation import means, observations
from test_guarded_feedback_residual import write_source

ROOT = Path(__file__).resolve().parents[4]
from wheeled_algo import motion_grip as proposal
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.static_grip import KIND as PARENT_KIND
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
import hashlib


def motion(means):
    parent = means[1]
    with torch.no_grad():
        parent.static_feedback[2].bias.add_(.015)
        parent.fast_feedback[2].weight.normal_(std=.01)
    return parent, proposal.MotionGripActor(parent, 'a' * 64)


def test_identity_retains_existing_parameter_order_and_function(means):
    parent, child = motion(means)
    assert list(parent.state_dict()) == list(child.state_dict())
    assert list(dict(parent.named_parameters())) == list(dict(child.named_parameters()))
    assert all(torch.equal(value, child.state_dict()[name]) for name, value in parent.state_dict().items())
    obs = observations()
    assert torch.equal(parent(obs), child(obs))


def test_gradient_updates_motion_heads_and_preserves_stable_static_function(means):
    parent, child = motion(means)
    actor = nn.Module()
    actor.mlp = child
    actor.register_parameter('std', nn.Parameter(torch.full((6,), .04)))
    proof = proposal.configure_motion_grip_training(actor, True)
    assert proof['trainable_parameters'] == 1126 * 2 + 994
    assert len(proof['trainable_names']) == 12
    obs = observations()
    static = torch.randn(64, 35)
    static[:, :3] = 0.
    static[:, 20:22] = .1
    before = deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=.0001)
    output = child(obs).detach()
    (child(obs) - output - .02).square().mean().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() for name, p in actor.named_parameters()
               if name in proof['trainable_names'])
    assert all(p.grad is None for name, p in actor.named_parameters() if name not in proof['trainable_names'])
    optimizer.step()
    assert all(not torch.equal(actor.state_dict()[name], before[name]) for name in proof['trainable_names'])
    assert all(torch.equal(actor.state_dict()[name], before[name]) for name in proof['frozen_state_names'])
    assert torch.equal(parent(static), child(static))
    assert not torch.equal(parent(obs), child(obs))
    assert proposal.configure_motion_grip_training(actor, False)['trainable_parameters'] == 0
    assert all(p.grad is None for p in actor.parameters())


def test_export_matches_adapted_motion_and_preserves_static_reference(means, tmp_path):
    parent, child = motion(means)
    original = SimpleNamespace(mlp=parent.command.base)
    with torch.no_grad():
        child.feedback[2].bias.add_(.02)
        child.fast_feedback[2].bias.sub_(.02)
        child.wheel_authority[2].bias.add_(.03)
    source = tmp_path / 'source.onnx'
    write_source(source, original)
    graph = child.export_graph(source)
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata['actor_adaptation'] == proposal.KIND
    assert 'static_grip_adaptation' not in metadata
    assert not json.loads(metadata['static_feedback_recall'])['static_trainable']
    assert json.loads(metadata['motion_grip_adaptation'])['parent_checkpoint_sha256'] == 'a' * 64
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    obs = observations()
    with torch.inference_mode():
        expected = child(obs).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    with torch.no_grad():
        child.static_feedback[2].bias.add_(.001)
    with pytest.raises(ValueError, match='Frozen adapted static'):
        child.verify_original(original)


def test_rejects_other_parent_and_undeclared_command_training(means):
    with pytest.raises(ValueError, match='exact v7'):
        proposal.MotionGripActor(means[0], 'a' * 64)
    _, child = motion(means)
    with pytest.raises(ValueError, match='freezes the command'):
        child.set_trainable(True, train_command=True)


def test_checkpoint_read_uses_authenticated_v7_static_reference(means, tmp_path):
    parent = means[1]
    with torch.no_grad():
        parent.static_feedback[2].bias.add_(.015)
    recalled = means[2]
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=parent.command.base))
    actor_state = lambda mean: {'mlp.' + name: value for name, value in mean.state_dict().items()}
    parent_path = tmp_path / 'parent.pt'
    torch.save(dict(infos=dict(actor_adaptation=PARENT_KIND,
        static_feedback_recall=dict(checkpoint=str(recalled), sha256=hashlib.sha256(recalled.read_bytes()).hexdigest())),
        actor_state_dict=actor_state(parent)), parent_path)
    reference = dict(checkpoint=str(parent_path), sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest())
    child = proposal.MotionGripActor(parent, reference['sha256'])
    with torch.no_grad():
        child.fast_feedback[2].bias.add_(.02)
    state = dict(infos=dict(actor_adaptation=proposal.KIND, motion_grip_parent=reference),
                 actor_state_dict=actor_state(child))
    loaded = mean_from_checkpoint(state, source)
    obs = observations()
    assert torch.equal(loaded(obs), child(obs))
    assert type(loaded) is proposal.MotionGripActor
    bad = deepcopy(state)
    bad['actor_state_dict']['mlp.static_feedback.2.bias'].add_(.001)
    with pytest.raises(ValueError, match='Frozen adapted static'):
        mean_from_checkpoint(bad, source)
    bad = deepcopy(state)
    bad['infos']['motion_grip_parent']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='parent checkpoint changed'):
        mean_from_checkpoint(bad, source)
    bad = deepcopy(state)
    bad['infos']['motion_grip_parent'] = None
    with pytest.raises(ValueError, match='authenticated v7 parent'):
        mean_from_checkpoint(bad, source)


def motion_plan(tmp_path):
    parent = tmp_path / 'v7.pt'
    parent.write_bytes(b'authenticated-parent-placeholder-for-plan-only')
    plan = read_plan(ROOT, 'contracts/flat_base_finetune_ground_0p2_0p5_v1.json')
    plan.update(actor_adaptation=proposal.KIND,
                motion_grip_parent=dict(checkpoint=str(parent), sha256=hashlib.sha256(parent.read_bytes()).hexdigest()))
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return plan, path


@pytest.mark.parametrize('evaluation', [False, True])
def test_motion_plan_reaches_cpu_preflight(evaluation, tmp_path):
    plan, path = motion_plan(tmp_path)
    count = 32 if evaluation else 4096
    config, seed = write_bundle(ROOT, tmp_path / 'config', count, evaluation=evaluation, plan_file=path)
    assert config['contact_material']['ground_static_friction_range'] == [.2, .5]
    assert config['physics_dt'] == .001 and config['policy_dt'] == .02
    assert config.get('actor_adaptation') == (None if evaluation else proposal.KIND)
    assert config.get('motion_grip_parent') == (None if evaluation else plan['motion_grip_parent'])
    receipt = design_preflight(config, load_chassis_control(ROOT / config['control_math_source']),
        ROOT, stage='speed', role='quick' if evaluation else 'train', num_envs=count, seed=seed)
    assert receipt['status'] == 'static_passed'


def test_motion_plan_rejects_changed_reference_and_other_kind(tmp_path):
    plan, path = motion_plan(tmp_path)
    bad = deepcopy(plan)
    bad['motion_grip_parent']['sha256'] = '0' * 64
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='exact declared v7'):
        read_plan(ROOT, path)
    bad = deepcopy(plan)
    bad['actor_adaptation'] = PARENT_KIND
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='only valid for the declared v8'):
        read_plan(ROOT, path)


def test_motion_reader_rejects_a_hashed_parent_of_the_wrong_architecture(tmp_path):
    parent = tmp_path / 'wrong_parent.pt'
    torch.save(dict(infos=dict(actor_adaptation='original-static-feedback-and-wheel-authority-residual-v6')),
               parent)
    reference = dict(checkpoint=str(parent), sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    state = dict(infos=dict(actor_adaptation=proposal.KIND, motion_grip_parent=reference), actor_state_dict={})
    with pytest.raises(ValueError, match='declared v7 checkpoint'):
        mean_from_checkpoint(state, tmp_path / 'unused_source.onnx')
