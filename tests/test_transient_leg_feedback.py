from copy import deepcopy
from pathlib import Path

import pytest
import torch

from test_static_grip_adaptation import means
from test_transient_response import frames
from test_transient_campaign_plan import campaign
from wheeled_algo.directional_motion import DirectionalMotionActor
from wheeled_algo.directional_runtime import preserve_directional_graph, PreservedDirectionalRuntime
from wheeled_algo.transient_leg_feedback import SETTINGS
from wheeled_algo.transient_motion import ACTOR_SETTINGS
from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan
from wheeled_tasks.chassis.transient_dwell_plan import EVIDENCE as DWELL_EVIDENCE
from wheeled_tasks.chassis.transient_leg_feedback_plan import (
    make_leg_feedback_plan, validate_leg_feedback_migration, EVIDENCE, DECISION)

ROOT = Path(__file__).resolve().parents[1]


def pair(means):
    old = DirectionalMotionActor(means[1], 'a'*64, transient_settings=ACTOR_SETTINGS)
    new = DirectionalMotionActor(means[1], 'a'*64, transient_settings=ACTOR_SETTINGS,
                                 leg_feedback_settings=SETTINGS)
    with torch.no_grad():
        old.motion_banks.bias.fill_(.03)
    new.load_state_dict(old.state_dict())
    return old, new


def test_parent_leg_recurrence_changes_but_learned_leg_intent_stays_direct(means):
    old, new = pair(means)
    obs = frames()
    obs[:, 0], obs[:, 22:28] = 3., .5
    parent, original = old.conditioned_parent(obs), old(obs)
    delta = original-parent
    expected = torch.cat((obs[:, 22:26] + .25*(parent[:, :4]-obs[:, 22:26]), parent[:, 4:]), -1) + delta
    torch.testing.assert_close(new(obs), expected)
    assert torch.equal(new(obs)[:, 4:], original[:, 4:])
    # A direct learned correction keeps its original gradient instead of .25.
    old(obs)[:, :4].sum().backward()
    new(obs)[:, :4].sum().backward()
    torch.testing.assert_close(old.motion_banks.bias.grad, new.motion_banks.bias.grad)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


def test_declared_quiet_dynamics_and_export_scope_are_explicit(means):
    old, new = pair(means)
    obs = frames()
    obs[:, 22:26] = .5
    assert new.motion_gate(obs).count_nonzero() == 0
    assert not torch.equal(old(obs)[:, :4], new(obs)[:, :4])
    assert torch.equal(old(obs)[:, 4:], new(obs)[:, 4:])
    assert preserve_directional_graph(new, None) is new
    with pytest.raises(ValueError, match='complete mean'):
        PreservedDirectionalRuntime(means[1], new)
    bad = deepcopy(SETTINGS)
    bad['wheels'] = 'relaxed'
    with pytest.raises(ValueError, match='Undeclared'):
        DirectionalMotionActor(means[1], 'a'*64, transient_settings=ACTOR_SETTINGS,
                               leg_feedback_settings=bad)


def test_complete_export_keeps_fast_residual_and_wheel_math(means, tmp_path):
    import numpy as np
    import onnxruntime as ort
    from types import SimpleNamespace
    from test_guarded_feedback_residual import write_source
    _, new = pair(means)
    source = tmp_path/'source.onnx'
    write_source(source, SimpleNamespace(mlp=new.command.base))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(new.export_graph(source).SerializeToString(),
                                  sess_options=options, providers=['CPUExecutionProvider'])
    obs = frames(8)
    obs[:, 0] = torch.linspace(-3., 3., 8)
    obs[:, 22:28] = torch.randn(8, 6)
    np.testing.assert_allclose(session.run(None, {'obs': obs.numpy()})[0],
                               new(obs).detach().numpy(), atol=2e-5, rtol=2e-5)


def recipe():
    base, plan = campaign()
    return make_leg_feedback_plan(base, plan['transient_campaign'],
                                  digest(ROOT/DWELL_EVIDENCE), digest(ROOT/EVIDENCE), digest(ROOT/DECISION))


def test_recipe_preserves_goals_physics_population_reward_and_consumption(tmp_path):
    plan = recipe()
    path = tmp_path/'plan.json'
    path.write_bytes(encode(plan))
    assert read_plan(ROOT, path) == plan
    base, _ = campaign()
    for key in ('policy_dt', 'physics_dt', 'num_steps_per_env', 'gamma', 'lam',
                'transient_response', 'contact_material', 'response_targets', 'learning_rate'):
        assert plan[key] == base[key]
    assert plan['transient_campaign']['short_update_ceiling'] == 512
    assert plan['additional_critic_warmup_updates'] == 0
    plan['transient_leg_feedback']['actor']['parent_leg_alpha'] = .5
    path.write_bytes(encode(plan))
    with pytest.raises(ValueError, match='undeclared'):
        read_plan(ROOT, path)


def test_migration_rejects_reward_or_sensor_source_drift():
    plan = recipe()
    old = deepcopy(plan)
    old.pop('transient_leg_feedback')
    old['schema'] = 'flat-base-finetune-plan-v22'
    info = dict(num_envs=512, source_sha256={'src/wheeled_algo/directional_motion.py': 'old', 'sensors.py': 'same'})
    source = {'src/wheeled_algo/directional_motion.py': 'new', 'sensors.py': 'same'}
    proof = validate_leg_feedback_migration(info, old, plan, source)
    assert proof['additional_critic_warmup'] == 0
    with pytest.raises(ValueError, match='unrelated'):
        validate_leg_feedback_migration(info, old, plan, {**source, 'sensors.py': 'changed'})
    altered = deepcopy(plan)
    altered['learning_rate'] *= 2
    with pytest.raises(ValueError, match='unrelated'):
        validate_leg_feedback_migration(info, old, altered, source)
