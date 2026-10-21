from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from test_static_grip_adaptation import means
from wheeled_algo.directional_motion import DirectionalMotionActor, configure_directional_training
from wheeled_algo.directional_runtime import MotionSelector
from wheeled_algo.transient_motion import ACTOR_SETTINGS, migrate_actor_buffer
from wheeled_tasks.chassis.skill_commands import profile_command
from wheeled_tasks.chassis.transient_response import REWARD_SETTINGS, TransientResponse
from wheeled_tasks.chassis.transient_response_plan import make_transient_plan

ROOT = Path(__file__).resolve().parents[1]


def frames(count=4):
    obs = torch.zeros(count, 35)
    obs[:, 3], obs[:, 9] = 1.525, -1.
    return obs


def test_small_stop_errors_and_launch_have_leg_gradients_while_quiet_remains_exact(means):
    old = DirectionalMotionActor(means[1], 'a' * 64)
    new = DirectionalMotionActor(means[1], 'a' * 64, transient_settings=ACTOR_SETTINGS)
    obs = frames()
    obs[0, 6], obs[1, 6] = .05, -.05  # +/-0.1rad/s, previously closed.
    obs[2, 20], obs[2, 21] = .1, -.1  # 0.06m/s rolling at zero request.
    obs[3, 0] = .5  # Low-speed launch also used to be frozen.
    assert old.motion_gate(obs).count_nonzero() == 0
    assert (new.motion_gate(obs) > 0).all()
    torch.testing.assert_close(MotionSelector(ACTOR_SETTINGS)(obs), new.motion_gate(obs) > 0.)
    new(obs).sum().backward()
    assert new.motion_banks.bias.grad.reshape(4, 6)[:, :4].count_nonzero()
    quiet = frames()
    with torch.no_grad():
        new.motion_banks.bias.fill_(.3)
    assert torch.equal(new(quiet), old(quiet))


def test_opt_in_leg_bounds_and_frozen_parent_are_enforced(means):
    child = DirectionalMotionActor(means[1], 'a' * 64, transient_settings=ACTOR_SETTINGS)
    actor = torch.nn.Module()
    actor.mlp = child
    proof = configure_directional_training(actor, True)
    assert proof['trainable_parameters'] == 1656
    obs = frames()
    obs[:, 0] = 3.
    with torch.no_grad():
        child.motion_banks.bias.fill_(100.)
    delta = child(obs) - child.conditioned_parent(obs)
    torch.testing.assert_close(delta, torch.tensor([.6]*4+[.8]*2).expand_as(delta))
    assert all(not p.requires_grad for n, p in actor.named_parameters() if not n.startswith('mlp.motion_banks.'))
    bad = deepcopy(ACTOR_SETTINGS)
    bad['raw_action_bounds'][0] = 10.
    with pytest.raises(ValueError, match='declared'):
        DirectionalMotionActor(means[1], 'a'*64, transient_settings=bad)


def test_actual_parent_migration_changes_only_declared_buffer_and_no_rng():
    parent = torch.load(ROOT/'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt',
                        map_location='cpu', weights_only=True)
    rng = torch.get_rng_state()
    migrated = migrate_actor_buffer(parent, ACTOR_SETTINGS)
    changed = [key for key, value in parent['actor_state_dict'].items()
               if not torch.equal(value, migrated['actor_state_dict'][key])]
    assert changed == ['mlp.motion_bound']
    assert torch.equal(torch.get_rng_state(), rng)
    assert parent['infos'] == migrated['infos']
    for key in ('critic_state_dict',):
        assert all(torch.equal(value, migrated[key][name]) for name, value in parent[key].items())
    for key, state in parent['optimizer_state_dict']['state'].items():
        for name, value in state.items():
            assert torch.equal(value, migrated['optimizer_state_dict']['state'][key][name])


def test_transient_export_preserves_quiet_parent_and_carries_small_yaw_correction(means, tmp_path):
    import numpy as np
    import onnxruntime as ort
    from types import SimpleNamespace
    from test_guarded_feedback_residual import write_source
    from wheeled_algo.directional_runtime import PreservedDirectionalRuntime
    child = DirectionalMotionActor(means[1], 'a'*64, transient_settings=ACTOR_SETTINGS)
    with torch.no_grad():
        child.motion_banks.bias.fill_(.02)
    runtime_mean = PreservedDirectionalRuntime(means[1], child)
    source = tmp_path/'source.onnx'
    write_source(source, SimpleNamespace(mlp=child.command.base))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(runtime_mean.export_graph(source).SerializeToString(),
        sess_options=options, providers=['CPUExecutionProvider'])
    parent_runtime = ort.InferenceSession(means[1].export_graph(source).SerializeToString(),
        sess_options=options, providers=['CPUExecutionProvider'])
    obs = frames()
    obs[1, 6], obs[2, 6], obs[3, 0] = .05, -.05, .5
    actual = runtime.run(None, {'obs': obs.numpy()})[0]
    baseline = parent_runtime.run(None, {'obs': obs.numpy()})[0]
    np.testing.assert_array_equal(actual[:1], baseline[:1])
    assert np.abs(actual[1:] - baseline[1:]).max() > .005
    np.testing.assert_allclose(actual, runtime_mean(obs).detach().numpy(), atol=2e-5, rtol=2e-5)


def reward_inputs(command=1., pitch_deg=6.):
    angle = math.radians(pitch_deg)
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]])
    return (torch.zeros(1, 3), torch.zeros(1, 3), gravity,
            torch.tensor([[command, 0., .305]]), torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool), torch.zeros(1, 6))


def terms():
    return {k: torch.tensor([v]) for k, v in dict(pitch_velocity=-2., locomotion_pitch_floor=-1.,
        pitch_exp=.5, locomotion_roll_floor=-3., leg_action_smoothness=-2., motor_torque=-4.,
        track_height=1.).items()}


def test_pitch_relaxation_is_supported_transient_only_and_not_a_leg_motion_bonus():
    accumulator = TransientResponse(REWARD_SETTINGS, 'cpu')
    current = terms()
    accumulator.apply(current, *reward_inputs())
    assert current['pitch_velocity'] == 0. and current['leg_action_smoothness'] == -1.
    assert current['transient_response_error'] < 0.
    assert current['motor_torque'] == -4. and current['locomotion_roll_floor'] == -3.
    assert current['track_height'] == 1.
    quiet = terms()
    accumulator.apply(quiet, *reward_inputs(command=0.))
    for name, value in terms().items():
        torch.testing.assert_close(quiet[name], value)
    assert quiet['transient_response_error'] == 0.
    unsupported = list(reward_inputs())
    unsupported[5] = torch.zeros(1, dtype=torch.bool)
    rejected = terms()
    accumulator.apply(rejected, *unsupported)
    torch.testing.assert_close(rejected['pitch_velocity'], terms()['pitch_velocity'])
    # Pitch outside the transient allowance still pays a posture cost.
    excessive = terms()
    accumulator.apply(excessive, *reward_inputs(pitch_deg=12.))
    assert excessive['pitch_velocity'] < 0.
    result = accumulator.drain()
    assert result['FlatTransient/sample_frames'] == 4
    assert accumulator.drain() == {}
    # Small residual speed/yaw still receives a linear stopping cost.
    for error in (.1, .2):
        inputs = reward_inputs(command=error, pitch_deg=0.)
        current = terms()
        accumulator.apply(current, *inputs)
        assert current['transient_response_error'].item() == pytest.approx(-1.5 * error / .5)


def test_asymmetric_complete_cycle_and_legacy_profile_are_distinct():
    time = torch.tensor([0., 2.5, 5., 7.5, 10.])
    profile = dict(kind='start_stop', segment_seconds=2.5, opposite_speed_ratio=5./6.)
    result = profile_command(time, [3., 0., .305], profile)
    torch.testing.assert_close(result[:, 0], torch.tensor([3., 0., -2.5, 0., 3.]))
    profile['opposite_speed_ratio'] = 6./5.
    result = profile_command(time, [-2.5, 0., .305], profile)
    torch.testing.assert_close(result[:, 0], torch.tensor([-2.5, 0., 3., 0., -2.5]))
    profile.pop('opposite_speed_ratio')
    result = profile_command(time, [3., 0., .305], profile)
    torch.testing.assert_close(result[:, 0], torch.tensor([3., 0., -3., 0., 3.]))
    for invalid in (0., -1., float('nan'), True, 10.):
        profile['opposite_speed_ratio'] = invalid
        with pytest.raises(ValueError):
            profile_command(time, [3., 0., .305], profile)


def test_plan_material_clock_budget_and_parent_cannot_be_silently_changed(tmp_path):
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, digest, encode
    parent = 'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'
    base_file = 'reports/flat_base_speed_20261020/directional200_preparation001/plan.json'
    base = read_plan(ROOT, base_file)
    plan = make_transient_plan(base, parent, digest(ROOT/parent), base_file, digest(ROOT/base_file))
    path = tmp_path/'plan.json'
    path.write_bytes(encode(plan))
    assert read_plan(ROOT, path) == plan
    assert sum(g['fraction'] for g in plan['train_groups']) == pytest.approx(1.)
    assert plan['contact_material'] == base['contact_material']
    assert plan['acceptance'] == base['acceptance']
    assert plan['response_targets']['reverse_max_m_s'] == 2.5
    for field, value in [('physics_dt', .005), ('first_block_updates', 10000),
                         ('response_targets', {}), ('contact_material', {'enabled': False})]:
        broken = deepcopy(plan)
        broken[field] = value
        path.write_bytes(encode(broken))
        with pytest.raises(ValueError, match='undeclared|200 Hz'):
            read_plan(ROOT, path)


def test_new_envelope_changes_future_targets_without_regrading_historical_cases():
    from wheeled_tasks.chassis.transient_response_plan import response_envelope
    base = json.loads((ROOT/'contracts/v6_task_campaign_speed_worker_v18.json').read_text())
    original = deepcopy(base['evaluation']['cases'])
    selected = [c for c in original if c['name'] in
                ('backward_3', 'start_stop_3', 'rotate_4', 'rotate_4_reverse')]
    revised = response_envelope(selected)
    cases = {c['name']: c for c in revised}
    assert cases['backward_2p5']['command'][0] == -2.5
    assert cases['backward_2p5']['skill']['command'][0] == -2.5
    assert cases['start_stop_3_to_2p5']['skill']['opposite_speed_ratio'] == 5./6.
    assert cases['yaw_start_stop_negative']['skill']['yaw_start_stop_segment_seconds'] == 2.5
    assert original == base['evaluation']['cases']
    # The independent raw-request reconstruction uses the same preceding tick.
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from analyze_flat_raw_requests import requested_planar
    case = cases['start_stop_3_to_2p5']
    ticks = np.array([1, 501, 1001, 1501, 2001], dtype=np.int64)
    raw = requested_planar(ticks, case, .005)
    np.testing.assert_allclose(raw[:, 0], [3., 0., -2.5, 0., 3.])


def test_envelope_entry_creates_four_replicas_for_every_revised_case(tmp_path):
    import subprocess
    import sys
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, digest, encode
    parent = 'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'
    base_file = 'reports/flat_base_speed_20261020/directional200_preparation001/plan.json'
    base = read_plan(ROOT, base_file)
    plan = make_transient_plan(base, parent, digest(ROOT/parent), base_file, digest(ROOT/base_file))
    path = tmp_path/'plan.json'
    path.write_bytes(encode(plan))
    output = tmp_path/'evaluation'
    subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_flat_base.py'),
        '--plan', str(path), '--output', str(output), '--envelope', '--preflight-only'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    config = json.loads((output/'config/contract.json').read_text())
    cases = config['evaluation']['cases']
    assert config['target_num_envs'] == 4 * len(cases)
    assert [g['name'] for g in config['scene_groups']] == [c['name'] for c in cases]
    by_name = {c['name']: c for c in cases}
    assert by_name['backward_2p5']['skill']['command'][0] == -2.5
    assert {'yaw_start_stop_positive', 'yaw_start_stop_negative'} <= by_name.keys()


@pytest.mark.parametrize('height_variant', ['default', 'maximum'])
def test_rmcs_height_profile_protocol_composes_with_transient_targets(tmp_path, monkeypatch, height_variant):
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    import evaluate_flat_base
    import flat_default_height_profile
    import flat_maximum_height_profile
    from wheeled_tasks.chassis.flat_base_continuation import read_plan, digest, encode
    parent = 'runs/flat_base_20261020/pilot034_directional200_1346/model_001346.pt'
    base_file = 'reports/flat_base_speed_20261020/directional200_preparation001/plan.json'
    base = read_plan(ROOT, base_file)
    plan = make_transient_plan(base, parent, digest(ROOT/parent), base_file, digest(ROOT/base_file))
    path = tmp_path/'plan.json'
    path.write_bytes(encode(plan))
    # Isolate composition from the mutable deployment checkout. This fixture is
    # not evidence that today's RMCS YAML matches the sealed reference.
    contract = dict(schema=f'flat-rmcs-packaged-{height_variant}-height-v1',
        height_m=.305 if height_variant == 'default' else .42,
        command_height_bounds_m=[.23, .42])
    module = flat_default_height_profile if height_variant == 'default' else flat_maximum_height_profile
    monkeypatch.setattr(module, f'read_{height_variant}_height_contract', lambda *args: contract)
    output = tmp_path/'evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate_flat_base.py', '--plan', str(path),
        '--output', str(output), '--envelope', '--rmcs-command-reference', '--deployment-pd',
        f'--packaged-{height_variant}-height', '--preflight-only'])
    assert evaluate_flat_base.main() == 0
    config = json.loads((output/'config/contract.json').read_text())
    cases = config['evaluation']['cases']
    assert config['target_num_envs'] == 4 * len(cases)
    assert '-transient-targets-v1-material-' in config['evaluation']['protocol_id']
    assert f'rmcs-{height_variant}-height' in config['evaluation']['protocol_id']
    moving = [c for c in cases if any(c['command'][:2])]
    assert all(c['command'][2] == contract['height_m'] for c in moving)
