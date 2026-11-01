"""Snapshot lifetime, selected-lane reset and diagnostic reduction contracts."""
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.prone_motion_sampling import (
    PhysicalSamples, apply_course_cases, apply_fixed_cases, role_reductions, recovery_reductions)
from wheeled_tasks.chassis.tensor_cuda_graph import TensorCudaGraph


@pytest.mark.parametrize('steps', [2, 10])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA sample storage'))])
def test_packed_samples_preserve_every_value_stride_and_previous_interval(steps, device):
    recorder = PhysicalSamples(steps)
    previous = None
    for interval in range(3):
        recorder.begin()
        expected = []
        for i in range(steps):
            value = torch.arange(60, device=device).float().reshape(5, 12)+i+interval*100
            record = dict(position=value[:, ::2], velocity=value[:, 1::2].double(),
                          valid=value[:, :3] > 10, index=value[:, 0].long(), scalar=value.sum())
            expected.append({name: tensor.clone() for name, tensor in record.items()})
            recorder.append(record)
            value.fill_(-999.)
        actual = recorder.finish()
        assert list(actual) == list(expected[0])
        for name, tensor in actual.items():
            reference = torch.stack([row[name] for row in expected])
            torch.testing.assert_close(tensor, reference, rtol=0., atol=0.)
            assert tensor.is_contiguous() and tensor.stride() == reference.stride()
        if previous is not None:
            for name, value in previous[0].items():
                torch.testing.assert_close(value, previous[1][name], atol=0., rtol=0.)
        previous = actual, {name: value.clone() for name, value in actual.items()}


def test_sample_schema_interval_and_count_fail_closed():
    with pytest.raises(ValueError): PhysicalSamples(0)
    recorder = PhysicalSamples(2).begin()
    recorder.append(dict(x=torch.ones(3, 2)))
    with pytest.raises(RuntimeError, match='Incomplete'): recorder.finish()
    with pytest.raises(ValueError, match='schema'): recorder.append(dict(x=torch.ones(3, 3)))
    with pytest.raises(ValueError, match='schema'): recorder.append(dict(y=torch.ones(3, 2)))
    recorder.append(dict(x=torch.ones(3, 2)))
    with pytest.raises(RuntimeError, match='Too many'): recorder.append(dict(x=torch.ones(3, 2)))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fixed_role_reduction_matches_selected_means_with_missing_and_disabled_roles(dtype):
    generator = torch.Generator().manual_seed(261101)
    fields = torch.randn(37, 9, generator=generator, dtype=dtype)
    roles = torch.arange(37) % 8
    active = torch.arange(37) % 3 != 0
    fields[~active] = float('nan')
    outside = torch.rand(2, 37, 2, generator=generator) > .7
    means, spring, present = role_reductions(fields, roles, active, outside, torch.arange(11))
    for role in range(11):
        mask = (roles == role) & active
        assert bool(present[role]) == bool(mask.any())
        if mask.any():
            torch.testing.assert_close(means[role], fields[mask].mean(0), atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(spring[role], outside[:, mask].float().mean(), atol=1e-7, rtol=1e-6)
        else:
            assert torch.equal(means[role], torch.zeros(9, dtype=dtype))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA recovery reductions'))])
def test_recovery_reductions_preserve_all_groups_events_and_excluded_nan(device):
    n = 83
    idx = torch.arange(n, device=device)
    lane, family, leg = idx % 3 != 0, idx % 7, idx % 4
    first, enabled, terminated = idx % 5 == 0, idx % 2 == 0, idx % 7 == 0
    seconds = idx.float()*.01
    seconds[~first] = float('nan')
    groups = torch.tensor([(f,l) for f in range(7) for l in range(4)], device=device)
    args = lane, family, leg, first, enabled, terminated, seconds, groups
    actual = recovery_reductions(*args)
    if device == 'cuda':
        graph = TensorCudaGraph(recovery_reductions, *args)
        torch.testing.assert_close(graph(*args), actual, atol=0., rtol=0.)
    for i, (f,l) in enumerate(groups.tolist()):
        mask = lane & (family == f) & (leg == l)
        expected = torch.stack(((mask & enabled).float().sum(), (mask & first).float().sum(),
            (mask & terminated).float().sum(), torch.where(mask & first, seconds, 0.).sum()))
        torch.testing.assert_close(actual[i], expected, atol=1e-7, rtol=1e-6)


def reset_environment():
    return SimpleNamespace(device='cpu', pending_command=torch.full((7, 3), -5.),
        recovery_lane=torch.zeros(7, dtype=torch.bool), push_selected=torch.zeros(7, dtype=torch.bool),
        push_sign=torch.full((7,), 3.), sequence_time=torch.full((7, 4), float('inf')),
        sequence_command=torch.full((7, 4, 3), -9.), course_targets=torch.full((7, 3), -3.),
        course_moving_height=torch.ones(7, dtype=torch.bool), course_abrupt_brake=torch.ones(7, dtype=torch.bool),
        episode_limits=torch.full((7,), 7, dtype=torch.long), policy_dt=.01,
        evaluation_cases=None, plan=dict(evaluation=dict(seconds=12.), training_prone_seconds=70.,
                                         standing_seconds=30., motion_seconds=20.))


@pytest.mark.parametrize('evaluation', [False, True])
def test_batched_reset_matches_scalar_writes_for_unsorted_partial_lanes_and_sequences(evaluation):
    expected, actual = reset_environment(), reset_environment()
    if evaluation: expected.evaluation_cases = actual.evaluation_cases = [1]
    ids = torch.tensor([5, 1, 3])
    rows = [dict(pose_index=17, recovery=True, command=[0., 0., .305], role='prone',
                 target=[.3, 0., .305], push=True, push_sign=-1., sequence=[(.2, [.5, 0., .305])]),
            dict(pose_index=2, recovery=False, command=[0., 0., .32], role='stand',
                 target=[0., 0., .32], evaluation_seconds=15.),
            dict(pose_index=9, recovery=False, command=[-.4, .6, .42], role='arc_rl',
                 target=[-.4, .6, .42], moving_height=True, abrupt_brake=True)]
    for lane, row in zip(ids.tolist(), rows):
        expected.recovery_lane[lane] = row['recovery']
        expected.pending_command[lane] = expected.pending_command.new_tensor(row['command'])
        expected.push_selected[lane] = row.get('push', False)
        expected.push_sign[lane] = row.get('push_sign', 1.)
        for stage, (start, command) in enumerate(row.get('sequence', [])):
            expected.sequence_time[lane, stage] = start
            expected.sequence_command[lane, stage] = expected.pending_command.new_tensor(command)
        expected.course_targets[lane] = expected.course_targets.new_tensor(row['target'])
        expected.course_moving_height[lane] = row.get('moving_height', False)
        expected.course_abrupt_brake[lane] = row.get('abrupt_brake', False)
        duration = (row.get('evaluation_seconds', 12.) if evaluation else 70. if row['recovery']
                    else 30. if row['role'] == 'stand' else 20.)
        expected.episode_limits[lane] = round(duration/.01)
    torch.testing.assert_close(apply_fixed_cases(actual, ids, rows), torch.tensor([17, 2, 9]), atol=0., rtol=0.)
    apply_course_cases(actual, ids, rows)
    for name, value in vars(expected).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(actual, name), value, atol=0., rtol=0.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA grouped metrics')
def test_grouped_metric_graph_keeps_previous_results_and_eager_values():
    with torch.inference_mode():
        args = (torch.randn(256, 9, device='cuda'), torch.arange(256, device='cuda') % 11,
                torch.arange(256, device='cuda') % 3 != 0,
                torch.rand(2, 256, 2, device='cuda') > .4, torch.arange(11, device='cuda'))
        replay = TensorCudaGraph(role_reductions, *args)
        first = replay(*args)
        frozen = [value.clone() for value in first]
        args[0].add_(5.)
        for actual, expected in zip(replay(*args), role_reductions(*args)):
            torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        for actual, expected in zip(first, frozen):
            torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
