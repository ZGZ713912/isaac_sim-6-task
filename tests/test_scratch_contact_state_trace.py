"""Packed contact buffers must not turn missing/overflowed data into support."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("scratch_contact_state",
    Path(__file__).resolve().parents[1] / "scripts/capture_scratch_contact_state.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def buffers():
    force = torch.tensor([[99.], [10.], [20.], [30.], [88.]])
    normal = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    distance = torch.arange(5.)[:, None] / 1000.
    return force, normal, distance


def test_unsigned_sdk_contact_pairs_preserve_wheel_order_and_empty_pair_sentinel():
    counts = torch.tensor([[99], [2], [88], [0]], dtype=torch.uint32)
    starts = torch.tensor([[99], [1], [88], [4294967295]], dtype=torch.uint32)
    selected_counts, selected_starts = observer.select_wheel_contact_pairs(counts, starts,
        torch.tensor([[1, 3]]))
    assert selected_counts.dtype == selected_starts.dtype == torch.int64
    assert selected_counts.tolist() == [[[2], [0]]]
    assert selected_starts.tolist() == [[[1], [4294967295]]]
    result = observer.reduce_contact_points(*buffers(), selected_counts, selected_starts, maximum=3)
    assert result['force'].tolist() == [[[0., 0., 30.], [0., 0., 0.]]]
    assert not result['overflow'].any()


def test_packed_counts_offsets_filters_and_empty_pairs_are_respected():
    counts = torch.tensor([[[2, 1], [0, 0]]])
    starts = torch.tensor([[[1, 3], [123456, -99]]])
    result = observer.reduce_contact_points(*buffers(), counts, starts, maximum=3)
    assert result['force'].tolist() == [[[0., 30., 30.], [0., 0., 0.]]]
    assert result['point_count'].tolist() == [[3, 0]]
    assert result['minimum_separation'][0, 0] == pytest.approx(.001)
    assert torch.isposinf(result['minimum_separation'][0, 1])
    assert not result['overflow'].any()


def test_uninitialized_padding_cannot_contaminate_valid_contact_forces():
    force, normal, distance = buffers()
    force[[0, 4]], normal[[0, 4]], distance[[0, 4]] = torch.nan, torch.nan, torch.nan
    result = observer.reduce_contact_points(force, normal, distance,
        torch.tensor([[[2], [0]]]), torch.tensor([[[1], [123456]]]), maximum=4)
    assert result['force'].tolist() == [[[0., 0., 30.], [0., 0., 0.]]]
    assert not result['overflow'].any()
    assert torch.isposinf(result['minimum_separation'][0, 1])


@pytest.mark.parametrize('counts,starts,maximum', [
    ([[[2], [0]]], [[[4], [0]]], 3),
    ([[[1], [0]]], [[[-1], [0]]], 3),
    ([[[4], [0]]], [[[0], [0]]], 3),
    ([[[-1], [0]]], [[[0], [0]]], 3),
])
def test_invalid_or_truncated_contact_data_is_explicit(counts, starts, maximum):
    result = observer.reduce_contact_points(*buffers(), torch.tensor(counts), torch.tensor(starts), maximum=maximum)
    assert result['overflow'].tolist() == [[True, False]]


def test_failed_native_evaluation_restores_the_existing_hooks(monkeypatch):
    native_gap, native_step = lambda: None, lambda action: None
    stage = SimpleNamespace(GetPrimAtPath=lambda _: SimpleNamespace(IsValid=lambda: False))
    env = SimpleNamespace(num_envs=21, dt=.001, decimation=5, sim=SimpleNamespace(stage=stage),
        closure_gap=native_gap, step=native_step)
    monkeypatch.setattr(observer, 'contact_view_bindings', lambda _: [])

    def fail(*args):
        assert env.step is not native_step and env.closure_gap is not native_gap
        raise RuntimeError('deliberate evaluation failure')

    with pytest.raises(RuntimeError, match='deliberate evaluation failure'):
        observer.evaluate_with_contact_state(env, None, None, None, 0, evaluate=fail, digest=None)
    assert env.step is native_step and env.closure_gap is native_gap


@pytest.mark.parametrize('fault', [None, 'sensor_order', 'filter', 'duplicate_lane'])
def test_native_contact_binding_must_match_the_physical_environment(fault):
    paths = ['/env_0/Robot/base', '/env_0/Robot/left', '/env_0/Robot/right']
    sensor_paths = paths[::-1] if fault == 'sensor_order' else paths
    view = SimpleNamespace(sensor_paths=sensor_paths, sensor_count=3, filter_count=1,
        max_contact_data_count=192,
        filter_paths=[['/env_1/Terrain/surface_0/geometry/mesh' if fault == 'filter'
                       else '/env_0/Terrain/surface_0/geometry/mesh']]*3)
    env = SimpleNamespace(num_envs=1, env_paths=['/env_0'], surfaces=[[0]],
        robot=SimpleNamespace(body_names=['base', 'left', 'right']),
        contact_views=[(torch.tensor([0]), view)]*(2 if fault == 'duplicate_lane' else 1))
    if fault is None:
        assert observer.contact_view_bindings(env)[0]['exact_sensor_order']
    else:
        with pytest.raises(ValueError):
            observer.contact_view_bindings(env)
