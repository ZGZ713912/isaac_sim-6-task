"""Preserve lane semantics without data-dependent indices or scalar readback."""
import pytest
import torch

from wheeled_tasks.chassis.masked_tensor import copy_masked_rows_, masked_mean, masked_max


@pytest.mark.parametrize('shape', [(8,), (8, 2), (8, 2, 3)])
@pytest.mark.parametrize('selection', ['none', 'some', 'all'])
def test_masked_state_and_reductions_match_selected_rows(shape, selection):
    mask = torch.arange(8).remainder(2).bool()
    if selection == 'none':
        mask.zero_()
    elif selection == 'all':
        mask.fill_(True)
    value = -torch.arange(1, torch.tensor(shape).prod().item()+1).float().reshape(shape)
    original = torch.full(shape, 7.)
    expected = original.clone()
    expected[mask] = value[mask]
    before = torch.get_rng_state().clone()
    address = original.data_ptr()
    assert copy_masked_rows_(original, mask, value) is original
    assert original.data_ptr() == address
    torch.testing.assert_close(original, expected, atol=0., rtol=0.)
    torch.testing.assert_close(masked_mean(value, mask), value[mask].mean() if mask.any() else torch.tensor(0.))
    torch.testing.assert_close(masked_max(value, mask), value[mask].max() if mask.any() else torch.tensor(0.))
    assert torch.equal(before, torch.get_rng_state())


def test_nonfinite_unselected_lanes_do_not_leak_into_state_or_logs():
    mask = torch.tensor([False, True, False, True])
    source = torch.tensor([[float('nan'), float('inf')], [-4., -2.],
                           [float('-inf'), float('nan')], [-8., -6.]])
    target = torch.ones_like(source)
    copy_masked_rows_(target, mask, source)
    assert torch.equal(target[~mask], torch.ones(2, 2))
    assert masked_mean(source, mask) == -5.
    assert masked_max(source, mask) == -2.


def test_sampling_helpers_do_not_materialize_dynamic_indices():
    mask = torch.tensor([False, True, True, False])
    source, target = torch.ones(4, 2), torch.zeros(4, 2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        copy_masked_rows_(target, mask, source)
        masked_mean(source, mask)
        masked_max(source, mask)
    names = {item.key for item in profile.key_averages()}
    assert not names & {'aten::nonzero', 'aten::item', 'aten::_local_scalar_dense'}
