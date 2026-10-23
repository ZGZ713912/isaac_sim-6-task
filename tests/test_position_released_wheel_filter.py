"""Boundary and contract checks for the evaluation-only position filter."""
import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_position_filter import PositionReleasedWheelFilter, position_filter_variants
from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter


class Mean(nn.Module):
    quiet_velocity_feedback_mode = "standing_displacement"

    def forward(self, obs):
        return obs[:, 16:22].clamp(-2., 2.)


def frames():
    obs = torch.randn(128, 35, generator=torch.Generator().manual_seed(78)) * .1
    obs[:, :3] = 0.
    obs[:, 4:6] = 0.
    obs[:, 9] = -1.
    obs[:, 14:16] = 0.
    return obs


def test_baseline_endpoints_and_no_extra_state_or_rng():
    mean, obs = Mean(), frames()
    wrapper = PositionReleasedWheelFilter(mean, [.01, .025])
    rng = torch.get_rng_state()
    assert torch.equal(wrapper(obs), QuietWheelOutputFilter(mean, .25)(obs))
    assert not wrapper.state_dict()
    assert torch.equal(rng, torch.get_rng_state())
    for sign in (-1., 1.):
        obs[:, 14] = sign * .5  # .5/20=.025m, exact unfiltered endpoint.
        assert torch.equal(wrapper(obs), mean(obs))
    obs[:, 26:28] = mean(obs)[:, 4:]
    obs[:, 14] = .3
    assert torch.equal(wrapper(obs), mean(obs))


def test_nonquiet_and_legs_untouched_and_continuous_band():
    mean, obs = Mean(), frames()
    wrapper = PositionReleasedWheelFilter(mean, [.01, .025])
    obs[:, 14] = torch.linspace(0., 1., len(obs))
    assert torch.equal(wrapper(obs)[:, :4], mean(obs)[:, :4])
    base = mean(obs)[:, 4:]
    lo = obs[:, 26:28] + .25 * (base - obs[:, 26:28])
    output = wrapper(obs)[:, 4:]
    assert ((output >= torch.minimum(lo, base) - 1e-7)
            & (output <= torch.maximum(lo, base) + 1e-7)).all()
    obs[:, 0] = .5
    assert torch.equal(wrapper(obs), mean(obs))
    obs[:, 0] = 0.
    obs[:, 9] = 1.
    assert torch.equal(wrapper(obs), mean(obs))


@pytest.mark.parametrize("band", [[], [.01], [.025, .01], [.01, .06], [True, .02],
                                  [float("nan"), .02], [.01, .02, .01, .02]])
def test_invalid_bands_rejected(band):
    with pytest.raises(ValueError):
        position_filter_variants(band)


def test_mismatched_sensor_contract_and_double_filter_rejected():
    mean = Mean()
    mean.quiet_velocity_feedback_mode = "wheels"
    with pytest.raises(ValueError):
        PositionReleasedWheelFilter(mean, None)
    mean.quiet_velocity_feedback_mode = "standing_displacement"
    mean.quiet_output_wheel_alpha = .25
    with pytest.raises(ValueError):
        PositionReleasedWheelFilter(mean, [.01, .025])


def test_randomized_multi_variant_grid_rejected_before_checkpoint_or_native(monkeypatch, capsys, tmp_path):
    import sys
    from evaluate_robust_getup_batch import main
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(tmp_path / "missing.pt"),
        "--output", str(tmp_path / "not-created"), "--variants", "robust", "--replicas", "16",
        "--case-names", "stand", "stand_low", "stand_high", "--domain-scale", ".35",
        "--quiet-wheel-position-release-bands", ".01", ".025"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "nominal domain0" in capsys.readouterr().err
    assert not (tmp_path / "not-created").exists()
