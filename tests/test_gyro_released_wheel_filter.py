"""Sensor scaling, scope, endpoints and evaluation contract for gyro release."""
import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_gyro_filter import GyroReleasedWheelFilter, gyro_filter_variants
from wheeled_algo.evaluation_quiet_output import QuietWheelOutputFilter


class Mean(nn.Module):
    def forward(self, obs):
        return obs[:, 16:22].clamp(-2., 2.)


def frames():
    obs = torch.randn(64, 35, generator=torch.Generator().manual_seed(82)) * .1
    obs[:, :3] = 0.
    obs[:, 4:7] = 0.
    obs[:, 9] = -1.
    return obs


def test_baseline_endpoints_scaled_axes_fixed_point_and_rng():
    mean, obs = Mean(), frames()
    model = GyroReleasedWheelFilter(mean, [.05, .2])
    rng = torch.get_rng_state()
    filtered = QuietWheelOutputFilter(mean, .25)(obs)
    assert torch.equal(model(obs), filtered)
    assert torch.equal(GyroReleasedWheelFilter(mean, None)(obs), filtered)
    obs[:, 6] = 9.  # Yaw does not release the extra balance filter.
    assert torch.equal(model(obs), filtered)
    for axis in (4, 5):
        for sign in (-1., 1.):
            obs[:, 4:6] = 0.
            obs[:, axis] = sign * .1  # .2rad/s after the declared gyro scaling.
            assert torch.equal(model(obs), mean(obs))
    obs[:, 4:6] = .04
    obs[:, 26:28] = mean(obs)[:, 4:]
    assert torch.equal(model(obs), mean(obs))
    assert not model.state_dict()
    assert torch.equal(rng, torch.get_rng_state())


def test_no_leg_or_nonquiet_change_and_continuous_transition():
    mean, obs = Mean(), frames()
    model = GyroReleasedWheelFilter(mean, [.05, .2])
    obs[:, 4] = torch.linspace(0., .15, len(obs))
    actual = model(obs)
    assert torch.equal(actual[:, :4], mean(obs)[:, :4])
    base, filtered = mean(obs)[:, 4:], QuietWheelOutputFilter(mean, .25)(obs)[:, 4:]
    assert ((actual[:, 4:] >= torch.minimum(base, filtered) - 1e-7)
            & (actual[:, 4:] <= torch.maximum(base, filtered) + 1e-7)).all()
    obs[:, 0] = .7
    assert torch.equal(model(obs), mean(obs))
    obs[:, 0] = 0.
    obs[:, 9] = 1.
    assert torch.equal(model(obs), mean(obs))


@pytest.mark.parametrize("band", [[], [.05], [0., .2], [.2, .05], [.1, .5], [True, .2],
                                  [float("nan"), .2], [.05, .2, .05, .2]])
def test_invalid_band_rejected(band):
    with pytest.raises(ValueError):
        gyro_filter_variants(band)
    with pytest.raises(ValueError):
        GyroReleasedWheelFilter(Mean(), band)


def test_double_filter_and_multiple_bands_on_one_actor_rejected():
    mean = Mean()
    mean.quiet_output_wheel_alpha = .25
    with pytest.raises(ValueError):
        GyroReleasedWheelFilter(mean, [.05, .2])
    with pytest.raises(ValueError):
        GyroReleasedWheelFilter(Mean(), [.05, .2, .03, .1])


@pytest.mark.parametrize("extra, text", [(["--domain-scale", ".35"], "nominal domain0"),
    (["--quiet-wheel-output-alphas", ".25"], "only one action-filter"),
    (["--quiet-wheel-gyro-release-band", ".05", ".2"], "only one action-filter")])
def test_invalid_grid_rejected_before_checkpoint_or_native(monkeypatch, capsys, tmp_path, extra, text):
    import sys
    from evaluate_robust_getup_batch import main
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(tmp_path / "missing.pt"),
        "--output", str(tmp_path / "not-created"), "--variants", "robust", "--replicas", "4",
        "--case-names", "stand", "stand_low", "stand_high", "--quiet-wheel-gyro-release-bands", ".05", ".2", *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert text in capsys.readouterr().err
    assert not (tmp_path / "not-created").exists()
