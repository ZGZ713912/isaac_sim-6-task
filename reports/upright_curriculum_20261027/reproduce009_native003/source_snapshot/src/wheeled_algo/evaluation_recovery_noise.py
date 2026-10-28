"""Explicit sampled-policy exposure diagnostic, excluded from mean acceptance."""
from copy import deepcopy
import hashlib
import math

import torch
from torch import nn

from .recovery_exploration import RecoveryNoiseMean


def recovery_noise_variants(values):
    if (not 1 <= len(values) <= 2 or len(set(values)) != len(values)
            or any(isinstance(value, bool) or not math.isfinite(value)
                   or not .1 <= value < 1. for value in values)):
        raise ValueError("Recovery noise grid requires1..2 distinct alternatives within[.1,1); original1 is included")
    return {"robust": 1., **{f"recovery_leg_noise_{value:.17g}": value for value in values}}


class SampledRecoveryEvaluation(nn.Module):
    def __init__(self, mean, settings, learned_log_std, fraction, seed):
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Sampling seed must be an integer within[0,2**32)")
        if learned_log_std.shape != (6,) or not torch.isfinite(learned_log_std).all():
            raise ValueError("Sampling requires the six finite checkpoint log-std values")
        self.sampler = RecoveryNoiseMean(deepcopy(mean), settings["quiet_exploration_fraction"],
            settings["quiet_wheel_exploration_fraction"], fraction)
        with torch.no_grad():
            self.sampler.log_std.copy_(learned_log_std)
        self.requires_grad_(False)
        self.seed = seed
        self.generator = None
        self.calls = 0
        self.samples_per_call = None

    def forward(self, obs):
        if self.generator is None:
            self.generator = torch.Generator(device=obs.device).manual_seed(self.seed)
            self.samples_per_call = len(obs)
        if len(obs) != self.samples_per_call or obs.device != self.generator.device:
            raise ValueError("Sampled diagnostic must keep its batch size and device fixed")
        output = self.sampler(obs)
        mean, log_std = output.unbind(-2)
        # Matches the inspected native HeteroscedasticGaussianDistribution.
        std = log_std.clamp(math.log(.003), math.log(.8)).exp()
        epsilon = torch.randn(mean.shape, generator=self.generator, device=mean.device, dtype=mean.dtype)
        self.calls += 1
        return mean + epsilon * std

    def sampling_receipt(self):
        if self.generator is None:
            raise ValueError("No sampled diagnostic frames were evaluated")
        state = self.generator.get_state().cpu().numpy().tobytes()
        return dict(seed=self.seed, calls=self.calls, samples_per_call=self.samples_per_call,
            generator_device=str(self.generator.device), generator_state_sha256=hashlib.sha256(state).hexdigest(),
            distribution="checkpoint_sensor_conditioned_normal", std_range=[.003,.8],
            recovery_leg_fraction=self.sampler.recovery_leg_fraction, wheel_sampling_changed=False,
            deployed_mean_changed=False, eligible_for_mean_qualification=False)
