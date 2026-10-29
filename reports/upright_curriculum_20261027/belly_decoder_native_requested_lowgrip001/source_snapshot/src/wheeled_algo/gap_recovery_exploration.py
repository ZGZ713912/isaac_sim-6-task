"""Opt-in fallen exploration with the exact orbit-v2 conditioned likelihood."""
from copy import deepcopy
import hashlib
import math

import torch
from torch import nn

from .gap_entry_handoff import HandoffGapConditionedMean, handoff_entry_gate
from .robust_getup import tilt_gate


class GapRecoveryNoiseMean(HandoffGapConditionedMean):
    def __init__(self, mean, quiet_exploration_fraction=.35,
                 quiet_wheel_exploration_fraction=None, recovery_leg_fraction=1.):
        if (isinstance(recovery_leg_fraction, bool) or not math.isfinite(recovery_leg_fraction)
                or not .1 <= recovery_leg_fraction <= 1.):
            raise ValueError("Gap recovery leg noise fraction must be finite within [.1,1]")
        super().__init__(mean, quiet_exploration_fraction, quiet_wheel_exploration_fraction)
        self.recovery_leg_fraction = float(recovery_leg_fraction)

    def forward(self, obs):
        output = super().forward(obs)
        if self.recovery_leg_fraction == 1.:
            return output
        entry = handoff_entry_gate(obs) if self.mean.upright_entry_recovery else torch.zeros_like(obs[:, :1])
        fallen = tilt_gate(obs, self.mean.stance_aware_recovery, self.mean.recovery_shape_aware)
        factor = 1. - (1. - self.recovery_leg_fraction) * fallen * (1. - entry)
        log_std = torch.cat((output[:, 1, :4] + factor.log(), output[:, 1, 4:]), -1)
        return torch.stack((output[:, 0], log_std), -2)


class SampledGapPolicyEvaluation(nn.Module):
    """Private common random draws; this can never qualify a deterministic mean."""
    def __init__(self, mean, settings, learned_log_std, seed):
        super().__init__()
        if settings.get("gap_entry_handoff_version") != "orbit-v2":
            raise ValueError("Sample the authenticated orbit-v2 training distribution only")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Sampling seed must be an integer within [0,2**32)")
        if learned_log_std.shape != (6,) or not torch.isfinite(learned_log_std).all():
            raise ValueError("Sampling requires six finite checkpoint log-std values")
        self.sampler = GapRecoveryNoiseMean(deepcopy(mean), settings["quiet_exploration_fraction"],
            settings.get("quiet_wheel_exploration_fraction"),
            settings.get("gap_recovery_leg_exploration_fraction", 1.))
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
            raise ValueError("Sampled gap evaluation must retain batch size and device")
        mean, log_std = self.sampler(obs).unbind(-2)
        # The installed RSL5.5.1 runner uses this clamp before Normal.sample.
        std = log_std.clamp(math.log(.003), math.log(.8)).exp()
        epsilon = torch.randn(mean.shape, generator=self.generator, device=mean.device, dtype=mean.dtype)
        self.calls += 1
        return mean + epsilon * std

    def sampling_receipt(self):
        if self.generator is None:
            raise ValueError("No sampled policy frames were evaluated")
        state = self.generator.get_state().cpu().numpy().tobytes()
        return dict(seed=self.seed, calls=self.calls, samples_per_call=self.samples_per_call,
            generator_device=str(self.generator.device), generator_state_sha256=hashlib.sha256(state).hexdigest(),
            distribution="checkpoint_orbit_v2_conditioned_normal", std_range=[.003, .8],
            recovery_leg_fraction=self.sampler.recovery_leg_fraction,
            scope="fallen tilt authority times one minus upright-entry authority",
            wheel_sampling_changed=False, deterministic_mean_changed=False,
            eligible_for_mean_qualification=False)
