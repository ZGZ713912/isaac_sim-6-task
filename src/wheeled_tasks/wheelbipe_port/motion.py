"""Deployable policy history and signed motion objectives, independent of PhysX.

True velocity is an argument to reward/logging functions only. Policy history
accepts the existing clipped/scaled sensor frame; it cannot query the plant.
"""
from __future__ import annotations

import math

import torch

from .diagnostics import grouped_sum


def validate_motion(spec):
    c = spec["motion"]
    if (c["frame_dim"] != 35 or type(c["history_frames"]) is not int
            or c["history_frames"] != 5 or c["history_order"] != "newest_first"
            or spec["policy_hz"] != 50):
        raise ValueError("Motion history requires five 35D frames at 50 Hz, newest first")
    for key in ("tracking_sigma_m_s", "error_scale_m_s", "reset_grace_s"):
        if not math.isfinite(c[key]) or c[key] <= 0.:
            raise ValueError("Motion scales must be positive finite values: "+key)
    if (spec["reward_overrides"]["track_lin_vel_xy"] != 1.
            or spec["reward_overrides"]["track_lin_vel_xy_square"] != -1.
            or spec["parameter_overrides"]["stand_still_deadzone_threshold"] != 1.e-6):
        raise ValueError("Motion objective replaces legacy tracking and retains nonzero low-speed commands")
    if spec["motion_reward_weights"] != dict(motion_tracking=2., motion_error=-1.):
        raise ValueError("Motion reward weights differ from the explicit V10 objective")


def observation_contract(spec=None):
    c = (spec or {}).get("motion", {})
    frames = c.get("history_frames", 1)
    return dict(frame_dim=35, history_frames=frames, history_order=c.get("history_order", "newest_first"),
        actor_dim=35*frames, critic_dim=78, action_dim=6,
        frame_period_s=.02, oldest_frame_age_s=(frames-1)*.02,
        reset_fill="repeat first sensor frame", privileged_actor_inputs=False)


class PolicyHistory:
    """One append per policy tick; selected resets never retain prior episodes."""
    def __init__(self, count, frames, frame_dim, device):
        self.data = torch.zeros(count, frames, frame_dim, device=device)
        self.ready = torch.zeros(count, dtype=torch.bool, device=device)
        self.tick = None

    def reset(self, ids):
        self.ready[ids] = False

    def update(self, frame, tick):
        if frame.shape != self.data[:, 0].shape:
            raise ValueError("Policy history sensor frame shape differs from its contract")
        if tick != self.tick:
            self.data[:, 1:] = self.data[:, :-1].clone()
            self.tick = tick
        self.data[:, 0] = frame
        self.data[:] = torch.where(self.ready[:, None, None], self.data, frame[:, None])
        self.ready[:] = True
        # Do not expose a view that the next observation call will mutate.
        return self.data.flatten(1).clone()


def motion_terms(command, speed, eligible, cfg):
    """Reward densities, symmetric in signed command/speed, with no error clamp.

    The smooth L1-like cost retains a nonzero slope far from the target. Scales
    are explicit speeds in m/s, not legacy multipliers called 'sigma'.
    """
    error = speed-command
    return dict(motion_tracking=eligible*torch.exp(-(error/cfg["tracking_sigma_m_s"]).square()),
        motion_error=eligible*(torch.sqrt(1.+(error/cfg["error_scale_m_s"]).square())-1.))


def replace_motion_terms(terms, command, speed, eligible, cfg):
    # Jump/drop windows and reset grace retain the V9 horizontal objective.
    for name in ("track_lin_vel_xy", "track_lin_vel_xy_square"):
        terms[name] = terms[name]*(~eligible)
    terms.update(motion_terms(command,speed,eligible,cfg))
    return terms


def motion_statistics(command, speed, height, eligible, wheel_target, wheel_speed):
    """Before-reset direction × height diagnostics; zero counts mean untested."""
    direction = torch.where(command > 1.e-6, 1, torch.where(command < -1.e-6, 2, 0))
    band = (height >= .27).long()+(height >= .37).long()
    group = direction*3+band
    error = speed-command
    w = eligible.float()
    values = torch.stack((w, w*command, w*speed, w*error, w*error.abs(),
        w*(command*speed < 0.), w*wheel_target, w*wheel_speed), -1)
    sums = grouped_sum(group, values, 9)
    mean = sums[:, 1:]/sums[:, :1].clamp_min(1.)
    result = {}
    keys = ("command_mean_m_s", "actual_mean_m_s", "signed_error_mean_m_s", "error_abs_mean_m_s",
            "wrong_direction_fraction", "wheel_target_proxy_mean_m_s", "wheel_speed_proxy_mean_m_s")
    for i, direction in enumerate(("stand", "forward", "reverse")):
        for j, band in enumerate(("low", "mid", "high")):
            prefix = "Motion/"+direction+"/"+band+"/"
            row = i*3+j
            result[prefix+"sample_envs"] = sums[row, 0]
            # RSL averages scalar logs over a rollout. These sufficient sums
            # divided by sample_envs retain correct weighting across that mean.
            result[prefix+"signed_error_sum_m_s"] = sums[row, 3]
            result[prefix+"error_abs_sum_m_s"] = sums[row, 4]
            result.update({prefix+key: mean[row, k] for k, key in enumerate(keys)})
    return result


def actor_from_state_dict(state, actor_dim, device="cpu"):
    """Refuse incompatible history ABIs instead of padding an older actor."""
    if any("obs_normalizer" in key for key in state):
        raise ValueError("This actor contract requires the original non-normalized sensor frames")
    expected = {f"actor.{i*2}.{suffix}" for i in range(4) for suffix in ("weight","bias")}
    if {key for key in state if key.startswith("actor.")} != expected:
        raise ValueError("Actor must contain exactly four Linear layers with ELU activations")
    layers = []
    for i, (ni, no) in enumerate(((actor_dim, 256), (256, 128), (128, 64), (64, 6))):
        weight, bias = state[f"actor.{i*2}.weight"], state[f"actor.{i*2}.bias"]
        if weight.shape != (no, ni) or bias.shape != (no,):
            raise ValueError(f"Actor layer {i} differs from {actor_dim}D observation contract")
        layer = torch.nn.Linear(ni, no, device=device)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        layers.append(layer)
        if i < 3:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).eval()
