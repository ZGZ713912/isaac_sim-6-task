"""Shared 50 Hz RMCS command math for flat training and deployment replay."""
from __future__ import annotations

import torch


class RmcsFlatCommandReference:
    """Batched command math; each lane advances once per policy tick."""

    def __init__(self, count, device, contract):
        self.cfg = contract
        self.reference = torch.zeros(count, 3, device=device)
        self.reference[:, 2] = contract["nominal_height_m"]
        self.heading = torch.zeros(count, device=device)
        self.heading_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.last_tick = torch.full((count,), -1, dtype=torch.long, device=device)
        self.requested = self.reference.clone()

    def reset(self, ids):
        self.reference[ids] = self.reference.new_tensor([0., 0., self.cfg["nominal_height_m"]])
        self.heading_valid[ids] = False
        self.last_tick[ids] = -1

    def update(self, requested, yaw, ticks):
        if (requested.shape != self.reference.shape or yaw.shape != ticks.shape
                or yaw.shape != self.heading.shape or not torch.isfinite(requested).all()
                or not torch.isfinite(yaw).all()):
            raise ValueError("Invalid flat RMCS command-reference inputs")
        ids = (ticks != self.last_tick).nonzero(as_tuple=False).flatten()
        cfg = self.cfg
        raw = requested[ids]
        if not len(ids):
            return self.reference
        if ((raw[:, 0].abs() > cfg["vx_max_m_s"] + 1e-6).any()
                or (raw[:, 1].abs() > cfg["yaw_rate_max_rad_s"] + 1e-6).any()
                or (raw[:, 2] < .23 - 1e-6).any() or (raw[:, 2] > .43 + 1e-6).any()):
            raise ValueError("Replay request exceeds the packaged flat actor envelope")
        straight = (raw[:, 0] != 0) & (raw[:, 1] == 0) & cfg["heading_hold_enabled"]
        capture = straight & ~self.heading_valid[ids]
        self.heading[ids] = torch.where(capture, yaw[ids], self.heading[ids])
        error = self.heading[ids] - yaw[ids]
        error = torch.atan2(error.sin(), error.cos())
        hold_yaw = (cfg["heading_hold_gain"] * error).clamp(
            -cfg["heading_hold_yaw_rate_max_rad_s"], cfg["heading_hold_yaw_rate_max_rad_s"])
        desired_yaw = torch.where(straight, hold_yaw, raw[:, 1]).clamp(
            -cfg["yaw_rate_max_rad_s"], cfg["yaw_rate_max_rad_s"])
        self.heading_valid[ids] = straight
        vx = raw[:, 0]
        demand = torch.maximum((vx - cfg["wheel_track_m"] * desired_yaw / 2).abs(),
                               (vx + cfg["wheel_track_m"] * desired_yaw / 2).abs())
        scale = (cfg["wheel_speed_limit_rad_s"] * cfg["wheel_radius_m"] / demand.clamp_min(1e-12)).clamp_max(1.)
        vx, desired_yaw = vx * scale, desired_yaw * scale
        vx = torch.where((vx * desired_yaw).abs() > cfg["lateral_acceleration_m_s2"],
            vx.sign() * cfg["lateral_acceleration_m_s2"] / desired_yaw.abs().clamp_min(1e-12), vx)
        max_height_step = cfg["height_reference_rate_max_m_s"] / cfg["policy_hz"]
        height = self.reference[ids, 2] + (raw[:, 2] - self.reference[ids, 2]).clamp(
            -max_height_step, max_height_step)
        self.reference[ids] = torch.stack((vx, desired_yaw, height), -1)
        self.requested[ids] = raw
        self.last_tick[ids] = ticks[ids]
        return self.reference


def install_training_reference(env, settings):
    """Mix direct RMCS commands with the existing ramped training lanes.

    Sampling and profile timing remain owned by SkillCommands. The raw target
    survives replacement of the visible commands, so height limiting cannot
    recursively chase its own previous reference. Installation does not change
    physics, observations, rewards, the population, or the sampled amplitudes.
    """
    if (not env.cfg.get("flat_base_continuation") or env.cfg.get("evaluation_exact_cases")
            or env.policy_dt != .02 or env.cfg["physics_dt"] != .005
            or not env.cfg.get("command_slew") or env.skills is None
            or settings["moving_direct_fraction"] != .5 or settings["standing_direct_fraction"] != 1.):
        raise ValueError("Training RMCS reference requires the declared mixed 50/200 Hz flat pilot")
    allowed = {"stand", "forward", "backward", "rotate", "curve", "start_stop"}
    direct = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    population = {}
    for ids, spec in env.skills.batches:
        if (spec.get("kind") not in allowed or any(key in spec for key in
                ("height_motion", "height_transition_seconds", "command_sequence", "height_pulse"))):
            raise ValueError("Training RMCS reference received an undeclared command profile")
        selected = ids if spec["kind"] == "stand" else ids[::2]
        direct[selected] = True
        population[spec["group_name"]] = dict(total=len(ids), direct=len(selected), ramped=len(ids)-len(selected))
    reference = RmcsFlatCommandReference(env.num_envs, env.device, settings["contract"])
    original_reset, original_observations = env.reset, env.get_observations

    def reset(ids):
        original_reset(ids)
        reference.reset(ids)

    def observations():
        requested = torch.cat((env.command_target, env.skills.command_base[:, 2:3]), -1)
        commands = reference.update(requested, env.skills.yaw(), env.episode_length_buf)
        env.commands[direct] = commands[direct]
        env.command_acceleration[direct] = 0.
        if env.references is not None:
            env.raw_height[direct] = env.commands[direct, 2]
            env.update_command_reference()
        return original_observations()

    env.reset, env.get_observations = reset, observations
    env.rmcs_training_reference, env.rmcs_direct_lanes = reference, direct
    return dict(schema="flat-mixed-rmcs-command-training-v1", policy_hz=50, physics_hz=200,
        contract=settings["contract"], population=population,
        direct_lanes=int(direct.sum()), ramped_lanes=int((~direct).sum()),
        sampling="all standing lanes; alternating members of every moving group; no RNG consumed")
