"""Additional flat replay of RMCS's policy-rate command reference.

This covers ordinary, already handed-over RL. It does not execute the recovery
controller, upstream keyboard integration, CAN transport, or physical motors.
The existing acceleration-limited evaluation remains a separate protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


PROTOCOL_FIXED = "flat-base-paired107-rmcs-reference-fixed-v1"
PROTOCOL_ENVELOPE = "flat-base-paired107-rmcs-reference-envelope-v1"


def read_reference_contract(rmcs):
    """Bind the transcription to the exact source inspected for this replay."""
    root = Path(rmcs) / "rmcs_ws/src"
    sources = {
        "rmcs_rl/src/wheel_leg/observation.cpp": "7e98fc95cd4b198bc3b02790e0c1736267eb448ceb3efd1b94180b01d771376c",
        "rmcs_rl/src/wheel_leg/policy.hpp": "debb9dda28b911271f4967a7ccab1f7d21cf041c6ed9ab11d0017981624ad5d5",
        "rmcs_bringup/config/wheel-leg-infantry-rl.yaml": "7d893f68040cb1e352c67e39af4a86e72b4b294d0981229fb5c1102361dcd063",
    }
    for relative, expected in sources.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("RMCS command-reference source changed: " + relative)
    return dict(schema="flat-rmcs-command-reference-v1", source_sha256=sources,
        policy_hz=50, physics_hz=1000, nominal_height_m=.305,
        height_command_is_reference=True, height_reference_rate_max_m_s=2/3,
        wheel_radius_m=.06, wheel_track_m=.4373, wheel_speed_limit_rad_s=90.,
        lateral_acceleration_m_s2=2.06, heading_hold_enabled=True, heading_hold_gain=5.,
        vx_max_m_s=3., yaw_rate_max_rad_s=4., heading_hold_yaw_rate_max_rad_s=4.,
        envelope_source="export_flat_base.py packaged YAML caps: vx=3, yaw/heading/spin=4",
        planar_slew="none; next policy tick", takeover_blending=False, spinning=False,
        scope="ordinary flat RL command assembly after takeover; native simulation only")


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


def install_reference(env, contract):
    """Replace command assembly before observation, leaving physics untouched."""
    from wheeled_tasks.chassis.skill_commands import profile_command

    if (not env.cfg.get("flat_base_continuation") or not env.cfg.get("evaluation_exact_cases")
            or env.cfg["policy_dt"] != .02 or env.cfg["physics_dt"] != .001
            or any(c.get("terrain") != "flat" or c.get("task") != "survive" for c in env.cases.values())):
        raise ValueError("RMCS reference replay requires the separate 50/1000 Hz flat suite")
    reference = RmcsFlatCommandReference(env.num_envs, env.device, contract)
    batches = [(torch.tensor([i for i, n in enumerate(env.scene_groups) if n == name], device=env.device), case)
               for name, case in env.cases.items()]
    original_reset, original_observations = env.reset, env.get_observations

    def reset(ids):
        original_reset(ids)
        reference.reset(ids)

    def observations():
        requested = torch.empty_like(env.commands)
        elapsed = env.episode_length_buf * env.policy_dt
        for ids, case in batches:
            requested[ids] = profile_command(elapsed[ids], case["command"], case.get("skill", {}))
        command = reference.update(requested, env.skills.yaw(), env.episode_length_buf)
        env.commands.copy_(command)
        # Flat context hides acceleration-dependent task references. Keep the
        # shared height diagnostics consistent with the command in this input.
        env.command_acceleration.zero_()
        if env.references is not None:
            env.raw_height.copy_(env.commands[:, 2])
            env.update_command_reference()
        return original_observations()

    env.reset, env.get_observations = reset, observations
    env.rmcs_command_reference = reference
    return reference
