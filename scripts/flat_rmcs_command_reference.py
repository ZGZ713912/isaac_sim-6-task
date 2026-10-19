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


from wheeled_tasks.chassis.flat_command_reference import RmcsFlatCommandReference


def install_reference(env, contract, *, diagnostic_training_clock=False):
    """Replace command assembly before observation, leaving physics untouched."""
    from wheeled_tasks.chassis.skill_commands import profile_command

    if diagnostic_training_clock and not env.cfg.get("evaluation", {}).get("diagnostic_only"):
        raise ValueError("The 200 Hz reference comparison requires an explicit diagnostic-only contract")
    expected_dt = .005 if diagnostic_training_clock else .001
    if (not env.cfg.get("flat_base_continuation") or not env.cfg.get("evaluation_exact_cases")
            or env.cfg["policy_dt"] != .02 or env.cfg["physics_dt"] != expected_dt
            or any(c.get("terrain") != "flat" or c.get("task") != "survive" for c in env.cases.values())):
        raise ValueError("RMCS reference replay has an undeclared policy/feedback clock or non-flat case")
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
