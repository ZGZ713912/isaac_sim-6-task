"""Explicit tracking-stage costs and safety, retaining the original009 actor."""
import torch


def enabled(plan):
    return plan.get("repair", {}).get("version") == "standing-retention-v2"


def costs(position_error, height, commands, plan):
    zero = torch.zeros_like(height)
    if not enabled(plan):
        return dict(position=zero, height=zero)
    cfg = plan["repair"]
    quiet = (commands[:, :2].abs() < .01).all(-1)
    radius = position_error.norm(dim=-1)
    position = cfg["position_weight"] * ((radius-cfg["position_deadband_m"]).clamp_min(0.)
        / cfg["position_width_m"]).square().clamp_max(cfg["position_cost_cap"])
    height_cost = cfg["height_weight"] * torch.log1p(((height-commands[:, 2])/cfg["height_scale_m"]).square())
    return dict(position=position*quiet, height=height_cost)


def height_failure(height, command_height, learning_height, plan):
    target_error = (height-command_height).abs() > .025
    if not enabled(plan):
        return target_error
    cfg = plan["repair"]
    # Requested transition errors are learning signals, not physical falls.
    # Final success still requires the unchanged full-trace height thresholds.
    unsafe = (height < cfg["minimum_safe_height_m"]) | (height > cfg["maximum_safe_height_m"])
    return unsafe | (target_error & ~learning_height)


def critic_anchor(critic, error_xy, columns):
    """Use only the two formerly-zero privileged wheel-position entries."""
    result = critic.clone()
    result[:, list(columns)] = (error_xy*10.).clamp(-10., 10.)
    return result
