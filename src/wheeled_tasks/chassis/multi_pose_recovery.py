"""Reset-only curriculum and continuous, shared flip-to-rise task state.

No motor targets, physical state writes, teacher or checkpoint loading live here.
Family/leg labels are privileged diagnostics, never additional actor inputs.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random

import torch

FAMILIES = ("belly_down", "left_side", "right_side", "roof_down", "front_down", "rear_down", "diagonal")
LEG_CLASSES = ("front_front", "rear_rear", "front_rear", "rear_front")


def validate(config):
    c = deepcopy(config)
    if c.get("schema") != "wheel-leg-multi-pose-recovery-v1":
        raise ValueError("Need the explicit multi-pose recovery contract")
    if c.get("families") != list(FAMILIES) or c.get("leg_classes") != list(LEG_CLASSES):
        raise ValueError("Recovery must cover all declared body and independent leg classes")
    if c.get("motor_owner") != "one_continuous_six_axis_rl" or c.get("state_writes") != "episode_reset_only":
        raise ValueError("Flip-to-rise must retain continuous RL and reset-only physical writes")
    if c.get("leg_speed_limit_rad_s") is not None:
        raise ValueError("The requested recovery uses structural protection, not a uniform leg speed cap")
    for key in ("time_cost_per_second", "failure_cost", "flip_event_bonus", "potential_gain",
                "upright_max_body_rate_rad_s", "upright_max_leg_speed_rad_s", "upright_hold_seconds"):
        if type(c.get(key)) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("Invalid recovery parameter: "+key)
    if not 0 < c["upright_tilt_deg"] < 60 or not 0 < c["upright_max_encoder_error_m"] < .1:
        raise ValueError("Invalid shared suffix entry geometry")
    if c.get("curriculum_mode") not in ("staged", "all"):
        raise ValueError("Recovery curriculum must be staged or all")
    stages = c["curriculum"]
    if not stages or [s["start_update"] for s in stages] != sorted(set(s["start_update"] for s in stages)):
        raise ValueError("Recovery curriculum stages must be ordered and distinct")
    if stages[0]["start_update"] != 0:
        raise ValueError("Recovery curriculum must start at update zero")
    for stage in stages:
        weights = stage["family_weights"]
        if (set(weights) != set(FAMILIES) or any(not math.isfinite(v) or v <= 0 for v in weights.values())
                or not math.isclose(sum(weights.values()), 1., abs_tol=1e-8)):
            raise ValueError("Every curriculum stage must retain every fallen family")
    return c


def leg_class(orbits_deg):
    """Body-frame wheel-center fore/aft sign; neutral boundaries retain continuity."""
    if len(orbits_deg) != 2 or not all(math.isfinite(v) for v in orbits_deg):
        raise ValueError("Need two finite physical orbital angles")
    front = [math.sin(math.radians(v)) >= 0 for v in orbits_deg]
    return (0 if front == [True, True] else 1 if front == [False, False]
            else 2 if front == [True, False] else 3)


def validate_bank(bank, config, manifest_sha256, spec_sha256):
    validate(config)
    if (bank.get("schema") != "wheel-leg-unified-reset-bank-v1"
            or bank.get("asset_manifest_sha256") != manifest_sha256
            or bank.get("model_spec_sha256") != spec_sha256):
        raise ValueError("Multi-pose bank is not bound to the current full vehicle")
    declaration = bank.get("multi_pose_recovery", {})
    if (declaration.get("schema") != "wheel-leg-multi-pose-bank-v1"
            or declaration.get("self_collision_response") is not False):
        raise ValueError("Missing explicit reset admission and self-collision scope")
    expected = set(bank["tree_joint_names"])
    pools = {(f, l): [] for f in FAMILIES for l in LEG_CLASSES}
    for i, pose in enumerate(bank["poses"]):
        info = pose.get("recovery")
        if not info:
            continue
        key = (info["family"], info["leg_class"])
        if key not in pools or info["leg_class"] != LEG_CLASSES[leg_class(info["orbital_pair_deg"])]:
            raise ValueError("Recovery bank label disagrees with physical leg angles")
        values = pose["full_joint_positions"]
        velocity = pose.get("full_joint_velocities", dict.fromkeys(expected, 0.))
        if (set(values) != expected or set(velocity) != expected
                or not all(math.isfinite(v) for v in (*values.values(), *velocity.values()))
                or not pose.get("admitted_as_training_reset") or not info.get("cad_surface_screen_passed")
                or pose["gap_m"] > 1e-7 or pose["min_collision_clearance_m"] < 0
                or not all(40. <= v <= 110. for v in pose["knee_deg"])):
            raise ValueError("Recovery reset lacks complete feasible, CAD-screened state")
        root = pose["root_position_m"]+pose["root_quaternion_xyzw"]
        if (len(root) != 7 or not all(math.isfinite(v) for v in root)
                or not math.isclose(sum(v*v for v in root[3:]), 1., abs_tol=1e-6)):
            raise ValueError("Invalid reset root transform")
        pools[key].append(i)
    if any(not pool for pool in pools.values()):
        raise ValueError("Missing body-family x independent-leg reset cell")
    return pools


class ResetSampler:
    """Seeded weighted family sampling; cycle all four leg classes per family."""
    def __init__(self, bank, config, seed):
        self.config = validate(config)
        self.bank = bank
        self.rng = random.Random(seed)
        self.class_cursor = dict.fromkeys(FAMILIES, 0)
        self.pools = {(f, l): [] for f in FAMILIES for l in LEG_CLASSES}
        for i, p in enumerate(bank["poses"]):
            if "recovery" in p and p["recovery"]["split"] == "train":
                r = p["recovery"]
                self.pools[(r["family"], r["leg_class"])].append(i)
        if any(not p for p in self.pools.values()):
            raise ValueError("Cannot silently replace a missing recovery cell with standard prone")

    def sample(self, completed):
        stages = self.config["curriculum"]
        stage = stages[-1] if self.config["curriculum_mode"] == "all" else next(
            s for s in reversed(stages) if completed >= s["start_update"])
        weights = stage["family_weights"]
        family = self.rng.choices(FAMILIES, [weights[f] for f in FAMILIES])[0]
        cls = LEG_CLASSES[self.class_cursor[family] % len(LEG_CLASSES)]
        self.class_cursor[family] += 1
        return self.rng.choice(self.pools[(family, cls)])


class RecoveryStages:
    """One-shot shared-suffix entry, actual active time and dynamic handoff state."""
    def __init__(self, count, device, config, policy_dt):
        self.config, self.dt = validate(config), policy_dt
        self.enabled_at = torch.full((count,), float("nan"), device=device)
        self.upright_at = self.enabled_at.clone()
        self.upright_hold = torch.zeros(count, device=device)
        self.entered = torch.zeros(count, dtype=torch.bool, device=device)
        self.initial_family = torch.full((count,), -1, dtype=torch.long, device=device)
        self.initial_leg_class = self.initial_family.clone()
        self.entry_leg_class = self.initial_family.clone()
        self.entry_q = torch.zeros(count, 4, device=device)
        self.entry_dq = self.entry_q.clone()
        self.entry_omega = torch.zeros(count, 3, device=device)

    def reset(self, ids, families, leg_classes):
        self.enabled_at[ids] = self.upright_at[ids] = float("nan")
        self.upright_hold[ids] = 0.
        self.entered[ids] = False
        self.initial_family[ids] = families
        self.initial_leg_class[ids] = leg_classes
        self.entry_leg_class[ids] = -1
        self.entry_q[ids] = self.entry_dq[ids] = 0.
        self.entry_omega[ids] = 0.

    def update(self, state, *, eligible, gravity, omega, q, dq, orbits, geometry_valid,
               encoder_error, failed):
        c = self.config
        enable = state["enable"] & eligible
        self.enabled_at[enable] = state["elapsed"][enable]
        # Initialized active dynamic states do not have a preceding passive interval.
        missing = eligible & ~state["passive"] & ~torch.isfinite(self.enabled_at)
        self.enabled_at[missing] = (state["elapsed"]-self.dt)[missing]
        active = eligible & ~state["passive"] & ~failed
        valid = (active & (gravity[:, 2] <= -math.cos(math.radians(c["upright_tilt_deg"])))
                 & (omega.norm(dim=-1) <= c["upright_max_body_rate_rad_s"])
                 & (dq.abs().amax(-1) <= c["upright_max_leg_speed_rad_s"])
                 & geometry_valid.all(-1) & (encoder_error <= c["upright_max_encoder_error_m"]))
        self.upright_hold.copy_(torch.where(valid, self.upright_hold+self.dt, 0.))
        event = valid & ~self.entered & (self.upright_hold+1e-7 >= c["upright_hold_seconds"])
        self.entered |= event
        self.upright_at[event] = state["elapsed"][event]
        front = orbits.sin() >= 0
        cls = torch.where(front[:, 0], torch.where(front[:, 1], 0, 2), torch.where(front[:, 1], 3, 1))
        self.entry_leg_class[event] = cls[event]
        self.entry_q[event], self.entry_dq[event] = q[event], dq[event]
        self.entry_omega[event] = omega[event]
        active_time = (state["elapsed"]-self.enabled_at).clamp_min(0.)
        return event, torch.nan_to_num(active_time, nan=0.)


def potential(gravity, height, support_error):
    """Bounded shaping without a prescribed fold angle or a sustained pose bonus."""
    up = ((1.-gravity[:, 2])*.5).clamp(0., 1.)
    lift = ((height-.12)/(.305-.12)).clamp(0., 1.)
    aligned = 1./(1.+(support_error/.12).square())
    return 2.*up + 2.*up*aligned + 4.*up*lift


def recovery_terms(state, config, impact_cost_density):
    """Time is primary; weak effort and impact regularization are secondary."""
    return dict(rise_time=-config["time_cost_per_second"]*torch.ones_like(state["height"]),
        effort=-.000025*state["torque"].square().mean(-1),
        action_rate=-.001*state["action_delta"], impact=-impact_cost_density)
