"""Reset-only curriculum and continuous, shared flip-to-rise task state.

No motor targets, physical state writes, teacher or checkpoint loading live here.
Family/leg labels are privileged diagnostics, never additional actor inputs.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random

import torch
from .masked_tensor import copy_masked_rows_

FAMILIES = ("belly_down", "left_side", "right_side", "roof_down", "front_down", "rear_down", "diagonal")
LEG_CLASSES = ("front_front", "rear_rear", "front_rear", "rear_front")


def standard_only(config):
    return config.get("reset_scope") == "standard_prone_only"


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
        if (set(weights) != set(FAMILIES) or any(not math.isfinite(v) or v < 0 for v in weights.values())
                or not math.isclose(sum(weights.values()), 1., abs_tol=1e-8)):
            raise ValueError("Recovery family weights must be nonnegative and sum to one")
    if c.get("reset_scope") not in (None,"standard_prone_only"):
        raise ValueError("Unknown recovery reset scope")
    if c.get("dense_reward") is not None:
        if c["dense_reward"].get("schema") == "wheel-leg-phase-recovery-reward-v1":
            from .recovery_reward import validate as validate_dense
            validate_dense(c["dense_reward"])
        else:
            from .standard_getup_reward import validate as validate_dense
            validate_dense(c["dense_reward"])
            if not standard_only(c):
                raise ValueError("Standard dense reward requires the single-posture reset scope")
    if c.get("practice_distribution") not in (None,"all_families_from_start"):
        raise ValueError("Unknown recovery practice distribution")
    if c.get("practice_distribution") and standard_only(c):
        raise ValueError("All-family practice requires the full pose bank")
    if c.get("start_curriculum") is not None:
        from .recovery_start_curriculum import validate as validate_starts
        validate_starts(c["start_curriculum"])
        if standard_only(c) or c.get("practice_distribution") != "all_families_from_start":
            raise ValueError("Recovery start curriculum must retain full-family practice")
    if standard_only(c):
        if not c.get("standard_reset_pose_name") or any(s["family_weights"] != {
                f:float(f == "belly_down") for f in FAMILIES} for s in stages):
            raise ValueError("Standard recovery cannot sample other fallen postures")
        if c.get("mastery_gate",{}).get("reset_scope") != "standard_prone_only":
            raise ValueError("Standard recovery requires matching cold-review scope")
    elif any(v <= 0 for v in stages[-1]["family_weights"].values()):
        raise ValueError("The final recovery stage must retain every fallen family")
    if c.get("mastery_gate") is not None:
        from .recovery_mastery import validate as validate_mastery
        validate_mastery(c["mastery_gate"])
        if len(stages) < 2:
            raise ValueError("Gated recovery needs separate prerequisite and complex stages")
        if stages[0]["family_weights"] != {f:float(f == "belly_down") for f in FAMILIES}:
            raise ValueError("Gated recovery must begin with only four-class belly recovery")
        if c["curriculum_mode"] != "staged":
            raise ValueError("A mastery gate cannot be bypassed with all-family sampling")
    if c.get("mastery_family_weights") is not None:
        w = c["mastery_family_weights"]
        if set(w) != set(FAMILIES) or any(not math.isfinite(v) or v < (0 if standard_only(c) else 1e-30) for v in w.values()) or not math.isclose(sum(w.values()),1.):
            raise ValueError("Post-mastery replay must retain every family")
        if standard_only(c) and w != {f:float(f == "belly_down") for f in FAMILIES}:
            raise ValueError("Standard replay cannot re-enable complex births")
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
        if config.get("require_native_reset_admission") and info.get("native_passive_admitted") is not True:
            raise ValueError("Recovery reset lacks native passive admission")
        root = pose["root_position_m"]+pose["root_quaternion_xyzw"]
        if (len(root) != 7 or not all(math.isfinite(v) for v in root)
                or not math.isclose(sum(v*v for v in root[3:]), 1., abs_tol=1e-6)):
            raise ValueError("Invalid reset root transform")
        pools[key].append(i)
    if standard_only(config):
        selected=[i for i,p in enumerate(bank["poses"]) if p["name"] == config["standard_reset_pose_name"]]
        if len(selected) != 1 or len([p for p in bank["poses"] if p.get("recovery")]) != 1:
            raise ValueError("Standard bank must contain exactly one declared recovery posture")
        r=bank["poses"][selected[0]]["recovery"]
        if r["family"] != "belly_down" or r["split"] != "standard" or r["leg_class"] not in config["mastery_gate"]["required_leg_classes"]:
            raise ValueError("Standard recovery posture and evaluation disagree")
        return pools
    if any(not pool for pool in pools.values()):
        raise ValueError("Missing body-family x independent-leg reset cell")
    gate = config.get("mastery_gate")
    if gate is not None:
        for cls in LEG_CLASSES:
            heldout = sum(bank["poses"][i]["recovery"]["split"] == "holdout" for i in pools[("belly_down",cls)])
            if heldout*len(gate["evaluation_headings_rad"]) < gate["minimum_trials_per_leg_class"]:
                raise ValueError("Mastery minimum exceeds independent available held-out trials")
    return pools


def validate_native_admission(bank, root, rates):
    """Bind every sampled pose to policy-independent, original-rate evidence."""
    import hashlib
    import json
    binding = bank["multi_pose_recovery"].get("native_reset_admission",{})
    path = (root/binding.get("file","")).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or (
            hashlib.sha256(path.read_bytes()).hexdigest() != binding.get("sha256")):
        raise ValueError("Native reset admission evidence changed or is missing")
    evidence = json.loads(path.read_text())
    if (evidence.get("schema") != "native-passive-reset-admission-v1" or evidence.get("new_PPO") != 0
            or evidence.get("checkpoint_reads") != [] or evidence.get("all_pass") is not True
            or evidence.get("seconds",0) < .15 or evidence.get("guard_m") != .003
            or evidence.get("physics_hz") != round(1/rates["physics_dt"])
            or evidence.get("policy_hz") != round(1/rates["policy_dt"])):
        raise ValueError("Native reset evidence differs from the training physical contract")
    if rates["pc_control_dt"] != rates["physics_dt"]:
        actual = evidence.get("control_timing", {})
        audit = actual.get("runtime_substep_verification", {})
        if (evidence.get("feedback_pd_hz") != round(1/rates["pc_control_dt"])
                or actual.get("pc_control_dt") != rates["pc_control_dt"]
                or audit.get("pd_updates") != round(rates["policy_dt"]/rates["pc_control_dt"])
                or audit.get("feedback_refreshes") != rates["decimation"]
                or not evidence.get("runtime_source_hashes")):
            raise ValueError("Native reset evidence lacks the separate feedback-PD clock")
        for name, value in evidence["runtime_source_hashes"].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != value:
                raise ValueError("Measured passive runtime source changed: " + name)
    for i,pose in enumerate(bank["poses"]):
        if not pose.get("recovery"):continue
        rows = [r for r in evidence["cases"] if r["pose_index"] == i]
        if {r["heading_rad"] for r in rows} != {0.,math.pi/2} or len(rows) != 2 or any(
                r["passed"] is not True or r["terminated"] or r["mechanical"]
                or r["closure_peak_m"] > .0028 or r["passive_motor_peak_nm"] != 0. for r in rows):
            raise ValueError("Missing complete native admission for reset: "+pose["name"])
    return evidence


class ResetSampler:
    """Seeded weighted family sampling; cycle all four leg classes per family."""
    def __init__(self, bank, config, seed, mastery=None):
        self.config = validate(config)
        self.bank = bank
        self.rng = random.Random(seed)
        self.standard_index = (next(i for i,p in enumerate(bank["poses"])
            if p["name"] == config["standard_reset_pose_name"]) if standard_only(config) else None)
        if config.get("mastery_gate") is not None:
            from .recovery_mastery import RecoveryMastery
            self.mastery = mastery or RecoveryMastery(config["mastery_gate"])
        else:
            self.mastery = None
        self.class_cursor = dict.fromkeys(FAMILIES, 0)
        self.pools = {(f, l): [] for f in FAMILIES for l in LEG_CLASSES}
        for i, p in enumerate(bank["poses"]):
            if "recovery" in p and p["recovery"]["split"] == "train":
                r = p["recovery"]
                self.pools[(r["family"], r["leg_class"])].append(i)
        if self.standard_index is None and any(not p for p in self.pools.values()):
            raise ValueError("Cannot silently replace a missing recovery cell with standard prone")
        self.start_curriculum = None
        if config.get("start_curriculum") is not None:
            from .recovery_start_curriculum import RecoveryStartCurriculum
            self.start_curriculum = RecoveryStartCurriculum(bank, config["start_curriculum"],
                "recovery-start:"+str(seed))

    def state_dict(self):
        state = dict(schema="wheel-leg-recovery-sampler-v1", rng=self.rng.getstate(),
                     class_cursor=dict(self.class_cursor))
        if self.start_curriculum is not None:
            state["start_curriculum"] = self.start_curriculum.state_dict()
        return state

    @staticmethod
    def validate_state(state):
        if (not isinstance(state, dict) or set(state) not in (
                {"schema", "rng", "class_cursor"}, {"schema", "rng", "class_cursor", "start_curriculum"})
                or state["schema"] != "wheel-leg-recovery-sampler-v1"
                or not isinstance(state["class_cursor"], dict)
                or set(state["class_cursor"]) != set(FAMILIES)
                or any(type(v) is not int or v < 0 for v in state["class_cursor"].values())):
            raise ValueError("Invalid recovery sampler checkpoint")
        try:
            random.Random().setstate(state["rng"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid recovery sampler RNG") from exc
        if "start_curriculum" in state:
            from .recovery_start_curriculum import RecoveryStartCurriculum
            RecoveryStartCurriculum.validate_state(state["start_curriculum"])

    def load_state_dict(self, state):
        self.validate_state(state)
        if (self.start_curriculum is None) != (state.get("start_curriculum") is None):
            raise ValueError("Recovery sampler checkpoint start curriculum differs")
        self.rng.setstate(state["rng"])
        self.class_cursor = dict(state["class_cursor"])
        if self.start_curriculum is not None:
            self.start_curriculum.load_state_dict(state["start_curriculum"])

    def sample_start(self, completed):
        if self.start_curriculum is None:
            return self.sample(completed), -1
        return self.start_curriculum.sample(lambda: self.sample(completed))

    def sample(self, completed):
        if self.standard_index is not None:return self.standard_index
        stages = self.config["curriculum"]
        if self.mastery is not None:
            # Update numbers alone cannot unlock side/inverted births. After
            # mastery, retain the original staged mixture on a relative clock.
            clock = (stages[1]["start_update"]+max(0,completed-self.mastery.unlocked_at)
                     if self.mastery.unlocked else 0)
        else:
            clock = completed
        stage = stages[-1] if (self.config["curriculum_mode"] == "all"
            or self.config.get("practice_distribution") == "all_families_from_start") else next(
            s for s in reversed(stages) if clock >= s["start_update"])
        weights = stage["family_weights"]
        if self.mastery is not None and self.mastery.unlocked and self.config.get("mastery_family_weights"):
            weights = self.config["mastery_family_weights"]
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
        copy_masked_rows_(self.enabled_at, enable, state["elapsed"])
        # Initialized active dynamic states do not have a preceding passive interval.
        missing = eligible & ~state["passive"] & ~torch.isfinite(self.enabled_at)
        copy_masked_rows_(self.enabled_at, missing, state["elapsed"]-self.dt)
        active = eligible & ~state["passive"] & ~failed
        valid = (active & (gravity[:, 2] <= -math.cos(math.radians(c["upright_tilt_deg"])))
                 & (omega.norm(dim=-1) <= c["upright_max_body_rate_rad_s"])
                 & (dq.abs().amax(-1) <= c["upright_max_leg_speed_rad_s"])
                 & geometry_valid.all(-1) & (encoder_error <= c["upright_max_encoder_error_m"]))
        self.upright_hold.copy_(torch.where(valid, self.upright_hold+self.dt, 0.))
        event = valid & ~self.entered & (self.upright_hold+1e-7 >= c["upright_hold_seconds"])
        self.entered |= event
        copy_masked_rows_(self.upright_at, event, state["elapsed"])
        front = orbits.sin() >= 0
        cls = torch.where(front[:, 0], torch.where(front[:, 1], 0, 2), torch.where(front[:, 1], 3, 1))
        copy_masked_rows_(self.entry_leg_class, event, cls)
        copy_masked_rows_(self.entry_q, event, q)
        copy_masked_rows_(self.entry_dq, event, dq)
        copy_masked_rows_(self.entry_omega, event, omega)
        active_time = (state["elapsed"]-self.enabled_at).clamp_min(0.)
        return event, torch.nan_to_num(active_time, nan=0.)


def potential(gravity, height, support_error):
    """Bounded shaping without a prescribed fold angle or a sustained pose bonus."""
    up = ((1.-gravity[:, 2])*.5).clamp(0., 1.)
    lift = ((height-.12)/(.305-.12)).clamp(0., 1.)
    aligned = 1./(1.+(support_error/.12).square())
    return 2.*up + 2.*up*aligned + 4.*up*lift


def recovery_terms(state, config, impact_cost_density, *, scores=None):
    """Time/effort costs, plus explicit continuous standard-getup guidance."""
    terms = dict(rise_time=-config["time_cost_per_second"]*torch.ones_like(state["height"]),
        effort=-.000025*state["torque"].square().mean(-1),
        action_rate=-.001*state["action_delta"], impact=-impact_cost_density)
    if config.get("dense_reward") is not None:
        if config["dense_reward"].get("schema") == "wheel-leg-phase-recovery-reward-v1":
            from .recovery_reward import dense_terms
        else:
            from .standard_getup_reward import dense_terms
        terms.update(dense_terms(state, config["dense_reward"], scores=scores))
    return terms
