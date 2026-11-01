"""Per-class evaluation gate; elapsed updates and training rewards cannot unlock it."""
from __future__ import annotations

from copy import deepcopy
import math

CLASSES = ("front_front","rear_rear","front_rear","rear_front")


def validate(config):
    c = deepcopy(config)
    if c.get("schema") != "wheel-leg-belly-mastery-v1":
        raise ValueError("Need an explicit belly mastery contract")
    for key in ("evaluation_interval_updates","minimum_trials_per_leg_class","required_consecutive_passes"):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError("Invalid mastery count: "+key)
    if not 0 < c.get("minimum_success_rate",0.) <= 1.:
        raise ValueError("Mastery success rate must be in (0,1]")
    for key in ("post_completion_hold_seconds","max_active_recovery_seconds"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("Invalid mastery time: "+key)
    if not c["evaluation_headings_rad"] or not all(math.isfinite(x) for x in c["evaluation_headings_rad"]):
        raise ValueError("Mastery needs finite evaluation headings")
    if len(set(c["evaluation_headings_rad"])) != len(c["evaluation_headings_rad"]):
        raise ValueError("Repeated headings cannot provide independent mastery trials")
    if type(c.get("require_current_pass",False)) is not bool:
        raise ValueError("Recovery retention flag must be boolean")
    if c.get("reset_scope") == "standard_prone_only":
        if (len(c.get("required_leg_classes",[])) != 1 or c["required_leg_classes"][0] not in CLASSES
                or len(c["evaluation_headings_rad"]) < c["minimum_trials_per_leg_class"]):
            raise ValueError("Standard cold review requires its single posture and declared heading trials")
    return c


class RecoveryMastery:
    def __init__(self, config):
        self.config = validate(config)
        self.unlocked_at = None
        self.last_evaluated_update = None
        self.pass_streak = 0
        self.reviews = []

    @property
    def unlocked(self):
        return self.unlocked_at is not None and (not self.config.get("require_current_pass")
            or not self.reviews or self.reviews[-1]["all_classes_passed"])

    def due(self, completed):
        return (not self.unlocked and completed > 0
                and completed % self.config["evaluation_interval_updates"] == 0
                and self.last_evaluated_update != completed)

    def review(self, rows, completed):
        if self.last_evaluated_update is not None and completed <= self.last_evaluated_update:
            raise ValueError("A repeated/older evaluation cannot advance mastery")
        standard = self.config.get("reset_scope") == "standard_prone_only"
        groups = {c:[] for c in self.config.get("required_leg_classes",CLASSES)}
        seen = set()
        for row in rows:
            if row.get("family") != "belly_down" or row.get("split") != ("standard" if standard else "holdout") or row.get("leg_class") not in groups:
                raise ValueError("Only independent held-out belly trials can admit complex recovery")
            identity = (row["pose_index"],row["heading_rad"])
            if identity in seen:
                raise ValueError("Duplicate mastery trials cannot inflate a class denominator")
            seen.add(identity)
            groups[row["leg_class"]].append(row)
        summary = {}
        for cls,values in groups.items():
            successes = sum(bool(r.get("passed")) for r in values)
            rate = successes/len(values) if values else 0.
            summary[cls] = dict(trials=len(values),successes=successes,success_rate=rate,
                passed=len(values) >= self.config["minimum_trials_per_leg_class"]
                    and rate >= self.config["minimum_success_rate"])
        passed = all(s["passed"] for s in summary.values())
        self.pass_streak = self.pass_streak+1 if passed else 0
        self.last_evaluated_update = completed
        if not self.unlocked and self.pass_streak >= self.config["required_consecutive_passes"]:
            self.unlocked_at = completed
        result = dict(completed_updates=completed,per_leg_class=summary,all_classes_passed=passed,
            pass_streak=self.pass_streak,complex_recovery_unlocked=self.unlocked_at is not None and
                (not self.config.get("require_current_pass") or passed),unlocked_at=self.unlocked_at)
        self.reviews.append(result)
        return result

    def state_dict(self):
        return dict(config=deepcopy(self.config),unlocked_at=self.unlocked_at,
            last_evaluated_update=self.last_evaluated_update,pass_streak=self.pass_streak,reviews=deepcopy(self.reviews))

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.state_dict()) or state["config"] != self.config:
            raise ValueError("Recovery mastery checkpoint and course disagree")
        for name in ("unlocked_at", "last_evaluated_update"):
            value = state[name]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Invalid recovery mastery clock: "+name)
        if type(state["pass_streak"]) is not int or state["pass_streak"] < 0 or not isinstance(state["reviews"], list):
            raise ValueError("Invalid recovery mastery history")
        if state["unlocked_at"] is not None and (
                state["last_evaluated_update"] is None or state["unlocked_at"] > state["last_evaluated_update"]):
            raise ValueError("Recovery unlock is newer than its evaluation")
        self.unlocked_at = state["unlocked_at"]
        self.last_evaluated_update = state["last_evaluated_update"]
        self.pass_streak = state["pass_streak"]
        self.reviews = deepcopy(state["reviews"])
