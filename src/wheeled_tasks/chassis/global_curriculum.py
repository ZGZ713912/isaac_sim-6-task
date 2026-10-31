"""Measured skill admission, persistent replay and no elapsed-update unlocks."""
from __future__ import annotations

from copy import deepcopy
import math

SKILLS = ("standing", "straight", "static_height", "turn", "spin", "braking",
          "moving_height", "turning_height", "spinning_height")
ROLES = dict(standing=("stand",), static_height=("height",),
    straight=("forward", "reverse"), moving_height=("forward", "reverse"),
    braking=("forward", "reverse"), turn=("arc_fl", "arc_fr", "arc_rl", "arc_rr"),
    turning_height=("arc_fl", "arc_fr", "arc_rl", "arc_rr"),
    spin=("spin_left", "spin_right"), spinning_height=("spin_left", "spin_right"))


def validate(config):
    c = deepcopy(config)
    if c.get("schema") != "wheel-leg-global-mastery-v1" or set(c["skills"]) != set(SKILLS):
        raise ValueError("Global curriculum requires all standing/motion/height skills")
    for k in ("evaluation_interval_updates", "required_consecutive_passes", "minimum_domain_updates"):
        if type(c[k]) is not int or c[k] < 1: raise ValueError("Invalid curriculum count: "+k)
    if not 0 < c["minimum_success_rate"] <= 1 or not 0 < c["current_level_fraction"] < 1:
        raise ValueError("Invalid curriculum success/replay fraction")
    if c["nominal_height_m"] != .32 or c["budget_policy"] != "one_fresh_run_no_extension":
        raise ValueError("Retain the URDF nominal birth and single-run budget")
    headings = c["evaluation_headings_rad"]
    if len(headings) < 2 or len(set(headings)) != len(headings) or not all(math.isfinite(x) for x in headings):
        raise ValueError("Need independent finite cold headings")
    if set(c["training_weights"]) != set(SKILLS) | {"recovery"} or any(
            not math.isfinite(v) or v <= 0 for v in c["training_weights"].values()):
        raise ValueError("Every learned skill must retain a positive replay share")
    if c["training_weights"]["standing"] != 25 or c["training_weights"]["recovery"] != 30:
        raise ValueError("Keep final25% standing and30% recovery shares")
    if not math.isclose(sum(c["training_weights"].values()),100.):
        raise ValueError("Final curriculum shares must sum to100")
    for k in ("standing_seconds", "motion_seconds"):
        if not math.isfinite(c[k]) or c[k] <= 2: raise ValueError("Invalid cold horizon")
    if c["motion_seconds"] < 20 or any(not math.isfinite(v) or v <= 0 for v in c["limits"].values()):
        raise ValueError("Need a full command/braking window and finite positive limits")
    # Check the dependency graph at LEVEL granularity, including prior levels.
    edges = {}
    for name, spec in c["skills"].items():
        if not spec["levels"]: raise ValueError("An empty skill cannot be mastered")
        for i, lev in enumerate(spec["levels"]):
            deps = [(name,i-1)] if i else []
            for req in spec["requires"]+lev.get("requires",[]):
                n,count = req["skill"],req["levels"]
                if n not in SKILLS or type(count) is not int or not 1 <= count <= len(c["skills"][n]["levels"]):
                    raise ValueError("Invalid skill dependency")
                deps.append((n,count-1))
            if any(type(v) not in (float,int) or not math.isfinite(v) or v <= 0
                   for k,v in lev.items() if k != "requires"):
                raise ValueError("Invalid skill target")
            edges[name,i] = deps
    visited,active = set(),set()
    def visit(key):
        if key in active: raise ValueError("Cyclic curriculum dependencies")
        if key in visited:return
        active.add(key)
        for parent in edges[key]:visit(parent)
        active.remove(key);visited.add(key)
    for key in edges:visit(key)
    if c["skills"]["straight"]["levels"][-1]["vx_m_s"] != 2. or c["skills"]["spin"]["levels"][-1]["rpm"] != 100.:
        raise ValueError("Keep final2m/s and100rpm targets")
    return c


def descriptor(config, skill, level, role, height_sign=0):
    lev = config["skills"][skill]["levels"][level]
    h = (config["nominal_height_m"] if skill == "standing" else
         .305+((.23 if height_sign == 0 else .42)-.305)*lev.get("height_fraction",0.))
    v,w = lev.get("vx_m_s",0.), lev.get("yaw_rad_s",lev.get("rpm",0.)*math.tau/60.)
    if role in ("reverse","arc_rl","arc_rr"):v = -v
    if role in ("spin_right","arc_fr","arc_rr"):w = -w
    return dict(curriculum_skill=skill,curriculum_level=level,role=role,recovery=False,
        target=[v,w,h],birth_height=config["nominal_height_m"] if skill == "standing" else .305,
        nominal_birth=skill == "standing",pitch_deg=0.,
        moving_height=skill in ("moving_height","turning_height","spinning_height"),
        abrupt_brake=skill == "braking")


def assessment_descriptors(config, skill, level):
    rows = []
    heights = (0,1) if "height" in skill else (0,)
    for role in ROLES[skill]:
        for h in heights:
            for heading in config["evaluation_headings_rad"]:
                row = descriptor(config,skill,level,role,h)
                row.update(heading_rad=heading,case_id=f"{skill}/{level}/{role}/{h}/{heading:.12g}",
                           source="cold_curriculum")
                rows.append(row)
    return rows


class GlobalCurriculum:
    def __init__(self, plan, recovery_mastery):
        self.config = validate(plan["global_curriculum"])
        self.recovery = recovery_mastery
        self.levels = dict.fromkeys(SKILLS,0)
        self.streak = {}
        self.skill_last_review = {}
        self.last_review = None
        self.reviews = []
        self.complex_passed = False
        self.domain_sequence = deepcopy(plan["domain_sequence"])
        self.domain_index = 0
        self.domains_admitted = False
        self.domain_started_at = 0
        self.max_updates = plan["max_updates"]

    def requirements_met(self, requirements):
        return all(self.levels[r["skill"]] >= r["levels"] for r in requirements)

    def learning_level(self, skill):
        spec = self.config["skills"][skill]
        if not self.requirements_met(spec["requires"]):return None
        i = min(self.levels[skill],len(spec["levels"])-1)
        if self.requirements_met(spec["levels"][i].get("requires",[])):return i
        return self.levels[skill]-1 if self.levels[skill] else None

    def due(self, completed):
        # New prerequisites may become available during this review. They must
        # receive training before another test, never re-enter mid-rollout.
        if self.reviews and self.reviews[-1]["completed_updates"] == completed:return False
        regular = completed > 0 and completed % self.config["evaluation_interval_updates"] == 0 and self.last_review != completed
        belly = self.levels["standing"] > 0 and self.recovery.due(completed)
        return regular or belly

    def jobs(self, completed, force=False):
        regular = force or (completed > 0 and completed % self.config["evaluation_interval_updates"] == 0 and self.last_review != completed)
        result = [(s,self.learning_level(s)) for s in SKILLS if regular and self.learning_level(s) is not None]
        if self.levels["standing"] and (force or self.recovery.due(completed) or regular and self.recovery.unlocked):
            result.append(("belly",0))
        if regular and self.recovery.unlocked:result.append(("complex",0))
        return result

    def review_skill(self, skill, level, rows, completed):
        key = (skill,level)
        if completed <= self.skill_last_review.get(key,-1):raise ValueError("Repeated/older skill review")
        available = self.learning_level(skill)
        if available is None or level > available:raise ValueError("Skill prerequisites have not passed")
        expected = {r["case_id"] for r in assessment_descriptors(self.config,skill,level)}
        ids = [r["case_id"] for r in rows]
        if len(set(ids)) != len(ids) or any(r.get("source") != "cold_curriculum" for r in rows):
            raise ValueError("Only unique cold trials may admit a skill")
        groups = {identity.rsplit("/",1)[0]:[] for identity in expected}
        for r in rows:
            prefix = r["case_id"].rsplit("/",1)[0]
            if prefix in groups:groups[prefix].append(r)
        per_group = {g:bool(values) and sum(r.get("passed") is True for r in values)/len(values)
            >= self.config["minimum_success_rate"] for g,values in groups.items()}
        passed = set(ids) == expected and all(per_group.values())
        self.skill_last_review[key] = completed
        self.streak[key] = self.streak.get(key,0)+1 if passed else 0
        # Previously learned levels remain replay. A blocked next level cannot
        # be admitted by re-testing its easier predecessor.
        if (level == self.levels[skill] and self.learning_level(skill) == level
                and self.streak[key] >= self.config["required_consecutive_passes"]):
            self.levels[skill] += 1
        return dict(skill=skill,level=level,trials=len(rows),passed=passed,completed_updates=completed,
                    mastered_levels=self.levels[skill],per_direction_height_group=per_group)

    def review_complex(self, rows, completed):
        if not self.levels["standing"] or not self.recovery.unlocked:
            raise ValueError("Complex recovery requires standing and all four belly classes")
        if completed <= self.skill_last_review.get(("complex",0),-1):raise ValueError("Repeated complex review")
        from .multi_pose_recovery import FAMILIES,LEG_CLASSES
        groups = {(f,c):[] for f in FAMILIES if f != "belly_down" for c in LEG_CLASSES}
        seen = set()
        for r in rows:
            identity = (r["pose_index"],r["heading_rad"])
            if r.get("split") != "holdout" or (r["family"],r["leg_class"]) not in groups or identity in seen:
                raise ValueError("Complex recovery needs independent held-out family/leg trials")
            seen.add(identity);groups[r["family"],r["leg_class"]].append(r)
        summary = {f+"/"+c:dict(trials=len(values),passed=len(values) >= 2 and
            sum(r.get("passed") is True for r in values)/len(values) >= self.config["minimum_success_rate"])
            for (f,c),values in groups.items()}
        passed = all(r["passed"] for r in summary.values())
        self.skill_last_review["complex",0] = completed
        self.streak["complex",0] = self.streak.get(("complex",0),0)+1 if passed else 0
        if self.streak["complex",0] >= self.config["required_consecutive_passes"]:self.complex_passed = True
        return dict(skill="complex",passed=passed,groups=summary,completed_updates=completed)

    @property
    def all_mastered(self):
        return self.recovery.unlocked and self.complex_passed and all(
            self.levels[s] == len(self.config["skills"][s]["levels"]) for s in SKILLS)

    def domain_context(self):
        if not self.domains_admitted:return "nominal",0.
        stage = self.domain_sequence[self.domain_index]
        return stage["profile"],stage["strength"]

    def finish_review(self, summaries, completed, *, regular=True):
        if self.reviews and completed <= self.reviews[-1]["completed_updates"]:
            raise ValueError("Repeated/older global review")
        if regular:
            if self.last_review is not None and completed <= self.last_review:raise ValueError("Repeated global review")
            self.last_review = completed
        required = set(SKILLS) | {"belly","complex"}
        passed_all = required.issubset(summaries) and all(summaries[s]["passed"] for s in required)
        before = self.domain_context()
        if (self.all_mastered and passed_all and completed < self.max_updates
                and completed-self.domain_started_at >= self.config["minimum_domain_updates"]):
            if not self.domains_admitted:
                self.domains_admitted = True
                self.domain_index = int(self.domain_sequence[0]["profile"] == "nominal" and len(self.domain_sequence)>1)
            elif self.domain_index+1 < len(self.domain_sequence):self.domain_index += 1
            self.domain_started_at = completed
        result = dict(completed_updates=completed,skills=deepcopy(summaries),all_skills_mastered=self.all_mastered,
                      domain_before=before,domain_after=self.domain_context(),budget_extended=False)
        self.reviews.append(result)
        return result

    def training_descriptors(self, count, generation):
        active = [s for s in SKILLS if self.learning_level(s) is not None]
        if self.levels["standing"]:active.append("recovery")
        weights = self.config["training_weights"]
        total = sum(weights[s] for s in active)
        # At least one lane per available skill; distribute the remaining lanes
        # by largest remainders, then rotate cases across episodes.
        if count < len(active):raise ValueError("Not enough environments for persistent skill replay")
        quotas = {s:count*weights[s]/total for s in active}
        slots = {s:max(1,math.floor(quotas[s])) for s in active}
        while sum(slots.values()) > count:
            s = max((s for s in active if slots[s]>1),key=lambda s:slots[s]-quotas[s])
            slots[s] -= 1
        while sum(slots.values()) < count:
            s = max(active,key=lambda s:quotas[s]-slots[s]);slots[s] += 1
        pool = [s for s in active for _ in range(slots[s])]
        rows = []
        for i,s in enumerate(pool):
            token = i+generation*(count+1)
            if s == "recovery":
                lev = self.learning_level("straight")
                speed = self.config["skills"]["straight"]["levels"][lev]["vx_m_s"] if lev is not None else 0.
                rows.append(dict(curriculum_skill="recovery",role="prone",recovery=True,
                    target=[speed,0.,.305],birth_height=.305,pitch_deg=0.))
                continue
            level = self.learning_level(s)
            if self.levels[s] and token%10 >= round(10*self.config["current_level_fraction"]):
                level = (token//10)%self.levels[s]
            role = ROLES[s][(token//2)%len(ROLES[s])]
            row = descriptor(self.config,s,level,role,token%2)
            # Once static height transitions are mastered, retain low/high
            # standing births as well as the literal URDF nominal birth.
            if s == "standing" and self.levels["static_height"] == len(self.config["skills"]["static_height"]["levels"]):
                h = (.32,.305,.23,.42)[token%4]
                row.update(nominal_birth=h == .32,birth_height=h,target=[0.,0.,h])
            rows.append(row)
        return rows

    def state_dict(self):
        return dict(schema="wheel-leg-global-mastery-state-v1",mastered_levels=dict(self.levels),
            pass_streaks=[dict(skill=s,level=i,count=n,last_update=self.skill_last_review.get((s,i))) for (s,i),n in self.streak.items()],
            complex_passed=self.complex_passed,belly=self.recovery.state_dict(),all_skills_mastered=self.all_mastered,
            last_review=self.last_review,domains_admitted=self.domains_admitted,domain_index=self.domain_index,
            domain_started_at=self.domain_started_at,domain_context=self.domain_context(),reviews=deepcopy(self.reviews))
