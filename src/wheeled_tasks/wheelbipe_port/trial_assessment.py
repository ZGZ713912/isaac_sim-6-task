"""Paired retention and physical jump checks; never controls a learner."""
from __future__ import annotations

import math


def assess(baseline, candidate, before_mixed, after_mixed, spec):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    def finite(value):
        return isinstance(value, (float, int)) and math.isfinite(value)
    limits = spec["maximum_added_error"]
    for label, receipt in (("baseline", baseline), ("candidate", candidate)):
        require(receipt.get("status") == "complete" and receipt.get("ppo_updates") == 0,
                label+": incomplete read-only legacy replay")
        require(all(receipt.get("arguments", {}).get(k) == v for k, v in spec["legacy_evaluation"].items()),
                label+": legacy replay configuration differs")
    old = {x["case"]: x for x in baseline.get("cases", [])}
    new = {x["case"]: x for x in candidate.get("cases", [])}
    for case in spec["legacy_cases"]:
        if case not in old or case not in new:
            errors.append(case+": missing legacy replay")
            continue
        b, a = old[case], new[case]
        require(a["command"] == b["command"] and a["resets"] <= b["resets"], case+": command changed or extra resets")
        bw, aw = b["windows"]["tail_half"], a["windows"]["tail_half"]
        target = a["command"]
        for key, index, tolerance in (("vx_m_s", 0, limits["vx_error_m_s"]),
                                      ("yaw_rad_s", 1, limits["yaw_error_rad_s"])):
            av, bv = aw[key]["mean"], bw[key]["mean"]
            require(finite(av) and finite(bv) and abs(av-target[index]) <= abs(bv-target[index])+tolerance,
                    case+": tracking regression "+key)
        for key, stat, tolerance in (("height_error_m", "abs_mean", limits["height_error_m"]),
                                      ("tilt_deg", "mean", limits["tilt_deg"])):
            av, bv = aw[key][stat], bw[key][stat]
            require(finite(av) and finite(bv) and av <= bv+tolerance, case+": regression "+key)
    expected_seeds = set(spec["mixed_evaluation"]["seeds"])
    before = {x.get("arguments", {}).get("seed"): x for x in before_mixed}
    after = {x.get("arguments", {}).get("seed"): x for x in after_mixed}
    require(set(before) == set(after) == expected_seeds, "Missing held-out mixed evaluation seed")
    results = []
    for seed in sorted(expected_seeds & before.keys() & after.keys()):
        b, a = before[seed], after[seed]
        require(b.get("checkpoint_sha256") == baseline.get("checkpoint_sha256")
            and a.get("checkpoint_sha256") == candidate.get("checkpoint_sha256"),
            f"{seed}: mixed and legacy checkpoints differ")
        require(b.get("arguments", {}).get("course_update") == 0
            and a.get("arguments", {}).get("course_update", 0) > 0,
            f"{seed}: wrong jump curriculum clocks")
        for label, receipt in (("before", b), ("after", a)):
            args = receipt.get("arguments", {})
            require(receipt.get("status") == "complete" and receipt.get("ppo_updates") == 0
                and receipt.get("optimizer_created") is False and receipt.get("finite_native_states") is True,
                f"{seed}/{label}: invalid mixed receipt")
            require(all(args.get(k) == v for k, v in spec["mixed_evaluation"].items() if k != "seeds")
                and args.get("randomization_profile") == "nominal" and args.get("randomization_strength") == 0.,
                f"{seed}/{label}: mixed configuration differs")
        gap = a.get("max_constraint_gap_policy_sample_m")
        require(finite(gap) and gap <= spec["closed_chain_gap_max_m"], f"{seed}: excessive sampled closed-chain gap")
        for task in ("stand", "forward", "reverse", "height", "turn", "dash", "spin"):
            if task not in a.get("tasks", {}) or task not in b.get("tasks", {}):
                errors.append(f"{seed}: missing ordinary task "+task)
                continue
            ar, br = a["tasks"][task], b["tasks"][task]
            require(ar["resets"] <= br["resets"], f"{seed}/{task}: extra resets")
            for key, tolerance in limits.items():
                av, bv = ar.get("tracking_mean_absolute", {}).get(key), br.get("tracking_mean_absolute", {}).get(key)
                require(finite(av) and finite(bv) and av <= bv+tolerance, f"{seed}/{task}: tracking regression "+key)
        outcomes = a.get("episode_outcomes", {})
        fractions = {}
        for task in ("jump", "airborne"):
            row = outcomes.get(task, {})
            count = row.get("completed", 0)
            minimum = a.get("allocated_envs", {}).get(task, 0)*3
            require(count >= max(minimum, 1), f"{seed}/{task}: too few complete physical episodes")
            fractions[task] = {key: row.get(key, 0)/max(count, 1) for key in ("flown", "succeeded")}
        require(fractions["jump"]["flown"] >= spec["minimum_active_jump_liftoff_fraction"], f"{seed}: active liftoff rate too low")
        require(fractions["jump"]["succeeded"] >= spec["minimum_active_jump_success_fraction"], f"{seed}: active stable-landing rate too low")
        require(fractions["airborne"]["succeeded"] >= spec["minimum_airborne_success_fraction"], f"{seed}: airborne stable-landing rate too low")
        results.append(dict(seed=seed, fractions=fractions, sampled_closed_chain_gap_m=gap))
    return dict(passed=not errors, errors=errors, seeds=results, qualification=False,
        scope=spec["scope"], permits_fresh_engineering_run=not errors)
