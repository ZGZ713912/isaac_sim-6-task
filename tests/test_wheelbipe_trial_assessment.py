from copy import deepcopy
import json
from pathlib import Path

import pytest

from wheeled_tasks.wheelbipe_port.trial_assessment import assess


def receipts():
    spec = json.loads((Path(__file__).resolve().parents[1]/"contracts/wheelbipe_jump_trial_v1.json").read_text())
    w = dict(vx_m_s=dict(mean=0.), yaw_rad_s=dict(mean=0.), height_error_m=dict(abs_mean=.01), tilt_deg=dict(mean=1.))
    base = dict(status="complete", ppo_updates=0, checkpoint_sha256="before", arguments=spec["legacy_evaluation"],
        cases=[dict(case=n, command=[0., 0., .305], resets=0, windows=dict(tail_half=deepcopy(w)))
               for n in spec["legacy_cases"]])
    candidate = deepcopy(base)
    candidate["checkpoint_sha256"] = "after"
    def mixed(seed, after):
        args = {k:v for k,v in spec["mixed_evaluation"].items() if k!="seeds"}
        args.update(seed=seed, randomization_profile="nominal", randomization_strength=0., course_update=2000 if after else 0)
        return dict(status="complete", ppo_updates=0, optimizer_created=False, finite_native_states=True,
            checkpoint_sha256="after" if after else "before", arguments=args,
            max_constraint_gap_policy_sample_m=.001, allocated_envs=dict(jump=51, airborne=26),
            tasks={n:dict(resets=0, tracking_mean_absolute={k:0. for k in spec["maximum_added_error"]})
                for n in ("stand", "forward", "reverse", "height", "turn", "dash", "spin")},
            episode_outcomes={n:dict(completed=200, flown=180, succeeded=170)for n in ("jump", "airborne")})
    before, after = [[mixed(seed, flag)for seed in spec["mixed_evaluation"]["seeds"]]for flag in (False, True)]
    return base, candidate, before, after, spec


def test_successful_jump_and_retention_can_admit_fresh_engineering_run():
    result = assess(*receipts())
    assert result["passed"] and result["permits_fresh_engineering_run"]
    assert result["qualification"] is False


@pytest.mark.parametrize("failure", ["no_active_jump", "retention", "missing_seed", "too_few_episodes", "checkpoint", "gap"])
def test_failure_or_missing_evidence_cannot_admit_fresh(failure):
    b, a, bm, am, spec = receipts()
    if failure == "no_active_jump":
        am[0]["episode_outcomes"]["jump"].update(flown=0, succeeded=0)
    elif failure == "retention":
        am[0]["tasks"]["stand"]["tracking_mean_absolute"]["vx_error_m_s"] = .5
    elif failure == "missing_seed":
        am.pop()
    elif failure == "too_few_episodes":
        am[0]["episode_outcomes"]["jump"]["completed"] = 2
    elif failure == "checkpoint":
        am[0]["checkpoint_sha256"] = "another"
    elif failure == "gap":
        am[0]["max_constraint_gap_policy_sample_m"] = .01
    result = assess(b, a, bm, am, spec)
    assert result["errors"] and not result["permits_fresh_engineering_run"]
