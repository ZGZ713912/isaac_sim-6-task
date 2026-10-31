"""Cold deterministic four-class belly evaluation; no optimizer or learner calls."""
from __future__ import annotations

from copy import deepcopy
import math
import random

import numpy as np
import torch


def cases_for(env, config):
    cases = []
    for i,pose in enumerate(env.poses):
        r = pose.get("recovery",{})
        if (r.get("family"),r.get("split")) != ("belly_down","holdout"):
            continue
        for heading in config["evaluation_headings_rad"]:
            cases.append(dict(role="prone",recovery=True,pose_index=i,heading_rad=heading,
                birth_height=pose["root_position_m"][2],pitch_deg=0.,target=[0.,0.,.305],command=[0.,0.,.305],
                family="belly_down",leg_class=r["leg_class"],split="holdout"))
    if not cases or len(cases) > env.num_envs:
        raise ValueError("Mastery evaluation needs enough environments for all independent four-class trials")
    return cases


@torch.inference_mode()
def evaluate(env, actor, config, *, stop_requested=lambda:False):
    """Restore RNGs/mode, then reset training; the caller refreshes its observation.

    The test deliberately ends old training episodes at an empty rollout. Their
    partial episode logger totals must be discarded by the rollout-boundary hook.
    Learner parameters, optimizer, policy normalizer and PPO storage are untouched.
    """
    cases = cases_for(env,config)
    n = len(cases)
    old_cases,old_fixed = env.evaluation_cases,env.fixed_cases
    was_training = actor.training
    rng = (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        env.generator.get_state(),env.domain.rng.get_state(),np.random.get_state(),random.getstate(),
        env.recovery_sampler.rng.getstate(),deepcopy(env.recovery_sampler.class_cursor))
    rows = [dict(family=c["family"],leg_class=c["leg_class"],split="holdout",pose_index=c["pose_index"],
        heading_rad=c["heading_rad"],completed=False,terminated=False,active_time_s=None,
        post_completion_hold_s=0.,closure_peak_m=0.,passed=False) for c in cases]
    active = torch.ones(n,dtype=torch.bool,device=env.device)
    hold = torch.zeros(n,device=env.device)
    try:
        actor.eval()
        env.evaluation_cases = [cases[i % n] for i in range(env.num_envs)]
        env.reset(torch.arange(env.num_envs,device=env.device))
        obs = env.get_observations()
        horizon = math.ceil((.15+config["max_active_recovery_seconds"]+config["post_completion_hold_seconds"]+.1)/env.policy_dt)
        for _ in range(horizon):
            if stop_requested():
                from train_scratch_standing import TrainingStopped
                raise TrainingStopped()
            obs,reward,done,extra = env.step(actor(obs))
            d = extra["diagnostics"]
            stable = d["stable"][:n] & d["recovered"][:n] & ~d["terminated"][:n]
            hold.copy_(torch.where(stable & active,hold+env.policy_dt,0.))
            for lane,row in enumerate(rows):
                if not bool(active[lane]):
                    continue
                row["closure_peak_m"] = max(row["closure_peak_m"],float(d["gap"][lane]))
                row["terminated"] |= bool(d["terminated"][lane])
                if bool(d["recovery_completed_event"][lane]):
                    row["completed"] = True
                    row["active_time_s"] = float(d["recovery_active_seconds"][lane])
                row["post_completion_hold_s"] = max(row["post_completion_hold_s"],float(hold[lane]))
                row["passed"] = (row["completed"] and not row["terminated"]
                    and row["active_time_s"] <= config["max_active_recovery_seconds"]
                    and row["closure_peak_m"] <= .003
                    and row["post_completion_hold_s"]+1e-6 >= config["post_completion_hold_seconds"])
                if row["passed"] or bool(done[lane]):
                    active[lane] = False
            if not bool(active.any()):
                break
        return dict(cases=rows,policy_steps=_+1,physics_dt_s=env.dt,policy_dt_s=env.policy_dt,
            scope="curriculum_admission_only",new_PPO=0)
    finally:
        env.evaluation_cases,env.fixed_cases = old_cases,old_fixed
        torch.set_rng_state(rng[0])
        if rng[1]: torch.cuda.set_rng_state_all(rng[1])
        env.generator.set_state(rng[2]); env.domain.rng.set_state(rng[3])
        np.random.set_state(rng[4]); random.setstate(rng[5])
        env.recovery_sampler.rng.setstate(rng[6]); env.recovery_sampler.class_cursor = rng[7]
        env.reset(torch.arange(env.num_envs,device=env.device))
        actor.train(was_training)
