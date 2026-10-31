"""Cold skill tests from native truth, isolated from PPO and training rewards."""
from __future__ import annotations

from copy import deepcopy
import math
import random

import numpy as np
import torch

from wheeled_tasks.chassis.global_curriculum import assessment_descriptors


def bind_case(env, row, seconds):
    row = deepcopy(row)
    pose = env.nominal_pose_index if row.get("nominal_birth") else env.height_pose[row["birth_height"]]
    row.update(pose_index=pose,command=[0.,0.,row["birth_height"]],evaluation_seconds=seconds)
    return row


def assess_trace(arrays, cases, initial_xy, limits, dt, horizon):
    """Keep failed/missing trials in the denominator; report actual body spin."""
    rows = []
    for lane,case in enumerate(cases):
        n = int(arrays["valid"][:,lane].sum())
        a = {k:v[:n,lane] for k,v in arrays.items()}
        finite = n > 0 and all(np.isfinite(v).all() for k,v in a.items() if k != "valid")
        r = dict(case_id=case["case_id"],source="cold_curriculum",role=case["role"],
            curriculum_skill=case["curriculum_skill"],curriculum_level=case["curriculum_level"],
            heading_rad=case["heading_rad"],target=case["target"],samples=n,passed=False)
        if not finite:
            r["checks"] = dict(finite_native_samples=False);rows.append(r);continue
        t,c,v,w = a["time"],a["command"],a["velocity"],a["omega"]
        xy = a["position"][:,:2]-initial_xy[lane]
        distance = np.linalg.norm(xy,axis=-1)
        tilt = np.rad2deg(np.arccos(np.clip(-a["gravity"][:,2],-1.,1.)))
        r.update(full_horizon=n == horizon,terminated=bool(a["terminated"].any()),
            closure_peak_m=float(a["gap"].max()),height_mae_m=float(np.abs(a["height"]-c[:,2]).mean()),
            tilt_peak_deg=float(tilt.max()),geometric_air_run_ms=float(a["air_run"].max()*1000.),
            xy_peak_m=float(distance.max()))
        checks = dict(full_horizon=r["full_horizon"],not_terminated=not r["terminated"],
            closure=r["closure_peak_m"] <= limits["closure_peak_m"],
            height=r["height_mae_m"] <= limits["height_mae_m"],tilt=r["tilt_peak_deg"] <= limits["tilt_peak_deg"],
            visible_air=r["geometric_air_run_ms"] <= limits["geometric_air_run_ms"]+1e-5)
        skill = case["curriculum_skill"]
        quiet = skill in ("standing","static_height")
        chosen = t >= 2. if quiet else (t >= 5.05) & (t < (12. if skill == "braking" else 13.))
        checks["measurement_window"] = int(chosen.sum()) >= round(2./dt)
        if chosen.any():
            vxerr = float(np.abs(v[chosen,0]-c[chosen,0]).mean())
            werr = float(np.abs(w[chosen,2]-c[chosen,1]).mean())
            lateral = float(np.abs(v[chosen,1]).mean())
            r.update(vx_mae_m_s=vxerr,yaw_mae_rad_s=werr,lateral_velocity_mae_m_s=lateral)
            checks["lateral_velocity"] = lateral <= limits["lateral_velocity_mae_m_s"]
            if quiet:
                speed = float(np.sqrt(np.mean(np.sum(v[chosen,:2]**2,axis=-1))))
                r["quiet_speed_rms_m_s"] = speed
                checks.update(quiet_position=r["xy_peak_m"] <= limits["standing_xy_max_m"],
                              quiet_speed=speed <= limits["standing_speed_rms_m_s"],yaw=werr <= limits["yaw_mae_rad_s"])
            else:
                target_v,target_w = abs(case["target"][0]),abs(case["target"][1])
                checks.update(velocity=vxerr <= (min(limits["velocity_mae_m_s"],limits["velocity_relative_mae"]*target_v)
                    if target_v else limits["standing_speed_rms_m_s"]),
                    yaw=werr <= (min(limits["yaw_mae_rad_s"],limits["yaw_relative_mae"]*target_w)
                    if target_w else limits["yaw_mae_rad_s"]))
                if "spin" in skill:
                    sign = math.copysign(1.,case["target"][1])
                    rpm = float(np.min(w[chosen,2]*sign)*60./math.tau)
                    r["actual_signed_direction_min_rpm"] = rpm
                    checks.update(actual_spin=rpm >= .8*target_w*60./math.tau,
                                  spin_center=r["xy_peak_m"] <= limits["spin_center_max_m"])
                if skill in ("straight","moving_height","braking"):
                    h = case["heading_rad"]
                    path = float(np.abs(xy@np.array([-math.sin(h),math.cos(h)])).max())
                    r["lateral_path_peak_m"] = path
                    checks["straight_path"] = path <= limits["straight_lateral_path_m"]
        if "height" in skill:
            height_active = (t >= (7.05 if case.get("moving_height") else 5.05)) & (t < 13.)
            changing = np.ptp(c[:,2]) > .02
            checks["actual_height_request"] = bool(changing and height_active.sum()*dt >= 2.)
            if height_active.any():
                r["active_height_mae_m"] = float(np.abs(a["height"][height_active]-c[height_active,2]).mean())
                checks["active_height"] = r["active_height_mae_m"] <= limits["height_mae_m"]
        if skill == "braking":
            after = np.flatnonzero(t >= 12.)
            checks["braking_window"] = bool(len(after) and t[-1] >= 19.9)
            if len(after):
                speed = np.linalg.norm(v[:,:2],axis=-1)
                bad = after[speed[after] > limits["brake_final_speed_m_s"]]
                stopped = (int(bad[-1])+1 if len(bad) else int(after[0]))
                delay = float(t[stopped]-12.) if stopped < n else None
                overshoot = float(np.linalg.norm(a["position"][after,:2]-a["position"][after[0],:2],axis=-1).max())
                r.update(brake_stop_time_s=delay,brake_overshoot_m=overshoot)
                checks.update(brake_stops=delay is not None and delay <= limits["brake_stop_seconds"],
                              brake_overshoot=overshoot <= limits["brake_overshoot_m"])
        r["checks"] = {k:bool(v) for k,v in checks.items()}
        r["passed"] = all(r["checks"].values())
        rows.append(r)
    return rows


@torch.inference_mode()
def evaluate_skill(env, actor, course, skill, level, *, stop_requested=lambda:False):
    seconds = course.config["standing_seconds"] if skill == "standing" else course.config["motion_seconds"]
    cases = [bind_case(env,r,seconds) for r in assessment_descriptors(course.config,skill,level)]
    if len(cases) > env.num_envs:raise ValueError("Cold skill matrix exceeds environment count")
    n = len(cases);horizon = round(seconds/env.policy_dt)
    old_cases,old_fixed,old_roles = env.evaluation_cases,env.fixed_cases,list(env.course_roles)
    mode = actor.training
    rng = (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        env.generator.get_state(),env.domain.rng.get_state(),np.random.get_state(),random.getstate(),
        env.recovery_sampler.rng.getstate(),deepcopy(env.recovery_sampler.class_cursor))
    trace = {};valid = torch.ones(n,dtype=torch.bool,device=env.device)
    try:
        actor.eval()
        env.evaluation_cases = [deepcopy(cases[i%n]) for i in range(env.num_envs)]
        env.reset(torch.arange(env.num_envs,device=env.device))
        initial = (env.robot.data.root_link_pose_w.torch[:n,:2]-env.origins[:n,:2]).cpu().numpy().copy()
        obs = env.get_observations()
        for _ in range(horizon):
            if stop_requested():
                from train_scratch_standing import TrainingStopped
                raise TrainingStopped()
            obs,_,done,extra = env.step(actor(obs))
            d = extra["diagnostics"]
            values = {k:d[v][:n] for k,v in dict(time="time",position="local_position",velocity="velocity",
                omega="omega",gravity="gravity",height="height",gap="gap",terminated="terminated",
                command="interval_requested_command",air_run="wheel_air_run_peak").items()}
            values["valid"] = valid.clone()
            for k,v in values.items():trace.setdefault(k,[]).append(v.cpu().numpy().copy())
            valid &= ~done[:n]
            if not bool(valid.any()):break
        arrays = {k:np.stack(v) for k,v in trace.items()}
        rows = assess_trace(arrays,cases,initial,course.config["limits"],env.policy_dt,horizon)
        return dict(cases=rows,definitions=cases,initial_xy=initial.tolist(),scope="curriculum_admission_only",
                    new_PPO=0,policy_steps=len(arrays["time"])),arrays
    finally:
        env.evaluation_cases,env.fixed_cases,env.course_roles = old_cases,old_fixed,old_roles
        torch.set_rng_state(rng[0])
        if rng[1]:torch.cuda.set_rng_state_all(rng[1])
        env.generator.set_state(rng[2]);env.domain.rng.set_state(rng[3])
        np.random.set_state(rng[4]);random.setstate(rng[5])
        env.recovery_sampler.rng.setstate(rng[6]);env.recovery_sampler.class_cursor = rng[7]
        env.reset(torch.arange(env.num_envs,device=env.device))
        actor.train(mode)


def complex_cases(env):
    cases = []
    for i,p in enumerate(env.poses):
        r = p.get("recovery",{})
        if r.get("split") != "holdout" or r.get("family") == "belly_down":continue
        cases.append(dict(role="prone",recovery=True,pose_index=i,heading_rad=0.,family=r["family"],
            leg_class=r["leg_class"],split="holdout",pitch_deg=0.,birth_height=p["root_position_m"][2],
            target=[0.,0.,.305],command=[0.,0.,.305]))
    return cases
