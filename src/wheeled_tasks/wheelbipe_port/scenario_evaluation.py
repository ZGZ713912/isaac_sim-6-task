"""Read-only V9 scenario grids, physical episode summaries and retention warnings."""
from __future__ import annotations

from copy import deepcopy
import itertools
import math

import torch


def wilson_lower(successes, count):
    if not count:
        return None
    z,p = 1.959963984540054,successes/count
    return (p+z*z/(2*count)-z*math.sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count)


def scenario_grid(suite, *, update=20000):
    cells = []
    def add(kind,task,**kw):
        cells.append(dict(kind=kind,task=task,**kw))
    if suite == "slope_spin":
        for angle,rpm,height in itertools.product((0,6,10,12),(-100,-80,80,100),(.28,.305,.36)):
            add(suite,"slope",angle_deg=angle,rpm=rpm,height=height,mode="spin")
    elif suite == "slope_still":
        for angle,mode,speed in itertools.product((0,6,10,12),("stand","drive"),(.3,.8)):
            add("slope_stand" if mode == "stand" else "slope_drive","slope",angle_deg=angle,mode=mode,speed=speed,height=.305)
    elif suite == "step_down":
        for rise,speed,yaw in itertools.product((.20,.25),(.5,1.,1.5),(0,-15,15)):
            add(suite,"step_down",rise=rise,speed=speed,yaw_deg=yaw,height=.305)
    elif suite == "bump":
        for rise,speed,yaw,reverse in itertools.product((.03,.05,.07),(.5,1.,1.5),(0,-30,30),(False,True)):
            add(suite,"bump",rise=rise,speed=speed,yaw_deg=yaw,reverse=reverse,height=.33)
    elif suite == "slope_jump":
        for angle,speed,yaw in itertools.product((0,6,9,12),(0.,.5),range(8)):
            add(suite,"slope",angle_deg=angle,speed=speed,yaw_deg=yaw*45,apex=.46,mode="jump",height=.305)
    elif suite == "balls":
        for friction,mode in itertools.product((.2,.35,.5),("stand","height","spin80","spin100","shuttle03","shuttle06","brake")):
            add(suite,"balls",friction=friction,mode=mode,coverage=.3,height=.305)
    elif suite == "motion":
        for height,speed in itertools.product((.23,.305,.42),(0.,-.05,.05,-.1,.1,-.25,.25,-.5,.5,-1.,1.,-2.7,2.7)):
            task = "forward" if speed > 0. else "reverse" if speed < 0. else "stand"
            add(task,task,command=[speed,0.,0.],height=height,motion_probe=True)
    elif suite == "regression":
        for kind,task,cmd in (("stand","stand",[0,0,0]),("forward","forward",[1,0,0]),
                ("reverse","reverse",[-1,0,0]),("spin","spin",[0,0,100*math.pi/30]),("jump","jump",None)):
            add(kind,task,command=cmd,height=.305,apex=.46)
        for over in (True,False):
            add("platform","platform",rise=.2,speed=.75,over=over,height=.305)
    elif suite == "retention":
        cells = scenario_grid("regression",update=update)
        if update >= 9000:
            for rise in (.20,.25): add("step_down","step_down",rise=rise,speed=1.,yaw_deg=0,height=.305)
            add("slope_spin","slope",angle_deg=10,rpm=100,height=.305,mode="spin")
        if update >= 14000:
            add("bump","bump",rise=.05,speed=1.,yaw_deg=0,reverse=False,height=.33)
        if update >= 15000:
            add("slope_jump","slope",angle_deg=9,speed=0.,yaw_deg=0,apex=.46,mode="jump",height=.305)
            add("balls","balls",friction=.35,mode="spin100",coverage=.3,height=.305)
    else:
        raise ValueError("Unknown V9 evaluation suite: "+suite)
    for index,cell in enumerate(cells):
        cell["name"] = cell["kind"]+"_"+"_".join(f"{k}={v}" for k,v in cell.items() if k not in ("kind","task","name"))
        cell["cell_index"] = index
    return cells


def scenario_override(spec, cell, target_envs):
    """Keep one control lane per other task and exactly target_envs test lanes."""
    task = cell["task"]
    count = target_envs+len(spec["tasks"])-1
    patch = dict(task_fractions={name:(target_envs if name == task else 1)/count for name in spec["tasks"]})
    duration = 8. if task in ("slope","bump") else 6.
    if cell.get("motion_probe"):
        duration = 10.
    if task == "balls": duration = 10. if cell["mode"] in ("stand","height") else (7. if cell["mode"].startswith("spin") else 6.)
    evaluation = dict(target=task,duration_s=duration,cell=cell,
        start_s=1.5 if cell["kind"] == "slope_spin" else .3,
        end_s=6.5 if cell["kind"] == "slope_spin" else duration)
    if task == "balls" and cell["mode"].startswith("spin"):
        evaluation.update(start_s=1.,end_s=6.)
    patch["evaluation"] = evaluation
    if task in ("slope","step_down","bump","balls"):
        cfg = deepcopy(spec[task])
        cfg["height_range_m"] = [cell["height"]]*2
        cfg["episode_seconds"] = duration
        if task == "slope":
            angle = cell["angle_deg"]
            cfg["lane_angles_deg"] = [angle]
            cfg["unlock"] = dict(start_update=0,full_update=0,start_angle_deg=angle,final_angle_deg=angle)
            mode = cell["mode"]
            cfg["mode_fractions"] = dict(spin=float(mode == "spin"),stand=float(mode == "stand"),drive=float(mode in ("drive","jump")))
            rpm = abs(cell.get("rpm",100))
            cfg["spin"].update(rpm_start=rpm,rpm_final=rpm,rpm_ramp=[0,0],late_update=1000000,review_fraction_of_max=[1.,1.])
            cfg["drive"].update(speed_range_m_s=[cell.get("speed",.3)]*2,arc_probability=0.)
            cfg["jump"].update(probability_ramp=[0,0],probability_final=.30 if mode == "jump" else 0.,
                angle_limit_ramp=[0,0],angle_limit_start_deg=angle,angle_limit_final_deg=angle,
                moving_fraction=float(cell.get("speed",0.) > 0),moving_speed_m_s=[max(cell.get("speed",0.),.3)]*2)
        elif task == "step_down":
            rise = cell["rise"]
            cfg.update(lane_heights_m=[rise],speed_min_m_s=cell["speed"],speed_max_start_m_s=cell["speed"],speed_max_final_m_s=cell["speed"],heading_offset_deg=0.)
            cfg["unlock"] = dict(start_update=0,full_update=0,start_height_m=rise,final_height_m=rise)
        elif task == "bump":
            rise = cell["rise"]
            cfg.update(lane_heights_m=[rise],lane_yaws_deg=[cell["yaw_deg"]],speed_range_m_s=[cell["speed"]]*2,
                reverse_fraction=float(cell["reverse"]),arc_fraction=0.)
            cfg["unlock"] = dict(start_update=0,full_update=0,start_height_m=rise,final_height_m=rise)
        else:
            mode = cell["mode"]
            cfg.update(solver_position_iterations=cfg.get("eval_solver_position_iterations",cfg["solver_position_iterations"]),patch_side_m=spec["balls"]["eval_patch_side_m"],friction_variants=[cell["friction"]],coverage_start=.3,coverage_final=.3,coverage_ramp=[0,0],coverage_review_probability=0.)
            cfg["mode_fractions"] = dict(stand=float(mode in ("stand","height")),spin=float(mode.startswith("spin")),shuttle=float(mode.startswith("shuttle") or mode == "brake"))
            if mode.startswith("spin"): cfg["spin_rpm"] = [float(mode[4:])]*2
            if mode.startswith("shuttle"): cfg["shuttle"]["speed_m_s"] = [.3 if mode.endswith("03") else .6]*2
        patch[task] = cfg
    elif task == "platform":
        cfg = deepcopy(spec[task]); rise = cell["rise"]
        cfg.update(lane_heights_m=[rise],speed_range_m_s=[cell["speed"]]*2,over_fraction=float(cell["over"]))
        cfg["unlock"] = dict(start_update=0,full_update=0,start_height_m=rise,final_height_m=rise)
        patch[task] = cfg
    if "apex" in cell:
        patch["jump"] = dict(apex_start_m=[cell["apex"]]*2,apex_final_m=[cell["apex"]]*2)
    return patch,count


class ScenarioRecorder:
    """Samples at _get_dones, before DirectRLEnv resets any completed lane."""
    def __init__(self, env):
        self.env,self.cfg = env,env.cfg.port_mixed_spec["evaluation"]
        self.mask = env.task_schedule.mask(self.cfg["target"])
        n,d = env.num_envs,env.device
        self.anchor = torch.zeros(n,2,device=d)
        self.tail_anchor = torch.zeros_like(self.anchor)
        self.valid = torch.zeros(n,device=d,dtype=torch.bool)
        self.tail_valid = torch.zeros_like(self.valid)
        self.values = {key:torch.zeros(n,device=d) for key in ("samples","roll_square","pitch_square","max_roll_deg","max_pitch_deg",
            "over3","over6","radius_max_m","speed_error_sum","rpm_sum","tail_samples","tail_distance_m","base_contact",
            "strip_success","strip_failure","height_dev_max_m","slip_sum","post_landing_slide_m",
            "speed_signed_error_sum","speed_actual_sum","speed_command_sum","wrong_direction_samples")}
        self.landing_xy = torch.zeros_like(self.anchor)
        self.slide_seen = torch.zeros_like(self.valid)
        self.values["speed_ratio_min"] = torch.ones(n,device=d)
        self.values["tail_rpm_min"] = torch.full((n,),float("inf"),device=d)
        self.episodes = []

    def observe(self, terminated, timeout):
        e,c,v = self.env,self.cfg,self.values
        s,o = e.task_schedule,e.terrain_objectives
        age = e._mixed_age()
        xy = e.robot.data.root_pos_w[:,:2]-e.terrain.env_origins[:,:2]
        active = self.mask & (age >= c["start_s"]) & (age <= c["end_s"])
        first = active & ~self.valid
        self.anchor[first] = xy[first]; self.valid |= active
        r,p = roll_pitch(e)
        tilt = torch.maximum(r.abs(),p.abs())
        speed = o.speed if o is not None else e.robot.data.root_lin_vel_b[:,0]
        direction = torch.where(e.command[:,2].abs() > 1e-6,e.command[:,2].sign(),s.terrain.sign if s.terrain is not None else 1.)
        rpm = e.robot.data.root_ang_vel_w[:,2]*direction*30/math.pi
        v["samples"] += active
        v["roll_square"] += r.square()*active; v["pitch_square"] += p.square()*active
        v["max_roll_deg"] = torch.maximum(v["max_roll_deg"],r.abs()*active)
        v["max_pitch_deg"] = torch.maximum(v["max_pitch_deg"],p.abs()*active)
        v["over3"] += (tilt > 3.) & active; v["over6"] += (tilt > 6.) & active
        v["radius_max_m"] = torch.maximum(v["radius_max_m"],(xy-self.anchor).norm(dim=-1)*active)
        v["speed_error_sum"] += (speed-e.command[:,0]).abs()*active
        v["speed_signed_error_sum"] += (speed-e.command[:,0])*active
        v["speed_actual_sum"] += speed*active
        v["speed_command_sum"] += e.command[:,0]*active
        v["wrong_direction_samples"] += (speed*e.command[:,0] < 0.)*active
        moving = active & (e.command[:,0].abs() > .01)
        ratio = speed/e.command[:,0].where(e.command[:,0].abs() > .01,torch.ones_like(speed))
        v["speed_ratio_min"] = torch.where(moving,torch.minimum(v["speed_ratio_min"],ratio),v["speed_ratio_min"])
        v["rpm_sum"] += rpm*active
        tail = active & (age >= c["end_s"]-2.)
        first_tail = tail & ~self.tail_valid
        self.tail_anchor[first_tail] = xy[first_tail]; self.tail_valid |= tail
        v["tail_samples"] += tail
        v["tail_distance_m"] = torch.where(tail,(xy-self.tail_anchor).norm(dim=-1),v["tail_distance_m"])
        v["tail_rpm_min"] = torch.where(tail,torch.minimum(v["tail_rpm_min"],rpm),v["tail_rpm_min"])
        base_ids = e._find_contact_sensor_indices(["base_link"])
        contact = e.contact_sensor.data.net_forces_w[:,base_ids].norm(dim=-1).amax(-1) > 1.
        v["base_contact"] = torch.maximum(v["base_contact"],contact.float()*self.mask)
        height = e.robot.data.root_pos_w[:,2]-e.terrain.env_origins[:,2]
        wheel_xy,_ = e._wheel_geometry()
        reference = height-s.support_height(wheel_xy).mean(-1)
        v["height_dev_max_m"] = torch.maximum(v["height_dev_max_m"],(reference-e.height_cmd).abs()*active)
        if o is not None:
            v["strip_success"] += o.bump.success_count
            v["strip_failure"] += o.bump.failure_count
            v["slip_sum"] += o.balls.slip_sum/o.balls.samples.clamp_min(1.)*active
            landed = self.mask & o.slope_landed
            first_land = landed & ~self.slide_seen
            self.landing_xy[first_land] = xy[first_land]
            self.slide_seen |= landed
            post = landed & (age >= o.slope_land_time+1.)
            downslope = torch.stack((e.task_schedule.terrain.slope_yaw.cos(),e.task_schedule.terrain.slope_yaw.sin()),-1)
            displacement = ((xy-self.landing_xy)*downslope).sum(-1)
            # Moving jumps retain the external velocity command. Remove its
            # declared travel before measuring unwanted downslope sliding.
            yaw = e.task_schedule.terrain.yaw
            desired = e.task_schedule.terrain.speed*torch.cos(yaw-e.task_schedule.terrain.slope_yaw)
            residual = (displacement-desired*(age-o.slope_land_time)).clamp_min(0.)
            v["post_landing_slide_m"] = torch.where(post,residual,v["post_landing_slide_m"])
        done = (terminated | timeout) & self.mask
        for i in done.nonzero().flatten().tolist():
            row = {key:float(value[i]) for key,value in v.items()}
            row.update(terminated=bool(terminated[i]),elapsed_s=float(age[i]))
            row["completed_window"] = bool(age[i] >= c["end_s"]-.025)
            row["jump_success"] = bool(e.jump_observer.succeeded[i]) and not bool(terminated[i])
            if c["cell"]["kind"] == "platform" and not c["cell"]["over"]:
                row["jump_success"] = bool(e.platform_observer.drop_succeeded[i]) and not bool(terminated[i])
            if o is not None:
                row.update(step_down_success=bool(o.step_down.success[i]) and not bool(terminated[i]),
                    window_tilt_max_deg=float(o.step_down.max_tilt_w[i]),window_speed_ratio_min=float(o.step_down.min_ratio_w[i]),
                    landing_force_peak_bw=float(o.step_down.force_peak[i]/o.step_down.weight[i]),
                    slope_jump_tilt_max_deg=float(o.slope_jump_tilt[i]),landing_offset_m=float(o.slope_jump_offset[i]))
                row["post_landing_window_complete"] = bool(o.slope_landed[i] and age[i] >= o.slope_land_time[i]+1.)
            self.episodes.append(row)
        if done.any():
            for key,value in v.items():
                value[done] = float("inf") if key == "tail_rpm_min" else (1. if key == "speed_ratio_min" else 0.)
            self.valid[done] = self.tail_valid[done] = False
            self.slide_seen[done] = False

    def summary(self):
        cell = self.cfg["cell"]
        rows = self.episodes
        count = len(rows)
        missing = []
        sample_count = sum(r["samples"] for r in rows)
        def mean_sum(key): return sum(r[key] for r in rows)/sample_count if sample_count else None
        def maximum(key): return max((r[key] for r in rows),default=None)
        def minimum(key):
            values = [r[key] for r in rows if math.isfinite(r[key])]
            return min(values) if values else None
        metrics = dict(roll_rms_deg=math.sqrt(mean_sum("roll_square")) if sample_count else None,
            pitch_rms_deg=math.sqrt(mean_sum("pitch_square")) if sample_count else None,
            roll_max_deg=maximum("max_roll_deg"),pitch_max_deg=maximum("max_pitch_deg"),
            tilt_over_3deg_fraction=mean_sum("over3"),tilt_over_6deg_fraction=mean_sum("over6"),
            radius_max_m=maximum("radius_max_m"),speed_error_mean_m_s=mean_sum("speed_error_sum"),
            speed_signed_error_mean_m_s=mean_sum("speed_signed_error_sum"),
            speed_actual_mean_m_s=mean_sum("speed_actual_sum"),speed_command_mean_m_s=mean_sum("speed_command_sum"),
            wrong_direction_fraction=mean_sum("wrong_direction_samples"),
            rpm_abs_mean=mean_sum("rpm_sum"),rpm_tail_min=minimum("tail_rpm_min"),
            drift_tail_m_s=max((r["tail_distance_m"]/max(r["tail_samples"]*self.env.step_dt,self.env.step_dt) for r in rows),default=None),
            height_dev_max_m=maximum("height_dev_max_m"),speed_ratio_min=minimum("speed_ratio_min"),
            base_contact_episodes=sum(bool(r["base_contact"]) for r in rows),terminated_episodes=sum(r["terminated"] for r in rows),
            wheel_slip_mean_m_s=mean_sum("slip_sum"))
        for key in ("window_tilt_max_deg","window_speed_ratio_min","landing_force_peak_bw","slope_jump_tilt_max_deg","landing_offset_m","post_landing_slide_m"):
            values = [r[key] for r in rows if key in r]
            metrics[key] = (min(values) if key == "window_speed_ratio_min" else max(values)) if values else None
        success = sum(self.passed_episode(cell,r) for r in rows)
        metrics["success_fraction"] = success/count if count else None
        metrics["wilson_95_lower"] = wilson_lower(success,count)
        strips = sum(r["strip_success"]+r["strip_failure"] for r in rows)
        metrics["strip_success_fraction"] = sum(r["strip_success"] for r in rows)/strips if strips else None
        if not count: missing.append("completed episodes")
        if not sample_count: missing.append("physical samples in the requested window")
        if cell["kind"] == "bump" and not strips: missing.append("completed strip crossings")
        # These require post-landing tracking or a paired baseline, rather than
        # pretending that a partial measurement qualifies the full acceptance.
        required = .95 if cell["kind"] in ("step_down","bump") and cell.get("rise",0) <= .2 else .90
        if cell["kind"] == "step_down": required = .95 if cell["rise"] == .2 else .90
        if cell["kind"] == "bump": required = .95 if cell["rise"] <= .05 else .85
        if cell["kind"] == "slope_jump": required = .85 if cell["angle_deg"] <= 9 else (.70 if cell["speed"] > 0 else .75)
        if cell["kind"] == "balls": required = .98
        return dict(cell=cell,episodes=count,successes=success,metrics=metrics,missing=missing,
            passed=bool(not missing and count >= 64 and metrics["success_fraction"] >= required),
            behavior_qualified=False,interpretation="Per-cell simulation evidence; all permanent capabilities and hardware qualification remain separate")

    @staticmethod
    def passed_episode(cell,r):
        if r["terminated"] or r["base_contact"] or not r["completed_window"] or not r["samples"]:
            return False
        tilt = max(r["max_roll_deg"],r["max_pitch_deg"])
        kind = cell["kind"]
        if kind == "step_down":
            return r.get("step_down_success",False) and r["window_tilt_max_deg"] <= 8 and r["window_speed_ratio_min"] >= .7 and r["landing_force_peak_bw"] <= 3.5
        if kind == "bump":
            return r["strip_success"] > 0 and r["strip_failure"] == 0 and tilt <= 5 and r["height_dev_max_m"] <= .03 and r["speed_ratio_min"] >= .7
        if kind == "slope_spin":
            angle = cell["angle_deg"]; target = abs(cell["rpm"])
            return (r["radius_max_m"] <= (.15 if angle == 12 else .10) and tilt <= (4 if angle == 12 else 3)
                and math.sqrt(r["roll_square"]/r["samples"]) <= 1 and math.sqrt(r["pitch_square"]/r["samples"]) <= 1
                and r["rpm_sum"]/r["samples"] >= .9*target and r["tail_rpm_min"] >= (80 if target == 100 else .9*target)
                and r["tail_distance_m"]/max(r["tail_samples"]*.02,.02) <= .02)
        if kind in ("slope_stand","stand"): return r["radius_max_m"] <= .05 and tilt <= 2.5
        if kind in ("slope_drive","forward","reverse"): return r["speed_error_sum"]/r["samples"] <= .1 and tilt <= 3
        if kind in ("slope_jump","jump","platform"):
            return r["jump_success"] if kind != "slope_jump" else (r["jump_success"] and r["slope_jump_tilt_max_deg"] <= 8
                and r["post_landing_window_complete"] and r["post_landing_slide_m"] <= .05
                and (cell["speed"] > 0 or r["landing_offset_m"] <= .15))
        if kind == "balls":
            mode = cell["mode"]
            if mode in ("stand","height"): return r["radius_max_m"] <= .05 and tilt <= 3
            if mode.startswith("spin"): return r["radius_max_m"] <= .20 and tilt <= 5 and r["rpm_sum"]/r["samples"] >= 70
            return tilt <= 5 and r["speed_error_sum"]/r["samples"] <= .15
        return tilt <= 5


def roll_pitch(env):
    from .terrain_objectives import roll_pitch_deg
    return roll_pitch_deg(env.robot.data.projected_gravity_b)


def retention_warnings(history, current):
    """Compare two consecutive reviews against the earlier best for each cell."""
    warnings = []
    for row in current:
        name,metrics = row["cell"]["name"],row["metrics"]
        previous = [item for review in history for item in review if item["cell"]["name"] == name
            and item.get("domain") == row.get("domain") and item.get("seed") == row.get("seed")]
        if len(previous) < 2: continue
        for key,amount,direction in (("success_fraction",.15,-1),("speed_error_mean_m_s",1.5,1),
                                     ("radius_max_m",1.5,1),("roll_max_deg",2.,1),("pitch_max_deg",2.,1)):
            values = [r["metrics"].get(key) for r in previous]
            values = [v for v in values if v is not None]
            value,last = metrics.get(key),previous[-1]["metrics"].get(key)
            if not values or value is None or last is None: continue
            best = max(values) if direction < 0 else min(values)
            bad = (lambda x: best-x >= amount) if direction < 0 else ((lambda x:x > max(best,1e-6)*amount) if "error" in key or "radius" in key else (lambda x:x-best > amount))
            if bad(value) and bad(last): warnings.append(dict(cell=name,metric=key,best=best,previous=last,current=value))
    return warnings
