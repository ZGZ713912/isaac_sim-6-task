"""Case catalogue and physical-unit analysis for a fresh unified actor.

Importable without Isaac. This is a declared future evaluation protocol, not
evidence that a policy can already perform these tasks.
"""
import math
import random
from copy import deepcopy

import numpy as np


def configure_protocol(cfg, design, task, spec, *, case_count, seed):
    """Declare all evaluation overrides before independent design preflight.

    Preserve the native controllers, motor/chain guards, real-fall detector and
    actual stable-completion predicate. The run does no optimization.
    """
    if not 1 <= case_count <= 256 or cfg["physics_dt"] != .001 or cfg["policy_dt"] != .005:
        raise ValueError("Full suite needs the isolated native1kHz/200Hz mechanism")
    if task.get("native_origin_mode") != "isolated_overlap_v1":
        raise ValueError("Refuse to mix long straight paths with other robots")
    world = spec["flat_test_world"]
    if (world["isolated_origins"] is not True or not cfg.get("full_rl_isolated_origins")
            or not cfg.get("flat_triangle_mesh") or not cfg.get("full_rl_square_ground")):
        raise ValueError("Full suite needs isolated flat mesh worlds")
    half, width = world["floor_half_length_m"], world["floor_width_m"]
    bx, by = world["boundary_xy_m"]
    if not (0 < bx < half and 0 < by < width/2 and 4 <= width <= 128):
        raise ValueError("Declared boundary must stay inside the native floor")
    if bx <= spec["seconds"] * max(abs(v) for v in spec["high_speeds_m_s"]) + 5.:
        raise ValueError("Straight request would outrun the declared test world")
    cfg_changes = dict(target_num_envs=case_count, episode_seconds=float(spec["seconds"]),
        ppo_minibatch_samples=case_count*cfg["num_steps_per_env"]//cfg["num_mini_batches"],
        flat_half_length_m=half, flat_floor_width_m=width, record_diagnostics=True)
    task_changes = dict(episode_seconds=float(spec["seconds"]), boundary_xy_m=[bx, by])
    before = dict(config={key: deepcopy(cfg.get(key)) for key in cfg_changes},
                  task={key: deepcopy(task.get(key)) for key in task_changes})
    cfg.update(deepcopy(cfg_changes))
    task.update(deepcopy(task_changes))
    row = design["stages"]["speed"]["roles"]["train"]
    for key,value in cfg_changes.items():
        if key in row["contract"]:
            row["contract"][key] = deepcopy(value)
    row["runtime"]["seed"] = seed
    row["invocation"] = dict(num_envs=case_count, seed=dict(rule="fixed",value=seed))
    overrides = dict(before=before, after=dict(config=cfg_changes,task=task_changes),
        initialization="declared validated pose-bank cases; state writes during reset only",
        recovery_completion="unchanged native stable-hold event; never infer from pose alone",
        policy="same authenticated actor for every case; no teacher or action override",
        standing_failure_reset_wrapper=False, optimizer_constructed=False)
    design["review_basis"]["only_runtime_override"]["full_suite_protocol"] = overrides
    return overrides


def build_cases(bank, spec, seed):
    if (spec["schema"] != "wheel-leg-fresh-full-suite-v1" or type(spec["replicas"]) is not int
            or spec["replicas"] not in (1, 2)):
        raise ValueError("Unsupported full-suite specification")
    mapping = bank["diagnostic_height_starts"]["mapping"]
    poses = bank["poses"]
    heights = dict(zip(spec["standing_heights_m"], (mapping[k] for k in ("stand_low", "stand", "stand_high")), strict=True))
    if spec["standing_heights_m"] != [.23, .305, .42] or spec["seconds"] != 30:
        raise ValueError("Preserve all three standing heights and30s confirmation")
    if spec["spin_rpm"] != [80, 100]:
        raise ValueError("Preserve both80rpm and100rpm requests")
    all_cases = []
    for replica in range(spec["replicas"]):
        rng = random.Random(seed + replica*1009)
        cases = []

        def add(name, capability, command, *, pose_index=None, recovery=False, **kwargs):
            cases.append(dict(name=name, capability=capability,
                capabilities=[capability, "smoothness", "robustness"], replica=replica,
                pose_index=heights[command[2]] if pose_index is None else pose_index,
                recovery=recovery, command=list(command), pitch_deg=0., **kwargs))

        for family in ("front", "back", "left", "right", "inverted", "compound"):
            for asymmetric in (False, True):
                pool = [i for i, p in enumerate(poses) if p["family"] == family
                    and ("50_85" in p["leg_shape"] or "85_50" in p["leg_shape"]) == asymmetric
                    and max(abs(p["roll_deg"]), abs(p["pitch_deg"])) ==
                        (180 if family == "inverted" else 65 if family == "compound" else 90)]
                if not pool:
                    raise ValueError("Missing validated recovery geometry: " + family)
                pose = rng.choice(pool)
                suffix = "asymmetric" if asymmetric else "symmetric"
                for followup, command in (("stand", [0., 0., .305]), ("motion", [.5, 1. if asymmetric else -1., .305])):
                    add(f"getup_{family}_{suffix}_{followup}", "initial_recovery", command,
                        pose_index=pose, recovery=True, requested_spawn_family=family, followup=followup)
        for height in heights:
            for pitch in spec["standing_pitches_deg"]:
                add(f"stand_h{height:g}_p{pitch:g}", "standing", [0., 0., height])
                cases[-1]["pitch_deg"] = pitch
        for category, key in (("low_speed_straight", "low_speeds_m_s"), ("high_speed_straight", "high_speeds_m_s")):
            for speed in spec[key]:
                add(f"straight_{speed:g}", category, [speed, 0., .305])
        for speed_key, yaw_key in (("arc_speeds_m_s", "arc_yaw_rad_s"),
                                    ("high_arc_speeds_m_s", "high_arc_yaw_rad_s")):
            for speed in spec[speed_key]:
                for sign in (-1., 1.):
                    yaw = sign*spec[yaw_key]
                    add(f"arc_v{speed:g}_w{yaw:g}", "four_direction_arcs", [speed, yaw, .305])
        for speed in spec["arc_speeds_m_s"]:
            for sign in (-1., 1.):
                yaw = sign*spec["arc_yaw_rad_s"]
                add(f"arc_transition_v{speed:g}_w{yaw:g}", "four_direction_arcs", [speed, yaw, .305],
                    sequence=[[5., [speed, -yaw, .305]], [11., [-speed, yaw, .305]],
                              [18., [0., 0., .305]]])
        for rpm in spec["spin_rpm"]:
            for sign in (-1., 1.):
                for height in heights:
                    add(f"spin_{sign*rpm:g}rpm_h{height:g}", "stationary_spin",
                        [0., sign*rpm*2.*math.pi/60., height], requested_rpm=sign*rpm)
        sequence = lambda vx, yaw: [[time, [vx, yaw, h]] for time, h in spec["height_sequence"]]
        for start_height in (.23, .42):
            target_heights = [.42, .23, .305] if start_height == .23 else [.23, .42, .305]
            add(f"standing_height_from{start_height:g}", "standing_height_change", [0., 0., start_height],
                sequence=[[t, [0., 0., h]] for (t, _), h in zip(spec["height_sequence"], target_heights, strict=True)])
        for speed in (1., -.75):
            for yaw in (-1., 0., 1.):
                add(f"moving_height_v{speed:g}_w{yaw:g}", "moving_height_change", [speed, yaw, .305],
                    sequence=sequence(speed, yaw))
        for sign in (-1., 1.):
            yaw = sign*100.*2.*math.pi/60.
            add(f"spin_height_{sign*100:g}rpm", "spinning_height_change", [0., yaw, .305],
                requested_rpm=sign*100., sequence=sequence(0., yaw))
        for speed in spec["high_speeds_m_s"]:
            add(f"brake_{speed:g}", "high_speed_braking", [speed, 0., .305],
                sequence=[[spec["braking_at_s"], [0., 0., .305]]])
        for vx, yaw, label in ((.7, 0., "forward"), (-.7, 0., "reverse"),
                                (.7, 1., "arc_left"), (.7, -1., "arc_right"),
                                (0., 100.*2.*math.pi/60., "spin_positive"),
                                (0., -100.*2.*math.pi/60., "spin_negative")):
            for sign in (-1., 1.):
                add(f"fall_{label}_push{sign:g}", "moving_fall_recovery", [vx, yaw, .305],
                    push=True, push_sign=sign)
        names = [c["name"] for c in cases]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate full-suite case names")
        all_cases.extend(cases)
    return all_cases


def declared_commands(case, time):
    """Raw requested commands; never replace these with actuator projection."""
    time = np.asarray(time)
    result = np.repeat(np.asarray(case["command"], dtype=float)[None], len(time), axis=0)
    for start, command in case.get("sequence", []):
        result[time >= start] = command
    return result


def first_completion_sample(events, policy_dt=.005, physical_dt=.001):
    """First actual completion only; its physical block has already elapsed."""
    events = np.asarray(events)
    if events.ndim != 1 or events.dtype != np.bool_:
        raise ValueError("Need a Boolean native completion-event sequence")
    indices = np.flatnonzero(events)
    if not len(indices):
        return None
    ratio = round(policy_dt/physical_dt)
    if ratio < 1 or physical_dt <= 0 or not math.isclose(ratio*physical_dt, policy_dt, abs_tol=1e-12):
        raise ValueError("Completion requires integral physical substeps")
    return int((indices[0]+1)*ratio)


def spin_measurement(omega_body_z, requested_rpm, *, minimum_rpm=80.):
    omega = np.asarray(omega_body_z)
    if omega.ndim != 1 or not len(omega) or not np.isfinite(omega).all() or requested_rpm == 0.:
        raise ValueError("Need actual finite body yaw and a signed spin request")
    signed_rpm = np.sign(requested_rpm)*omega*60./(2.*math.pi)
    return dict(actual_body_signed_rpm_mean=float(signed_rpm.mean()),
        actual_body_signed_rpm_min=float(signed_rpm.min()),
        actual_body_signed_rpm_p05=float(np.quantile(signed_rpm, .05)),
        minimum_rpm_frame_fraction=float(np.mean(signed_rpm >= minimum_rpm)),
        target_rpm=float(abs(requested_rpm)))


def path_metrics(position_xy, initial_xy, initial_heading_rad):
    position = np.asarray(position_xy)-np.asarray(initial_xy)
    if position.ndim != 2 or position.shape[-1] != 2 or not len(position) or not np.isfinite(position).all():
        raise ValueError("Need true local XY, including the initial reference")
    forward = np.array([math.cos(initial_heading_rad), math.sin(initial_heading_rad)])
    lateral = np.array([-forward[1], forward[0]])
    radius = np.linalg.norm(position, axis=1)
    return dict(max_xy_excursion_m=float(radius.max()), terminal_xy_offset_m=float(radius[-1]),
        max_straight_lateral_m=float(np.abs(position@lateral).max()),
        longitudinal_terminal_m=float((position@forward)[-1]))


def braking_measurement(time, velocity_x, position_xy, at_s, *, speed_threshold=.05, hold_s=1.):
    """Find a held stop after the command, retaining wrong-direction rebound."""
    time, speed, xy = np.asarray(time), np.asarray(velocity_x), np.asarray(position_xy)
    after = time >= at_s
    if (time.ndim != 1 or len(time) < 2 or speed.shape != time.shape
            or xy.shape != (len(time),2) or not after.any() or speed_threshold <= 0
            or not all(np.isfinite(v).all() for v in (time, speed, xy))
            or not np.isfinite(at_s) or not np.isfinite(hold_s) or hold_s <= 0):
        raise ValueError("Missing synchronized post-command braking trace")
    start = int(np.flatnonzero(after)[0])
    dt = float(np.median(np.diff(time)))
    if dt <= 0 or not np.allclose(np.diff(time), dt, atol=1e-8, rtol=1e-5):
        raise ValueError("Braking needs a complete, uniformly sampled valid prefix")
    width = round(hold_s/dt)
    if width < 1 or not math.isclose(width*dt, hold_s, abs_tol=1e-6):
        raise ValueError("Braking hold needs complete measured samples")
    quiet = np.abs(speed[start:]) <= speed_threshold
    runs = np.convolve(quiet.astype(int), np.ones(width, dtype=int), mode="valid") if len(quiet)>=width else []
    good = np.flatnonzero(np.asarray(runs)==width)
    stop = start+int(good[0]) if len(good) else None
    prefix = slice(max(0,start-round(1./dt)), start)
    before = speed[prefix]
    direction = np.sign(float(np.mean(before))) if len(before) else 0.
    return dict(held_stop_observed=stop is not None,
        settling_seconds=float(time[stop]-at_s) if stop is not None else None,
        distance_until_held_stop_m=float(np.linalg.norm(xy[stop]-xy[start])) if stop is not None else None,
        post_command_max_excursion_m=float(np.linalg.norm(xy[start:]-xy[start],axis=1).max()),
        reverse_rebound_speed_peak_m_s=float(np.maximum(-direction*speed[start:],0.).max()),
        terminal_speed_abs_m_s=float(abs(speed[-1])), pre_brake_mean_speed_m_s=float(before.mean()) if len(before) else None)


def analyze_case(case, arrays, lane, initial_root, spec, requirements):
    """Measure only the uninterrupted native prefix; never select a good trial.

    Engineering decisions are separate from the user's permanent acceptance
    requirements. Native recovery completion is compulsory for fallen starts.
    """
    from review_standing_acceptance import support_review

    mask = arrays["valid"][:, lane]
    if mask.dtype != np.bool_ or not mask[0] or (np.diff(mask.astype(int)) > 0).any():
        raise ValueError("Native case must be one uninterrupted episode")
    t = arrays["time"][:, lane][mask]
    dt = .005
    def trace(name):
        return arrays[name][:, lane][mask]
    position, gravity, omega, velocity = (trace(k) for k in ("position","gravity","omega","velocity"))
    requested, applied = trace("requested"), trace("applied")
    events = trace("completion_event")
    force = arrays["force"][mask,:,lane].reshape(-1,2,3)
    clearance = arrays["clearance"][mask,:,lane].reshape(-1,2)
    dq = arrays["leg_dq"][mask,:,lane].reshape(-1,4)
    tau = arrays["leg_torque"][mask,:,lane].reshape(-1,4)
    for value in (t,position,gravity,omega,velocity,requested,applied,force,clearance,dq,tau):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite native measurement")
    if not np.allclose(t, np.arange(1,len(t)+1)*dt, atol=2e-5):
        raise ValueError("Native prefix has missing/shifted policy samples")
    horizon = round(spec["seconds"]/dt)
    full = bool(len(t)==horizon)
    g = spec["engineering_gates"]
    tilt = np.degrees(np.arccos(np.clip(-gravity[:,2],-1.,1.)))
    completion_indices = np.flatnonzero(events)
    completion_seconds = [float(t[i]) for i in completion_indices]
    fell = trace("fallen")
    native_fail = bool(trace("terminated").any())
    root = np.asarray(initial_root)
    x,y,z,w = root[3:]
    heading = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    path = path_metrics(position[:,:2],root[:2],heading)
    steady = t >= spec["steady_start_s"]
    evaluated = steady.copy()
    if case["recovery"]:
        evaluated &= t > completion_seconds[0] if completion_seconds else False
    if case.get("push"):
        falls = np.flatnonzero(trace("in_motion_fall"))
        after_fall = [i for i in completion_indices if len(falls) and i>falls[0]]
        evaluated &= t > t[after_fall[0]] if after_fall else False
    tracking = None
    if evaluated.any():
        error = np.abs(velocity[evaluated,0]-requested[evaluated,0])
        yaw_error = np.abs(omega[evaluated,2]-requested[evaluated,1])
        vx_gate = np.maximum(g["motion_speed_MAE_abs_m_s"],
                             np.abs(requested[evaluated,0])*g["motion_speed_MAE_fraction"])
        yaw_gate = np.maximum(g["motion_yaw_MAE_abs_rad_s"],
                              np.abs(requested[evaluated,1])*g["motion_yaw_MAE_fraction"])
        tracking = dict(raw_speed_mae_m_s=float(error.mean()),raw_yaw_mae_rad_s=float(yaw_error.mean()),
            applied_speed_mae_m_s=float(np.abs(velocity[evaluated,0]-applied[evaluated,0]).mean()),
            applied_yaw_mae_rad_s=float(np.abs(omega[evaluated,2]-applied[evaluated,1]).mean()),
            height_mae_m=float(np.abs(position[evaluated,2]-requested[evaluated,2]).mean()),
            measured_speed_mean_m_s=float(velocity[evaluated,0].mean()),
            measured_yaw_mean_rad_s=float(omega[evaluated,2].mean()),
            raw_speed_tracking_gate=bool((error/vx_gate).mean()<=1.),
            raw_yaw_tracking_gate=bool((yaw_error/yaw_gate).mean()<=1.))
    leg_q = trace("motor_q")[:,[0,1,3,4]]
    late = leg_q[steady] if steady.any() else leg_q
    frequency = np.fft.rfftfreq(len(late),dt)
    spectrum = np.fft.rfft(late-late.mean(0),axis=0)
    spectrum[(frequency<3.)|(frequency>95.)]=0.
    band = np.fft.irfft(spectrum,n=len(late),axis=0)
    smooth = dict(leg_speed_physical_peak_rad_s=float(np.abs(dq).max()),
        leg_speed_physical_rms_rad_s=float(np.sqrt(np.mean(dq**2))),
        leg_torque_slew_rms_nm_per_ms=float(np.sqrt(np.mean(np.diff(tau,axis=0)**2))),
        leg_positive_power_mean_w=float(np.maximum(tau*dq,0.).sum(-1).mean()),
        leg_angle_3_95hz_rms_mrad=float(np.sqrt(np.mean(band**2))*1000.),
        wheel_speed_policy_peak_rad_s=float(np.abs(trace("motor_dq")[:,[2,5]]).max()),
        action_delta_rms_per_5ms=float(np.sqrt(np.mean(np.diff(trace("actions"),axis=0)**2))) if len(t)>1 else None,
        dynamic_leg_speed_alert=bool(np.abs(dq).max()>g["dynamic_leg_speed_alert_rad_s"]),
        interpretation="Physical leg speed/torque at1kHz; wheels/targets at200Hz. Dynamic band RMS includes intended transitions and is descriptive, not a universal smoothness pass.")
    base_checks = dict(full_horizon=full, no_native_guard=native_fail is False,
        closure_gap=bool(trace("gap").max()<requirements["inherited_references"]["closed_chain_gap_max_m"]))
    checks = dict(base_checks,physical_leg_speed_engineering_gate=not smooth["dynamic_leg_speed_alert"])
    support = dict(evaluated=False,reason="Nonstationary task; support is descriptive",support_pass=False)
    stationary = (case["command"][:2]==[0.,0.] and
        all(command[:2]==[0.,0.] for _,command in case.get("sequence",[])))
    if stationary:
        start = first_completion_sample(events) if case["recovery"] else 0
        # Completion at the final boundary leaves no observed standing suffix.
        if start is not None and start>=len(force):
            start=None
        support = support_review((np.linalg.norm(force,axis=-1)>2.).all(-1),standing_start_sample=start,
            ignored_max_ms=requirements["standing_contact"]["ignored_continuous_loss_max_ms"])
        checks["effective_standing_contact"] = support["support_pass"]
    extra={}
    cap=case["capability"]
    if cap in ("standing","standing_height_change") or cap=="initial_recovery" and stationary:
        start = first_completion_sample(events)//5 if case["recovery"] and len(completion_indices) else 0
        if case["recovery"] and not len(completion_indices):
            checks["completed_recovery"] = False
        elif start<len(t):
            standing_xy=position[start:,:2]
            reference=position[start-1,:2] if start else root[:2]
            radius=np.linalg.norm(standing_xy-reference,axis=-1)
            ending=t[start:] >= max(t[start], spec["seconds"]-spec["terminal_window_s"])
            slope = float(np.polyfit(t[start:][ending],radius[ending],1)[0]) if ending.sum()>=2 else None
            height=case["command"][2]
            limit=.1 if height==.42 else .05
            extra["standing"] = dict(max_xy_excursion_m=float(radius.max()),
                terminal_offset_m=float(radius[-1]),tail_radius_mean_m=float(radius[ending].mean()) if ending.any() else None,
                tail_outward_slope_m_s=slope,very_good_drift_within_1cm=bool(radius.max()<=.01))
            checks.update(standing_xy=bool(radius.max()<=limit),
                standing_tilt=bool(tilt[start:].max()<10.),standing_speed=bool(np.sqrt(np.mean(np.sum(velocity[start:,:2]**2,axis=-1)))<.03))
            if cap=="standing" and height==.42:
                checks["high_stance_recentering"] = bool(ending.any() and slope is not None
                    and radius[ending].mean()<=max(.05,.8*radius.max()) and slope<=.001)
    if cap in ("low_speed_straight","high_speed_straight"):
        checks["straight_path"] = path["max_straight_lateral_m"]<=g["straight_max_lateral_m"]
    if cap in ("stationary_spin","spinning_height_change"):
        spin=spin_measurement(omega[steady,2],case["requested_rpm"]) if steady.any() else None
        extra["spin"]=spin
        radius=np.linalg.norm(position[:,:2]-root[:2],axis=-1)
        tail=t>=spec["seconds"]-spec["terminal_window_s"]
        slope=float(np.polyfit(t[tail],radius[tail],1)[0]) if tail.sum()>=2 else None
        extra["spin_outward_slope_m_s"]=slope
        checks.update(minimum_actual_80rpm=bool(spin is not None and
            spin["minimum_rpm_frame_fraction"]>=g["spin_minimum_rpm_frame_fraction"]),
            center_excursion=path["max_xy_excursion_m"]<=g["spin_max_center_excursion_m"],
            center_outward_trend=bool(slope is not None and slope<=.001))
    if cap.endswith("height_change"):
        transitions=[]
        for j,(start,command) in enumerate(case["sequence"]):
            end=case["sequence"][j+1][0] if j+1<len(case["sequence"]) else spec["seconds"]+.001
            final=(t>=max(start,end-1.))&(t<end)
            error=float(np.abs(position[final,2]-command[2]).mean()) if final.any() else None
            transitions.append(dict(at_seconds=start,requested_height_m=command[2],final_height_mae_m=error))
        extra["height_transitions"]=transitions
        checks["each_final_height"] = all(r["final_height_mae_m"] is not None and
            r["final_height_mae_m"]<=g["transition_final_height_error_m"] for r in transitions)
    if cap=="high_speed_braking":
        if t[-1]>=spec["braking_at_s"]:
            brake=braking_measurement(t,velocity[:,0],position[:,:2],spec["braking_at_s"],
                speed_threshold=g["braking_stopped_speed_m_s"],hold_s=g["braking_stopped_hold_seconds"])
            extra["braking"]=brake
            # A lane that never reached the requested pre-braking speed is not
            # evidence of stopping from that high speed.
            checks["entered_requested_high_speed"] = abs(brake["pre_brake_mean_speed_m_s"]-case["command"][0]) <= .1*abs(case["command"][0])
            checks["held_stop"] = bool(brake["held_stop_observed"] and
                brake["settling_seconds"]<=g["braking_settling_max_seconds"] and
                brake["post_command_max_excursion_m"]<=g["braking_distance_max_m"])
        else:
            checks["held_stop"]=False
    if cap in ("initial_recovery","moving_fall_recovery"):
        fell_indices=np.flatnonzero(trace("in_motion_fall"))
        after=[i for i in completion_indices if cap=="initial_recovery" or len(fell_indices) and i>fell_indices[0]]
        first=after[0] if after else None
        enable_indices=np.flatnonzero(trace("enable_event"))
        origin=(t[enable_indices[0]] if len(enable_indices) else 0.) if cap=="initial_recovery" else (t[fell_indices[0]] if len(fell_indices) else None)
        delay=float(t[first]-origin) if first is not None and origin is not None else None
        extra["recovery"]=dict(actual_moving_fall=bool(len(fell_indices)),
            actual_completion_events_s=completion_seconds,first_applicable_completion_s=float(t[first]) if first is not None else None,
            active_recovery_seconds=delay,anti_fall_observed_without_recovery_test=bool(cap=="moving_fall_recovery" and not len(fell_indices)))
        checks.update(actual_fall_test=bool(cap=="initial_recovery" or len(fell_indices)),
            actual_completion=first is not None,rapid_recovery=bool(delay is not None and delay<=g["recovery_max_seconds"]),
            stable_task_followup=bool(first is not None and t[-1]-t[first]>=g["recovery_followup_seconds"]
                and not fell[first+1:].any() and trace("followup_success")[-1]))
    else:
        checks["no_fall"] = not fell.any()
    if tracking is not None:
        checks.update(raw_speed_tracking=tracking["raw_speed_tracking_gate"],
            raw_yaw_tracking=(checks["minimum_actual_80rpm"] if cap in ("stationary_spin","spinning_height_change")
                              else tracking["raw_yaw_tracking_gate"]),
            height=tracking["height_mae_m"]<(.01 if cap=="standing" else g["motion_height_MAE_m"]))
    else:
        checks["tracking_measured"]=False
    dynamic_window=evaluated if case["recovery"] or case.get("push") else np.ones(len(t),dtype=bool)
    checks["upright_task_phase"] = bool(dynamic_window.any() and tilt[dynamic_window].max()<=g["dynamic_tilt_max_deg"])
    if stationary and cap!="standing_height_change":
        checks["stationary_physical_leg_speed"] = bool(np.abs(dq).max()<3.) if not case["recovery"] else bool(
            support.get("evaluated") and np.abs(dq[support["standing_start_sample"]:]).max()<3.)
    return dict(case=case,observed_seconds=float(t[-1]),full_horizon=full,
        native_guard_triggered=native_fail,closure_gap_peak_m=float(trace("gap").max()),
        tilt_peak_deg=float(tilt.max()),actual_fall_detected=bool(fell.any()),path=path,
        tracking=tracking,smoothness=smooth,standing_contact=support,
        engineering_checks=checks,engineering_pass=bool(all(checks.values())),
        overall_policy_qualified=False,hardware_ready=False,**extra)
