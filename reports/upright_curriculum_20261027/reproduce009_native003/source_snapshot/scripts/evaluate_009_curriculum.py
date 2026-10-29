"""Cold behavior checks of one live actor; never create/load a learner."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from review_standing_acceptance import support_review
from train_scratch_standing import digest, write_json
from wheeled_tasks.chassis.standing009_curriculum import evaluation_cases


def review_arrays(arrays, cases, initial, plan, dt=.005):
    limits = plan["evaluation"]
    horizon = round(limits["seconds"]/dt)
    rows = []
    for i, case in enumerate(cases):
        valid = arrays["valid"][:, i]
        if not valid.any():
            rows.append(dict(role=case["role"], birth_height_m=case["birth_height_m"], baseline_pass=False,
                             full_horizon=False, no_valid_samples=True))
            continue
        velocity = arrays["velocity"][valid, i]
        omega = arrays["omega"][valid, i]
        command = arrays["commands"][valid, i]
        height = arrays["height"][valid, i]
        force = arrays["force"][valid, :, i]
        supported = (np.linalg.norm(force, axis=-1) > 2.).all(-1).reshape(-1)
        support = support_review(supported, standing_start_sample=0,
            ignored_max_ms=limits["ignored_contiguous_support_loss_ms"])
        moving = np.abs(command[:, :2]).max(-1) >= .01
        height_active = np.abs(command[:, 2]-case["birth_height_m"]) > .005
        tracking = moving if moving.any() else np.ones(len(command), dtype=bool)
        xy = arrays["position"][valid, i, :2]-initial[i, :2]
        row = dict(role=case["role"], birth_height_m=case["birth_height_m"], pitch_deg=case["pitch_deg"],
            full_horizon=int(valid.sum()) == horizon,
            terminated=bool(arrays["terminated"][valid, i].any()),
            tilt_peak_deg=float(np.degrees(np.arccos(np.clip(-arrays["gravity"][valid, i, 2], -1., 1.))).max()),
            height_mae_m=float(np.abs(height-command[:, 2]).mean()),
            active_height_mae_m=(float(np.abs(height[height_active]-command[height_active, 2]).mean()) if height_active.any() else None),
            xy_excursion_max_m=float(np.linalg.norm(xy, axis=-1).max()),
            speed_rms_m_s=float(np.sqrt(np.mean(np.sum(velocity[:, :2]**2, axis=-1)))),
            vx_mae_m_s=float(np.abs(velocity[tracking, 0]-command[tracking, 0]).mean()),
            yaw_mae_rad_s=float(np.abs(omega[tracking, 2]-command[tracking, 1]).mean()),
            requested_motion_observed=bool(moving.any()), requested_height_observed=bool(height_active.any()),
            measured_forward_velocity_mean_m_s=float(velocity[tracking, 0].mean()),
            measured_body_yaw_rate_mean_rad_s=float(omega[tracking, 2].mean()),
            leg_speed_peak_rad_s=float(np.abs(arrays["dq"][valid, :, i]).max()),
            closure_gap_peak_m=float(arrays["gap"][valid, i].max()),
            nonwheel_force_peak_n=float(arrays["nonwheel_force"][valid, i].max()),
            raw_wheel_clearance_peak_m=float(arrays["clearance"][valid, :, i].max()), support=support)
        row["baseline_pass"] = bool(row["full_horizon"] and not row["terminated"]
            and support["support_pass"] and row["tilt_peak_deg"] <= limits["tilt_peak_deg"]
            and row["height_mae_m"] <= limits["height_mae_m"]
            and row["closure_gap_peak_m"] <= limits["closure_gap_max_m"]
            and row["nonwheel_force_peak_n"] <= 2.)
        if case["role"] == "stand":
            row["baseline_pass"] &= (row["xy_excursion_max_m"] <= limits["standing_xy_max_m"]
                and row["speed_rms_m_s"] <= limits["standing_speed_rms_m_s"]
                and row["leg_speed_peak_rad_s"] <= limits["standing_leg_speed_peak_rad_s"])
        else:
            needs_motion = max(abs(v) for v in case["target"][:2]) >= .01
            needs_height = abs(case["target"][2]-case["birth_height_m"]) > .005
            row["baseline_pass"] &= (not needs_motion or row["requested_motion_observed"])
            row["baseline_pass"] &= (not needs_height or row["requested_height_observed"])
            row["baseline_pass"] &= (row["vx_mae_m_s"] <= limits["motion_vx_mae_m_s"]
                and row["yaw_mae_rad_s"] <= limits["motion_yaw_mae_rad_s"])
            if needs_height:
                row["baseline_pass"] &= row["active_height_mae_m"] is not None and row["active_height_mae_m"] <= limits["height_mae_m"]
        row["baseline_pass"] = bool(row["baseline_pass"])
        rows.append(row)
    return rows


def evaluate(env, actor, output, writer, completed, plan, task, *, stop_requested=lambda: False):
    cases = evaluation_cases(plan, task)
    count = len(cases)
    if count > env.num_envs:
        raise ValueError("Cold review needs all declared cases simultaneously")
    env.evaluation_cases = [deepcopy(cases[i % count]) for i in range(env.num_envs)]
    for row in env.evaluation_cases:
        row["pose_index"] = env.height_pose[row["birth_height_m"]]
    env.reset(torch.arange(env.num_envs, device=env.device))
    env.fixed_cases = env.evaluation_cases
    env.max_episode_length = round(plan["evaluation"]["seconds"]/env.policy_dt)
    env.episode_limits.fill_(env.max_episode_length)
    horizon = env.max_episode_length
    initial = env.robot.data.root_link_pose_w.torch[:count, :3].clone()-env.origins[:count]
    obs = env.get_observations()
    traces = {k: [] for k in ("position", "velocity", "omega", "gravity", "height", "commands", "time",
        "terminated", "valid", "force", "clearance", "dq", "torque", "gap", "nonwheel_force")}
    with torch.inference_mode():
        for _ in range(horizon):
            if stop_requested() and traces["valid"]:
                break
            valid = ~env.finished[:count].clone()
            obs, _, _, extra = env.step(actor(obs))
            state = extra["diagnostics"]
            contact, physical = env.standing_contact_monitor.last, env.physical_standing_last
            values = dict(position=state["local_position"][:count], velocity=state["velocity"][:count],
                omega=state["omega"][:count], gravity=state["gravity"][:count], height=state["height"][:count],
                commands=state["commands"][:count], time=state["time"][:count], terminated=state["terminated"][:count],
                valid=valid, force=contact["force"][:, :count], clearance=contact["clearance"][:, :count],
                dq=physical["dq"][:, :count], torque=physical["torque"][:, :count], gap=state["gap"][:count],
                nonwheel_force=state["nonwheel_force_peak_n"][:count])
            for key, value in values.items():
                traces[key].append(value.cpu().numpy())
    arrays = {key: np.stack(value) for key, value in traces.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_trace.npz", **arrays, initial_root_local=initial.cpu().numpy())
    rows = review_arrays(arrays, cases, initial.cpu().numpy(), plan, env.policy_dt)
    result = dict(schema="009-curriculum-cold-review-v1", completed_updates=completed, task=task,
        duration_seconds=plan["evaluation"]["seconds"], case_count=count,
        baseline_pass=sum(r["baseline_pass"] for r in rows), all_pass=all(r["baseline_pass"] for r in rows),
        standing_start="explicit initialized-standing reset; no moved start or suffix selection",
        cases=rows, trace_sha256=digest(output/"native_trace.npz"),
        domain_enabled=env.domain.enabled[:count].cpu().tolist(),
        source_scope=plan["evaluation"]["scope"], full_capability_qualified=False, hardware_ready=False,
        new_PPO=0, model_or_optimizer_loaded=False)
    if task == "standing":
        prefix = {k: v[:round(8./env.policy_dt)] for k, v in arrays.items()}
        p = deepcopy(plan)
        p["evaluation"]["seconds"] = 8.
        result["same_start_first8s_cases"] = review_arrays(prefix, cases, initial.cpu().numpy(), p, env.policy_dt)
        result["same_start_first8s_pass"] = sum(r["baseline_pass"] for r in result["same_start_first8s_cases"])
    write_json(output/"result.json", result)
    for i, row in enumerate(rows):
        for key in ("baseline_pass", "tilt_peak_deg", "height_mae_m", "xy_excursion_max_m", "speed_rms_m_s",
                    "vx_mae_m_s", "yaw_mae_rad_s", "leg_speed_peak_rad_s"):
            if key in row:
                writer.add_scalar(f"Eval009/{output.name}/case_{i:02d}/{key}", row[key], completed)
    writer.add_scalar(f"Eval009/{output.name}/pass_count", result["baseline_pass"], completed)
    writer.flush()
    return result
