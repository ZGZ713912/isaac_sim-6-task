"""Standard-prone diagnostics and completion-anchored cold standing review."""
from pathlib import Path

import numpy as np
import torch

from review_standing_acceptance import support_review
from train_scratch_standing import digest, nominal_action, write_json
from wheeled_tasks.chassis.standard_prone import valid_disabled_start
from wheeled_tasks.chassis.scut_observation import CONTROL_FROM_POLICY


def native_probe(env, steps, output):
    action = nominal_action(env).expand(env.num_envs, -1).clone()
    reset_q = env.bank_joints[env.initialization_pose_index, env.ids].expand(env.num_envs, -1)
    target_q = env.bank_joints[env.initialization_target_pose_index, env.ids].expand(env.num_envs, -1)
    control_action = action[:, CONTROL_FROM_POLICY]
    initial_legs, initial_wheels, _ = env.v5.decode(control_action, reset_q)
    target_error = initial_legs-target_q[:, list(env.v5.LEGS)]
    target_error = torch.atan2(target_error.sin(), target_error.cos())
    checks = dict(native_rates=env.dt == .001 and env.policy_dt == .005,
        all_births_standard_prone=bool(env.recovery_lane.all()),
        initial_motor_targets_match_declared_bias_geometry=bool(target_error.abs().max() < 1e-5 and initial_wheels.abs().max() == 0.))
    checks["initial_leg_exploration_is_not_clipped"] = bool(
        (control_action[:, list(env.v5.LEGS)].abs()+3.*env.prone_task["initial_std_policy_order"][0]
         < env.v5.action_bounds[list(env.v5.LEGS)]).all())
    rows, trace = [], {key: [] for key in ("knee_deg", "xy_excursion_m", "support_error_m", "passive", "terminated",
                                         "gravity", "height", "enable_event", "settled_family", "belly_force_peak_n",
                                         "encoder_knee_rate_deg_s", "encoder_leg_length_rate_m_s", "encoder_wheel_body_m",
                                         "encoder_valid", "encoder_vs_native_wheel_error_m", "encoder_vs_passive_knee_error_deg")}
    active_samples = 0
    body_contact_samples = 0
    for _ in range(steps):
        observation, reward, done, extra = env.step(action)
        state, route = extra["diagnostics"], extra["diagnostics"]["prone_route"]
        expected = (state["reward_density"] * env.policy_dt + state["reward_potential_progress"]
                    + 3.*state["recovery_completed_event"]) * extra["policy_learning_mask"] - 3.*state["terminated"]
        checks["independent_guided_reward"] = checks.get("independent_guided_reward", True) and bool(torch.allclose(reward, expected))
        checks["all_1ms_contact_samples"] = checks.get("all_1ms_contact_samples", True) and (
            env.standing_contact_monitor.last["force"].shape == (5, env.num_envs, 2, 3))
        checks["finite_sensors_and_route"] = checks.get("finite_sensors_and_route", True) and all(
            bool(torch.isfinite(value).all()) for value in (*observation.values(), *route.values()))
        checks["encoder_fk_agrees_with_independent_native_geometry"] = checks.get(
            "encoder_fk_agrees_with_independent_native_geometry", True) and bool(
                route["encoder_vs_native_wheel_error_m"].max() < .003
                and route["encoder_vs_passive_knee_error_deg"].max() < 1.)
        checks["passive_samples_excluded_from_learning"] = checks.get("passive_samples_excluded_from_learning", True) and not bool(
            (state["passive"] & extra["policy_learning_mask"]).any())
        checks["disabled_birth_has_zero_six_motor_effort"] = checks.get("disabled_birth_has_zero_six_motor_effort", True) and bool(
            (state["passive_motor_peak"][state["passive"]] == 0.).all())
        checks["failure_does_not_bootstrap"] = checks.get("failure_does_not_bootstrap", True) and not bool(
            (state["terminated"] & extra["policy_bootstrap_mask"]).any())
        enable = state["enable_event"]
        checks["actual_enable_pose_is_upright_belly_grounded"] = checks.get("actual_enable_pose_is_upright_belly_grounded", True) and bool(
            ((state["settled_family"][enable] == 0) & valid_disabled_start(
                state["gravity"], state["height"], state["belly_force_peak_n"], env.prone_task)[enable]).all())
        active_samples += int(extra["policy_learning_mask"].sum())
        body_contact_samples += int(((state["nonwheel_force_peak_n"] > 2.) & ~state["terminated"]).sum())
        for key in trace:
            value = route[key] if key in route else state[key]
            trace[key].append(value.cpu().numpy())
        rows.append(dict(active=int(extra["policy_learning_mask"].sum()), terminated=int(state["terminated"].sum()),
            mechanical=int(state["mechanical"].sum()), closure_gap=int(state["closure_gap"].sum()),
            impact=int(state["impact_bad"].sum()), max_gap_m=float(state["gap"].max()),
            knee_min_deg=float(route["knee_deg"].min()), knee_max_deg=float(route["knee_deg"].max()),
            leg_speed_peak_rad_s=float(env.physical_standing_last["dq"].abs().max()),
            xy_excursion_peak_m=float(route["xy_excursion_m"].max())))
    checks["native_body_contact_was_permitted"] = body_contact_samples > 0
    checks["actor_control_interval_reached"] = active_samples > 0
    material = env.contact_material.summary()
    declared = env.prone_task["effective_wheel_ground_friction_range"]
    actual = material["effective_wheel_static_min_max"]
    checks["friction_within_declared_band"] = declared[0]-1e-6 <= actual[0] <= actual[1] <= declared[1]+1e-6
    output = Path(output)
    np.savez_compressed(output / "native_probe_trace.npz", **{key: np.stack(value) for key, value in trace.items()})
    result = dict(schema="standard-prone-native-wiring-probe-v1", new_PPO=0, learned_policy=False,
        control="constant declared initialization target; physical reset stays disabled prone; no learned actor or optimizer",
        num_envs=env.num_envs, physics_steps=steps*env.decimation, active_control_samples=active_samples,
        allowed_body_contact_samples=body_contact_samples, checks=checks, all_wiring_checks_pass=all(checks.values()),
        contact_material=material, transitions=rows, actual_task=env.prone_task,
        learned_getup_qualified=False, hardware_ready=False)
    write_json(output / "native_startup.json", env.startup_report)
    write_json(output / "native_probe.json", result)
    if not all(checks.values()):
        raise RuntimeError("Standard-prone native wiring probe failed: " + str(checks))
    return result


def evaluate(env, actor, output, writer, completed):
    evaluation = env.prone_task["evaluation"]
    count = min(21, env.num_envs)
    cases = [dict(pose_index=env.initialization_pose_index, recovery=True, command=[0., 0., .305])
             for _ in range(env.num_envs)]
    env.evaluation_cases = cases
    env.reset(torch.arange(env.num_envs, device=env.device))
    env.fixed_cases = cases
    env.max_episode_length = horizon = round(evaluation["seconds"] / env.policy_dt)
    observation = env.get_observations()
    keys = ("position", "velocity", "gravity", "height", "time", "recovered_at", "terminated", "valid",
            "force", "clearance", "dq", "torque", "gap", "knees", "support_error", "excursion", "completed_event", "recovery_timeout")
    trace = {key: [] for key in keys}
    with torch.inference_mode():
        for _ in range(horizon):
            valid = ~env.finished[:count].clone()
            observation, _, _, extra = env.step(actor(observation))
            state, contact = extra["diagnostics"], env.standing_contact_monitor.last
            physical, route = env.physical_standing_last, state["prone_route"]
            recovery_timeout = ((state["time"] >= evaluation["recovery_deadline_seconds"])
                                & ~state["recovered"] & ~env.finished)
            values = dict(position=state["local_position"][:count], velocity=state["velocity"][:count],
                gravity=state["gravity"][:count], height=state["height"][:count], time=state["time"][:count],
                recovered_at=state["recovered_at"][:count], terminated=state["terminated"][:count], valid=valid,
                force=contact["force"][:, :count], clearance=contact["clearance"][:, :count],
                dq=physical["dq"][:, :count], torque=physical["torque"][:, :count], gap=state["gap"][:count],
                knees=route["knee_deg"][:count], support_error=route["support_error_m"][:count],
                excursion=route["xy_excursion_m"][:count], completed_event=state["recovery_completed_event"][:count],
                recovery_timeout=recovery_timeout[:count])
            for key, value in values.items():
                trace[key].append(value.cpu().numpy())
            #40s total minus30s required standing leaves10s for completion.
            # Preserve the timeout as a failed case, never select a later rise.
            env.finished |= recovery_timeout
            if env.finished[:count].all():
                break
    arrays = {key: np.stack(value) for key, value in trace.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "native_trace.npz", **arrays)
    rows = []
    for lane in range(count):
        valid = arrays["valid"][:, lane]
        events = np.flatnonzero(arrays["completed_event"][:, lane] & valid)
        row = dict(lane=lane, initialized_pose=env.prone_task["reset_pose_name"],
            recovery_completed=bool(len(events)), terminated=bool(arrays["terminated"][valid, lane].any()),
            recovery_deadline_missed=bool(arrays["recovery_timeout"][valid, lane].any()),
            maximum_xy_excursion_from_prone_m=float(arrays["excursion"][valid, lane].max()),
            minimum_physical_knee_deg=float(arrays["knees"][valid, lane].min()),
            leg_speed_peak_rad_s=float(np.abs(arrays["dq"][valid, :, lane]).max()),
            closure_gap_peak_m=float(arrays["gap"][valid, lane].max()), baseline_pass=False)
        if len(events):
            start = int(events[0])
            selected = valid & (np.arange(len(valid)) > start)
            row["recovery_completed_at_s"] = float(arrays["time"][start, lane])
            row["standing_seconds"] = int(selected.sum()) * env.policy_dt
            if selected.any():
                support = (np.linalg.norm(arrays["force"][selected, :, lane], axis=-1) > 2.).all(-1).reshape(-1)
                row["support"] = support_review(support, standing_start_sample=0,
                    ignored_max_ms=evaluation["ignored_contiguous_support_loss_ms"])
                xy = arrays["position"][selected, lane, :2] - arrays["position"][start, lane, :2]
                row["standing_xy_excursion_m"] = float(np.linalg.norm(xy, axis=-1).max())
                row["standing_tilt_peak_deg"] = float(np.degrees(np.arccos(np.clip(-arrays["gravity"][selected, lane, 2], -1., 1.))).max())
                row["standing_speed_rms_m_s"] = float(np.sqrt(np.mean(np.sum(arrays["velocity"][selected, lane, :2]**2, axis=-1))))
                row["standing_height_mae_m"] = float(np.abs(arrays["height"][selected, lane] - .305).mean())
                row["baseline_pass"] = bool(not row["terminated"] and row["standing_seconds"] >= evaluation["post_completion_standing_seconds"] - 1e-6
                    and row["support"]["support_pass"] and row["standing_xy_excursion_m"] <= evaluation["standing_xy_max_m"]
                    and row["standing_tilt_peak_deg"] <= 10. and row["standing_speed_rms_m_s"] <= .03
                    and row["standing_height_mae_m"] <= .01 and row["closure_gap_peak_m"] <= .003)
        rows.append(row)
    result = dict(schema="standard-prone-cold-review-v1", completed_updates=completed, case_count=count,
        recovery_completed=sum(r["recovery_completed"] for r in rows), baseline_pass=sum(r["baseline_pass"] for r in rows),
        cases=rows, same_initial_state_replicas_are_not_independent_seeds=True,
        support_start="First complete physical interval after actual completion event; no post-hoc suffix selection",
        source_scope=evaluation["scope"], full_capability_qualified=False, hardware_ready=False,
        trace_sha256=digest(output / "native_trace.npz"))
    write_json(output / "result.json", result)
    for key in ("recovery_completed", "baseline_pass"):
        writer.add_scalar("EvalStandardProne/" + key, result[key], completed)
    for lane, row in enumerate(rows):
        for key in ("maximum_xy_excursion_from_prone_m", "minimum_physical_knee_deg", "leg_speed_peak_rad_s"):
            writer.add_scalar(f"EvalStandardProne/case_{lane:02d}/{key}", row[key], completed)
    writer.flush()
    return result
