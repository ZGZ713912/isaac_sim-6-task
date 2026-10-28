"""Fixed final-command cold matrix; raw truth, actual completion and no suffix selection."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from train_scratch_standing import digest, write_json, nominal_action
from review_standing_acceptance import support_review
from wheeled_tasks.chassis import prone_motion as task

CAPABILITIES = ("R01", "R02", "S01", "M01", "M02", "M03", "M04", "H01", "H02", "H03", "B01", "Q01", "D01")


def evaluation_cases(env, plan):
    rows = []
    for height in (.23, .305, .42):
        for pitch in (0., -.5, .5):
            rows.append(dict(role="stand", birth_height=height, pitch_deg=pitch))
    rows += [dict(role=r, birth_height=.305, pitch_deg=0.) for r in
             ("forward", "reverse", "spin_left", "spin_right", "arc_fl", "arc_fr", "arc_rl", "arc_rr")]
    rows += [dict(role="height", birth_height=.305, target_height=h, pitch_deg=0.) for h in (.23, .42)]
    rows += [dict(role="prone", birth_height=.173, pitch_deg=p) for p in (0., -.5, .5)]
    for row in rows:
        prone = row["role"] == "prone"
        height = row.get("target_height", row["birth_height"] if not prone else .305)
        row.update(pose_index=env.prone_index if prone else env.height_pose[row["birth_height"]],
            recovery=prone, target=task.target(row["role"], plan["max_updates"], plan, final=True, height=height),
            command=[0., 0., .305 if prone else row["birth_height"]])
    return rows


def review_arrays(a, cases, initial_xy, plan, dt=.005):
    limits = plan["evaluation"]
    horizon = round(limits["seconds"]/dt)
    hold_steps = round(30./dt)
    rows = []
    for lane, case in enumerate(cases):
        n = int(a["valid"][:, lane].sum())
        first = np.flatnonzero(a["completed_event"][:n, lane])
        enables = np.flatnonzero(a["enable_event"][:n, lane]) if "enable_event" in a else np.array([],dtype=int)
        prone = case["recovery"]
        # Completion occurs at the END of its physics interval. Contact from
        # that interval belongs to the rise, so start at the next interval.
        start = int(first[0])+1 if prone and len(first) else None if prone else 0
        full_quiet = start is not None and n >= start+hold_steps
        q = slice(start, start+hold_steps) if full_quiet else slice(0, 0)
        force = a["wheel_force"][:n, :, lane]
        supported = (np.linalg.norm(force, axis=-1) > 2.).all(-1).reshape(-1)
        support = support_review(supported, standing_start_sample=start*5 if start is not None and start*5 < len(supported) else None,
            ignored_max_ms=limits["ignored_contiguous_support_loss_ms"]) if n else dict(evaluated=False, support_pass=False)
        v, omega, command = a["velocity"][:n, lane], a["omega"][:n, lane], a["commands"][:n, lane]
        xy = a["position"][:n, lane, :2]
        quiet_anchor = xy[start-1] if full_quiet and prone and start > 0 else initial_xy[lane]
        tilt = np.degrees(np.arccos(np.clip(-a["gravity"][:n, lane, 2], -1., 1.)))
        moving = (np.abs(command[:, :2]) > [0.05, .1]).any(-1)
        plateau = (np.abs(command[:, :2]-np.array(case["target"][:2])) < 1e-4).all(-1) & moving
        height_active = np.abs(command[:, 2]-.305) > .02
        quiet_case = case["role"] in ("prone", "stand")
        chosen = plateau if plateau.any() else height_active
        quaternion = a["root_quaternion"][:n, lane]
        yaw = np.unwrap(np.arctan2(2.*(quaternion[:, 3]*quaternion[:, 2]+quaternion[:, 0]*quaternion[:, 1]),
            1.-2.*(quaternion[:, 1]**2+quaternion[:, 2]**2)))
        row = dict(role=case["role"], birth_height_m=case["birth_height"], pitch_deg=case["pitch_deg"],
            full_horizon=n == horizon, terminated=bool(a["terminated"][:n, lane].any()), valid_policy_samples=n,
            rise_completed=bool(len(first)) if prone else None,
            completion_time_s=float(a["time"][first[0], lane]) if prone and len(first) else None,
            actor_enabled_time_s=float(a["time"][enables[0], lane]) if len(enables) else None,
            active_rise_time_s=float(a["time"][first[0], lane]-a["time"][enables[0], lane]) if len(first) and len(enables) else None,
            post_event_start_sample=start, full30s_quiet_window=full_quiet, support=support,
            raw_clearance_peak_m=float(a["wheel_clearance"][:n, :, lane].max()) if n else None,
            rise_xy_max_m=float(np.linalg.norm(xy[:start]-initial_xy[lane], axis=-1).max()) if prone and start else None,
            quiet_xy_max_m=float(np.linalg.norm(xy[q]-quiet_anchor, axis=-1).max()) if full_quiet else None,
            quiet_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(v[q, :2]**2, axis=-1)))) if full_quiet else None,
            quiet_tilt_rms_deg=float(np.sqrt(np.mean(tilt[q]**2))) if full_quiet else None,
            quiet_tilt_peak_deg=float(tilt[q].max()) if full_quiet else None,
            height_mae_m=float(np.abs(a["height"][:n, lane]-command[:, 2]).mean()) if n else None,
            active_height_mae_m=float(np.abs(a["height"][:n, lane][height_active]-command[height_active, 2]).mean()) if height_active.any() else None,
            motion_vx_mae_m_s=float(np.abs(v[chosen, 0]-command[chosen, 0]).mean()) if chosen.any() else None,
            motion_yaw_mae_rad_s=float(np.abs(omega[chosen, 2]-command[chosen, 1]).mean()) if chosen.any() else None,
            actual_spin_min_rpm=float(np.abs(omega[plateau, 2]).min()*60./math.tau) if plateau.any() and "spin" in case["role"] else None,
            actual_spin_mean_rpm=float(omega[plateau, 2].mean()*60./math.tau) if plateau.any() and "spin" in case["role"] else None,
            spin_center_max_m=float(np.linalg.norm(xy-initial_xy[lane], axis=-1).max()) if n and "spin" in case["role"] else None,
            lateral_path_peak_m=float(np.abs(xy[:, 1]-initial_xy[lane, 1]).max()) if n else None,
            heading_change_rad=float(yaw[-1]-yaw[0]) if n else None,
            requested_effective_difference_peak=float(np.abs(a["requested"][:n, lane]-command).max()) if n else None,
            wheel_speed_peak_rad_s=float(np.abs(a["motor_velocity"][:n, :, lane, [2, 5]]).max()) if n else None,
            leg_speed_peak_rad_s=float(np.abs(a["motor_velocity"][:n, :, lane, [0, 1, 3, 4]]).max()) if n else None,
            torque_peak_nm=float(np.abs(a["motor_effort"][:n, :, lane]).max()) if n else None,
            closure_gap_peak_m=float(a["gap"][:n, lane].max()) if n else None,
            full_acceptance_qualified=False)
        common = (row["full_horizon"] and not row["terminated"] and support["support_pass"]
                  and row["closure_gap_peak_m"] <= .003 and row["height_mae_m"] <= limits["height_mae_m"])
        # Prone height during the rise is deliberately far from305mm; only its
        # original post-completion30s window contributes to standing height MAE.
        if prone and full_quiet:
            row["post_completion_height_mae_m"] = float(np.abs(a["height"][q, lane]-.305).mean())
            common = (row["full_horizon"] and not row["terminated"] and support["support_pass"]
                      and row["closure_gap_peak_m"] <= .003 and row["post_completion_height_mae_m"] <= .01)
        if quiet_case:
            passed = (common and full_quiet and row["quiet_xy_max_m"] <= limits["standing_xy_max_m"]
                      and row["quiet_speed_rms_m_s"] <= limits["standing_speed_rms_m_s"]
                      and row["quiet_tilt_peak_deg"] <= limits["tilt_peak_deg"])
            if prone:
                passed = passed and row["completion_time_s"] <= plan["rise_deadline_seconds"] and row["rise_xy_max_m"] <= .1
                passed = (passed and int(plateau.sum()) >= round(2./dt)
                    and row["motion_vx_mae_m_s"] <= limits["motion_vx_mae_m_s"]
                    and row["motion_yaw_mae_rad_s"] <= limits["motion_yaw_mae_rad_s"])
        else:
            passed = (common and bool(chosen.any()) and row["motion_vx_mae_m_s"] <= limits["motion_vx_mae_m_s"]
                      and row["motion_yaw_mae_rad_s"] <= limits["motion_yaw_mae_rad_s"])
            if "spin" in case["role"]:
                passed = passed and row["actual_spin_min_rpm"] >= plan["spin_minimum_rpm"]
            if case["role"] == "height":
                passed = passed and bool(height_active.any()) and row["active_height_mae_m"] <= .01
        row["engineering_screen_pass"] = bool(passed)
        if "spring_compression" in a:
            row.update(spring_compression_min_m=float(a["spring_compression"][:n, :, lane].min()) if n else None,
                spring_compression_max_m=float(a["spring_compression"][:n, :, lane].max()) if n else None,
                spring_outside_identified_fraction=float(a["spring_outside_identified"][:n, :, lane].mean()) if n else None,
                spring_force_peak_n=float(a["spring_force"][:n, :, lane].max()) if n else None,
                spring_right_symmetry_assumed=True, spring_unmeasured_extension_is_not_calibrated=True)
        if "spring_ff_holding_effort_nm" in a and n:
            row.update(spring_feedforward_peak_nm=float(np.abs(a["spring_ff_holding_effort_nm"][:n, :, lane]).max()),
                spring_feedforward_invalid_geometry_fraction=float((~a["spring_ff_geometry_valid"][:n, :, lane]).mean()),
                spring_encoder_compression_error_peak_m=float(np.abs(a["spring_ff_compression_m"][:n, :, lane]
                    -a["spring_compression"][:n, :, lane]).max()),
                spring_total_request_saturated_fraction=float((np.abs(a["spring_ff_requested_effort_nm"][:n, :, lane]) > 40.).mean()))
        rows.append(row)
    return rows


@torch.inference_mode()
def evaluate(env, actor, output, writer, completed, plan, *, stop_requested=lambda: False):
    cases = evaluation_cases(env, plan)
    count = len(cases)
    env.evaluation_cases = [cases[i % count].copy() for i in range(env.num_envs)]
    env.reset(torch.arange(env.num_envs, device=env.device))
    initial_xy = env.robot.data.root_link_pose_w.torch[:count, :2].cpu().numpy().copy()-env.origins[:count, :2].cpu().numpy()
    observation = env.get_observations()
    trace = {}
    for _ in range(round(plan["evaluation"]["seconds"]/env.policy_dt)):
        if stop_requested():
            break
        valid = ~env.finished[:count].clone()
        observation, _, _, extras = env.step(actor(observation))
        state = extras["diagnostics"]
        physical = env.prone_motion_physical
        values = {key:state[source][:count] for key, source in dict(position="local_position", velocity="velocity",
            omega="omega", gravity="gravity", height="height", time="time", commands="commands", gap="gap",
            requested="interval_requested_command", terminated="terminated", completed_event="recovery_completed_event",
            recovered_at="recovered_at", root_quaternion="root_quaternion_xyzw", action="actions",
            passive="passive", enable_event="enable_event", belly_force="belly_force_peak_n",
            nonwheel_force="nonwheel_force_peak_n", quiet30_completed_event="quiet30_completed_event").items()}
        values.update({key:state["prone_route"][key][:count] for key in ("knee_deg", "support_error_m",
            "encoder_leg_length_m", "encoder_leg_length_rate_m_s", "encoder_wheel_body_m",
            "encoder_valid", "encoder_vs_native_wheel_error_m", "encoder_vs_passive_knee_error_deg")})
        values.update({key:value[:, :count] for key, value in physical.items()})
        values["valid"] = valid
        for key, value in values.items():
            trace.setdefault(key, []).append(value.cpu().numpy().copy())
    arrays = {key:np.stack(value) for key,value in trace.items()}
    rows = (review_arrays(arrays, cases, initial_xy, plan) if trace else
            [dict(role=c["role"], engineering_screen_pass=False, no_samples=True) for c in cases])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_trace.npz", **arrays)
    passed = sum(row["engineering_screen_pass"] for row in rows)
    result = dict(schema="prone-motion-cold-v1", case_count=count, pass_count=passed,
        all_pass=passed == count, interrupted=stop_requested(), cases=rows, case_definitions=cases,
        trace_sha256=digest(output/"native_trace.npz"), hardware_ready=False, full_capability_qualified=False,
        contact_scope="Initialized standing or the first actual completion event; <=2ms only diagnostic; <=2N is not proof of visible separation",
        capability_status={key:dict(status="partial_evidence" if key in ("R01", "S01", "M02", "M03", "M04", "H01", "Q01", "D01")
                                  else "not_tested", qualified=False) for key in CAPABILITIES})
    write_json(output/"result.json", result)
    if writer is not None:
        profile = env.getup_cfg.get("randomization_study", {}).get("profile", "nominal")
        prefix = "ColdProneMotion/"+profile
        writer.add_scalar(prefix+"/engineering_pass_count", passed, completed)
        writer.add_scalar(prefix+"/case_count", count, completed)
        for i, row in enumerate(rows):
            for key, value in row.items():
                if type(value) in (int, float, bool) and value is not None:
                    writer.add_scalar(f"{prefix}/case{i:02d}_{row['role']}/{key}", value, completed)
        writer.flush()
    return result


@torch.inference_mode()
def probe(env, plan, *, stop_requested=lambda: False):
    """No optimizer, no policy checkpoint; constants verify wiring only."""
    action = nominal_action(env).expand(env.num_envs, -1).clone()
    checks = dict(fixed_PD_and_spring=bool((env.domain.kp == 1.).all() and (env.domain.kd == 1.).all()
        and (env.domain.spring == 1.).all()), rates=env.dt == .001 and env.policy_dt == .005)
    checks["same_identified_axial_model_both_sides"] = bool(
        env.identified_spring.curve.force_knots_n == env.identified_spring.right_curve.force_knots_n
        and env.startup_report["identified_gas_spring"]["default_catalogue_force_replaced_for_this_task"])
    from wheeled_tasks.chassis.identified_wheel_response import IdentifiedWheelResponse
    reference = IdentifiedWheelResponse(env.num_envs, env.device, env.dt,
        env.cfg["identified_wheel_response"], root=Path(__file__).resolve().parents[1])
    reference.inertia = env.identified_wheels.inertia.clone()
    from wheeled_tasks.chassis.encoder_spring_feedforward import EncoderGasSpringFeedforward
    spring_reference = EncoderGasSpringFeedforward(env.cfg["v5_control"]["encoder_spring_feedforward"]).to(
        device=env.device, dtype=torch.float32)
    checks["encoder_only_feedforward_physical_spring_retained"] = bool(
        env.startup_report["encoder_spring_feedforward"]["physical_spring_retained"]
        and not env.startup_report["encoder_spring_feedforward"]["privileged_force_or_slider_input"])
    checks["measured_wheel_native_properties"] = env.startup_report["training_identified_wheels"]["native_properties_verified"]
    checks["independent_wheels_P05_delay_once"] = bool(env.v5.wheel_prior["kd"] == .5
        and reference.whole_delay == [5, 3] and reference.fractional_delay == [.5, .5])
    wheel_replay_max_error = 0.
    spring_replay_error = {key:0. for key in ("limited_effort_nm", "holding_effort_nm", "length_m",
        "compression_m", "axial_force_n", "length_jacobian_m_rad")}
    spring_encoder_error_peak = spring_encoder_error_sum = 0.
    spring_ff_peak = 0.
    spring_valid_count = spring_invalid_count = 0
    expected_limits = [round((70. if r == "prone" else 30. if r == "stand" else 20.)/env.policy_dt)
                       for r in env.course_roles]
    checks["actual_per_role_episode_limits"] = env.episode_limits.tolist() == expected_limits
    passive_seen = active_prone_seen = body_allowed = 0
    spring_min = torch.full((2, 2), float("inf"), device=env.device)
    spring_max = -spring_min.clone()
    spring_outside = torch.zeros(2, dtype=torch.int64, device=env.device)
    spring_count = 0
    for _ in range(300):
        if stop_requested():
            from train_scratch_standing import TrainingStopped
            raise TrainingStopped()
        observation, reward, done, extra = env.step(action)
        state = extra["diagnostics"]
        expected = ((sum(state["task_reward_terms"].values())*env.policy_dt+state["reward_potential_progress"]
                    +3.*state["recovery_completed_event"])*extra["policy_learning_mask"]-3.*state["terminated"])
        checks["actual_reward_payment"] = checks.get("actual_reward_payment", True) and bool(torch.allclose(reward, expected, atol=1e-5))
        checks["finite"] = checks.get("finite", True) and bool(torch.isfinite(reward).all()) and all(torch.isfinite(v).all() for v in observation.values())
        checks["all_six_1ms_velocities_and_efforts"] = env.prone_motion_physical["motor_velocity"].shape == (5, env.num_envs, 6)
        physical = env.prone_motion_physical
        for tick in range(env.decimation):
            ff = spring_reference.motor_efforts(physical["spring_ff_pd_effort_nm"][tick],
                physical["spring_ff_encoder_position"][tick])
            for key in spring_replay_error:
                expected_ff = ff if key == "limited_effort_nm" else spring_reference.last[key]
                difference = (physical["spring_ff_"+key][tick]-expected_ff).abs().max()
                spring_replay_error[key] = max(spring_replay_error[key], float(difference))
            reference.reset(physical["wheel_disabled"][tick])
            request = torch.zeros(env.num_envs, 6, device=env.device)
            velocity_input = torch.zeros_like(request)
            request[:, [2, 5]] = physical["wheel_input_effort"][tick]
            velocity_input[:, [2, 5]] = physical["wheel_input_velocity"][tick]
            expected_motor = reference.apply_torque(request, velocity_input)[:, [2, 5]]
            expected_motor[physical["wheel_disabled"][tick]] = 0.
            errors = [
                (physical["motor_effort"][tick][:, [2, 5]] - expected_motor).abs().max(),
                (physical["wheel_passive_effort"][tick] - reference.passive_effort).abs().max(),
                (physical["wheel_drive_bound"][tick] - reference.drive_bound).abs().max(),
                (physical["wheel_net_effort"][tick] - expected_motor - reference.passive_effort).abs().max()]
            wheel_replay_max_error = max(wheel_replay_max_error, float(torch.stack(errors).max()))
        if env.fixed_cases is None:
            reference.reset(done.nonzero().flatten())
        checks["actual_1khz_wheel_delay_envelope_and_separate_loss"] = wheel_replay_max_error <= 1e-6
        # Keep force, length, Jacobian and torque errors in their own units.
        checks["actual_1khz_encoder_compensation_replay"] = all(
            value <= (1e-7 if key in ("length_m", "compression_m", "length_jacobian_m_rad") else 1e-5)
            for key, value in spring_replay_error.items())
        checks["unclipped_PD_plus_holding_then_total_limit"] = checks.get("unclipped_PD_plus_holding_then_total_limit", True) and bool(
            torch.allclose(physical["spring_ff_requested_effort_nm"],
                physical["spring_ff_pd_effort_nm"]+physical["spring_ff_holding_effort_nm"], atol=1e-5, rtol=1e-6)
            and torch.allclose(physical["spring_ff_limited_effort_nm"],
                physical["spring_ff_requested_effort_nm"].clamp(-40., 40.), atol=1e-6, rtol=0.))
        checks["actual_leg_effort_with_compensation_at_most40nm"] = checks.get("actual_leg_effort_with_compensation_at_most40nm", True) and bool(
            (physical["motor_effort"][:, :, [0, 1, 3, 4]].abs() <= 40.+1e-6).all())
        checks["all_six_disabled_efforts_zero_after_compensation"] = checks.get("all_six_disabled_efforts_zero_after_compensation", True) and bool(
            (physical["motor_effort"][physical["wheel_disabled"]] == 0.).all())
        valid = physical["spring_ff_geometry_valid"]
        held = physical["spring_ff_holding_effort_nm"].unflatten(-1, (2, 2))
        checks["invalid_encoder_branch_has_zero_compensation"] = checks.get("invalid_encoder_branch_has_zero_compensation", True) and bool(
            (held[~valid] == 0.).all())
        spring_ff_peak = max(spring_ff_peak, float(held.abs().max()))
        error = (physical["spring_ff_compression_m"]-physical["spring_compression"]).abs()[valid]
        if error.numel():
            spring_encoder_error_peak = max(spring_encoder_error_peak, float(error.max()))
            spring_encoder_error_sum += float(error.sum())
        spring_valid_count += int(valid.sum())
        spring_invalid_count += int((~valid).sum())
        checks["wheel_loss_dissipative"] = checks.get("wheel_loss_dissipative", True) and bool(
            (physical["wheel_passive_effort"] * physical["wheel_input_velocity"] <= 1e-8).all())
        checks["wheel_current_at_most_4p5nm"] = checks.get("wheel_current_at_most_4p5nm", True) and bool(
            (physical["motor_effort"][:, :, [2, 5]].abs() <= 4.5+1e-6).all())
        checks["queued_wheel_torque_cleared_while_disabled"] = checks.get("queued_wheel_torque_cleared_while_disabled", True) and bool(
            (physical["motor_effort"][:, :, [2, 5]][physical["wheel_disabled"]] == 0.).all())
        curve = env.identified_spring.curve
        expected_spring = curve.force(physical["spring_compression"].clamp(
            min=curve.compression_knots_m[0], max=curve.compression_knots_m[-1]))
        checks["actual_1khz_measured_spring_force"] = checks.get("actual_1khz_measured_spring_force", True) and bool(
            torch.allclose(physical["spring_force"], expected_spring * env.domain.spring, atol=1e-4, rtol=1e-6))
        measured = torch.stack((physical["spring_compression"], physical["spring_force"]))
        spring_min = torch.minimum(spring_min, measured.amin(dim=(1, 2)))
        spring_max = torch.maximum(spring_max, measured.amax(dim=(1, 2)))
        spring_outside += physical["spring_outside_identified"].sum(dim=(0, 1))
        spring_count += physical["spring_compression"].shape[0] * env.num_envs
        checks["actor35_critic83"] = observation["policy"].shape == (env.num_envs,35) and observation["critic"].shape == (env.num_envs,83)
        checks["zero_disabled_torque"] = checks.get("zero_disabled_torque", True) and bool((state["passive_motor_peak"][state["passive"]] == 0.).all())
        checks["disabled_samples_masked"] = checks.get("disabled_samples_masked", True) and not bool((state["passive"] & extra["policy_learning_mask"]).any())
        passive_seen += int(state["passive"].sum())
        active_prone_seen += int((extra["policy_learning_mask"] & state["task_recovery_rising"]).sum())
        body_allowed += int((state["task_recovery_rising"] & (state["nonwheel_force_peak_n"] > 2.) & ~state["terminated"]).sum())
    checks.update(disabled_birth_observed=passive_seen > 0, active_prone_observed=active_prone_seen > 0,
                  body_contact_permitted_during_rise=body_allowed > 0,
                  active_nonzero_encoder_compensation_observed=spring_ff_peak > 1. and spring_valid_count > 0)
    return dict(checks={k:bool(v) for k,v in checks.items()}, all_checks_pass=all(checks.values()),
                new_PPO=0, learned_behavior=False, active_prone_samples=active_prone_seen,
                spring_feedforward=dict(startup=env.startup_report["encoder_spring_feedforward"],
                    replay_max_error_nm=max(spring_replay_error["limited_effort_nm"], spring_replay_error["holding_effort_nm"]),
                    replay_max_error_by_quantity=spring_replay_error, holding_effort_peak_nm=spring_ff_peak,
                    valid_side_samples=spring_valid_count, invalid_side_samples=spring_invalid_count,
                    valid_encoder_compression_error_peak_m=spring_encoder_error_peak,
                    valid_encoder_compression_error_mean_m=spring_encoder_error_sum/max(1, spring_valid_count),
                    geometry_error_scope="sampled/delayed encoders against native slider before the same1ms interval; diagnostic, not force calibration or learned behavior"),
                wheels=dict(model_sha256=env.identified_wheels.cfg["model_sha256"],
                    replay_max_error_nm=wheel_replay_max_error, physical_samples_per_side=spring_count,
                    native=env.startup_report["training_identified_wheels"]),
                gas_spring=dict(model_sha256=env.identified_spring.settings["sha256"],
                    side_order=["left", "right"], physical_samples_per_side=spring_count,
                    compression_min_m=spring_min[0].tolist(), compression_max_m=spring_max[0].tolist(),
                    force_min_n=spring_min[1].tolist(), force_max_n=spring_max[1].tolist(),
                    outside_identified_samples=spring_outside.tolist(),
                    outside_identified_fraction=(spring_outside / spring_count).tolist(),
                    outside_identified_policy="endpoint_hold_unvalidated"))
