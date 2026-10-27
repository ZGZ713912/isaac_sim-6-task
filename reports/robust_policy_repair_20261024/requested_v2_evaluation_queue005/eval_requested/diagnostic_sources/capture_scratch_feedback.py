"""Read the exact actor input and native response without changing evaluation."""
from pathlib import Path


def log_native_mid_trace(writer, physical, feedback, cases):
    """Event-backed trajectories use physical-sample steps, not PPO updates."""
    import numpy as np
    lane = next(i for i, case in enumerate(cases) if case["height_m"] == .305 and case["pitch_deg"] == 0.)
    valid = physical["valid"][:, lane]
    force = physical["force"][valid, :, lane].reshape(-1, 2, 3)
    dq = physical["dq"][valid, :, lane].reshape(-1, 4)
    clearance = physical["clearance"][valid, :, lane].reshape(-1, 2)
    fine = dict(minimum_wheel_force_n=np.linalg.norm(force, axis=-1).min(-1),
        dual_support=(np.linalg.norm(force, axis=-1) > 2.).all(-1),
        leg_speed_peak_rad_s=np.abs(dq).max(-1), wheel_clearance_peak_m=clearance.max(-1))
    gravity = physical["gravity"][valid, lane]
    command = feedback["action_applied_control"][valid, lane][:, [2, 5]]*10.
    speed = feedback["motor_velocity_control"][valid, lane][:, [2, 5]]
    coarse = dict(tilt_deg=np.degrees(np.arccos(np.clip(-gravity[:, 2], -1., 1.))),
        height_m=physical["position"][valid, lane, 2], forward_speed_m_s=physical["velocity"][valid, lane, 0],
        left_wheel_target_rad_s=command[:, 0], right_wheel_target_rad_s=command[:, 1],
        left_wheel_velocity_rad_s=speed[:, 0], right_wheel_velocity_rad_s=speed[:, 1])
    for rate, channels in ((1000, fine), (200, coarse)):
        for name, values in channels.items():
            for step, value in enumerate(values, start=1):
                writer.add_scalar(f"EvalScratchTrace{rate}Hz/mid_zero_pitch/{name}", float(value), step)
    return dict(case_index=lane, height_m=.305, pitch_deg=0., physical_frames=len(force),
        policy_frames=len(gravity), steps="one-based sample index; divide by tag rate to obtain seconds",
        source="original native samples; every valid frame from the first millisecond")


def evaluate_with_feedback(env, actor, output, writer, completed, *, evaluate, digest):
    import numpy as np
    import torch

    count = 21
    if env.num_envs < count:
        raise ValueError("Feedback trace needs the original 21 fixed standing cases")
    native_step = env.step
    traces = {name: [] for name in (
        "observation_policy", "action_raw_policy", "action_applied_control",
        "omega", "motor_position_control", "motor_velocity_control", "motor_effort_control",
        "velocity_world", "valid", "terminated", "time_s")}
    pending_input = None
    initial_position = None
    initial_native = None

    def observe_actor(module, inputs):
        nonlocal pending_input
        if pending_input is not None:
            raise RuntimeError("More than one actor call before a native step")
        pending_input = inputs[0]["policy"][:count].detach().clone()

    def step(actions):
        nonlocal pending_input, initial_position, initial_native
        if pending_input is None:
            raise RuntimeError("A native step has no matching actor input")
        frame = pending_input
        pending_input = None
        valid = ~env.finished[:count].clone()
        raw = actions[:count].detach().clone()
        if initial_position is None:
            initial_position = (env.robot.data.root_link_pose_w.torch[:count, :3]
                                - env.origins[:count]).detach().cpu().clone()
            # Read native state directly: get_observations would query the very
            # history cache whose reset boundary this recording must validate.
            pose = env.robot.data.root_link_pose_w.torch[:count].detach().clone()
            pose[:, :3] -= env.origins[:count]
            initial_native = dict(
                initial_root_pose_local=pose.cpu(),
                initial_gravity=env.robot.data.projected_gravity_b.torch[:count].detach().cpu().clone(),
                initial_omega=env.robot.data.root_com_ang_vel_b.torch[:count].detach().cpu().clone(),
                initial_motor_position_control=env.robot.data.joint_pos.torch[:count, env.ids].detach().cpu().clone(),
                initial_motor_velocity_control=env.robot.data.joint_vel.torch[:count, env.ids].detach().cpu().clone(),
                initial_action_control=env.actions[:count].detach().cpu().clone())
        result = native_step(actions)
        state = result[3]["diagnostics"]
        values = dict(observation_policy=frame, action_raw_policy=raw,
            action_applied_control=state["actions"][:count], omega=state["omega"][:count],
            motor_position_control=state["motor_position"][:count],
            motor_velocity_control=state["motor_velocity"][:count],
            motor_effort_control=state["motor_effort"][:count],
            velocity_world=state["velocity_world"][:count], valid=valid,
            terminated=state["terminated"][:count], time_s=state["time"][:count])
        for name, value in values.items():
            traces[name].append(value.detach().cpu().clone())
        return result

    handle = actor.register_forward_pre_hook(observe_actor)
    env.step = step
    try:
        result = evaluate(env, actor, output, writer, completed)
    finally:
        env.step = native_step
        handle.remove()
    arrays = {name: torch.stack(values).numpy() for name, values in traces.items()}
    arrays["initial_position_local"] = initial_position.numpy()
    arrays.update({name: value.numpy() for name, value in initial_native.items()})
    with np.load(Path(output) / "native_trace.npz") as physical:
        if not np.array_equal(arrays["valid"], physical["valid"]):
            raise RuntimeError("Feedback and physical trace valid masks disagree")
        if arrays["observation_policy"].shape != (physical["valid"].shape[0], count, 35):
            raise RuntimeError("Feedback trace has an unexpected actor ABI")
    path = Path(output) / "feedback_trace.npz"
    np.savez_compressed(path, **arrays)
    result["feedback_trace"] = dict(path=str(path), sha256=digest(path),
        source_sha256=digest(Path(__file__)), policy_hz=200,
        input_timing="exact actor argument before native step",
        initial_state_timing="native final reset pose and sensors before the first physical step; no observation-cache query",
        output_timing="native diagnostics before any autoreset",
        action_policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        action_control_order=["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"],
        scope="read-only standing sensor/action diagnosis; evaluator receipt declares nominal/randomized domains; no action, reward, reset or acceptance changes")
    from analyze_scratch_feedback_trace import explain
    with np.load(Path(output) / "native_trace.npz") as physical:
        explanation = explain(physical, arrays, result["cases"])
        result["tensorboard_native_mid_trace"] = log_native_mid_trace(writer, physical, arrays, result["cases"])
    explanation["source_sha256"] = digest(Path(explain.__code__.co_filename))
    result["feedback_explanation"] = explanation
    for height in (.23, .305, .42):
        selected = [r for r in explanation["cases"] if r["height_m"] == height]
        for metric in ("wheel_target_rms_rad_s", "wheel_speed_rms_rad_s", "roll_pitch_gyro_rms_rad_s",
                       "yaw_rms_rad_s", "body_xy_max_excursion_m"):
            writer.add_scalar(f"EvalScratchFeedback/{height:.3f}m/{metric}",
                sum(row[metric] for row in selected)/len(selected), completed)
    writer.flush()
    return result
