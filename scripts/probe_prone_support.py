"""Zero-PPO reset/action/capture-clock admission, distinct from task success."""
import math
import torch


@torch.inference_mode()
def probe(env, plan, *, stop_requested=lambda: False):
    objects = tuple(id(x) for x in (env.robot, env.sim, env.v5, env.actuator_response,
                                   env.identified_wheels, env.getup_feedback, env.domain, env.gimbal))
    training_observation = env.get_observations()["policy"]
    low_goal_visible = bool(torch.allclose(training_observation[:,3], env.goal_height*5.)
                            and (env.goal_height == .23).all())
    # This separate process has no learner/checkpoint. Exercise every derived
    # reset, including held-out placements, without granting policy behavior.
    indices = env.support_train + env.support_holdout
    if env.num_envs < len(indices):
        raise ValueError("Native admission must cover the complete new reset parameter bank")
    env.support_native_probe = True
    env.evaluation_cases = [dict(role="prone", recovery=True, pose_index=indices[i % len(indices)],
        birth_height=env.poses[indices[i % len(indices)]]["root_position_m"][2],
        target=[0.,0.,.305], command=[0.,0.,.305], heading_rad=(i//len(indices))*math.pi/2,
        pitch_deg=0., evaluation_seconds=1.) for i in range(env.num_envs)]
    env.reset(torch.arange(env.num_envs, device=env.device))
    observation = env.get_observations()
    q = env.robot.data.joint_pos.torch[:, env.ids][:, list(env.v5.LEGS)]
    root = env.robot.data.root_link_pose_w.torch
    height = root[:, 2]-env.origins[:, 2]
    checks = dict(policy35=observation["policy"].shape == (env.num_envs,35),
        finite_birth=all(bool(torch.isfinite(x).all()) for x in observation.values()),
        zero_previous_actor_action=bool((observation["policy"][:,22:28] == 0.).all()),
        complete_new_reset_bank={c["pose_index"] for c in env.course_cases} == set(indices),
        every_birth_normally_up=bool((env.robot.data.projected_gravity_b.torch[:,2] < -.999).all()),
        belly_height_not_suspended=bool(((height-.1830409426689148).abs() < 1e-5).all()),
        independent_leg_births=bool((q[:,0]+q[:,2]).abs().max() > .1),
        varied_mechanical_knees=bool(env.route_state()["knee_deg"].std() > 1.),
        four_leg_classes={env.poses[i]["recovery"]["leg_class"] for i in indices} == set(plan["prone_support"]["leg_classes"]),
        no_train_holdout_overlap=not bool(set(env.support_train) & set(env.support_holdout)),
        source_collision_cooked_passed=env.startup_report["chassis_collision"]["all_checks_pass"],
        no_scripted_takeover=not plan["prone_support"]["scripted_actions"],
        no_legacy_standing_curriculum=env.global_curriculum is None,
        actual_feedback_5ms=env.getup_feedback.dt == .005,
        native_clocks=env.dt == .001 and env.policy_dt == .01 and env.decimation == 10,
        original_PD=env.v5.settings["leg_kp"] == 160. and env.v5.settings["leg_kd"] == 2.5,
        fixed_PD_spring=bool((env.domain.kp == 1.).all() and (env.domain.kd == 1.).all() and (env.domain.spring == 1.).all()),
        physical_spring_friction_wheel_response=env.leg_friction.enabled and env.identified_wheels.enabled
            and env.v5.gas_compensation is not None,
        low_training_goal_visible=low_goal_visible,
        final_goal_visible=bool((env.goal_height == .305).all()
            and torch.allclose(observation["policy"][:,3], env.goal_height*5.)),
        original_actor_only=not env.support_config["actor_inputs_added"],
        capture_3s_native=env.support_config["capture_hold_s"] == 3. and env.getup_cfg["stable_seconds"] == .01)
    native_before = env.native_capture_steps
    peak = 0.
    faults = []
    for step in range(40):
        if stop_requested():
            raise InterruptedError("Prone support native admission stopped")
        action = torch.zeros(env.num_envs, 6, device=env.device)
        action[:, :4] = .02*math.sin(step*.3)
        action[:, 4:] = .02*math.sin(step*.5)
        observation, reward, done, info = env.step(action)
        d = info["diagnostics"]
        faults.extend(done.nonzero().flatten().tolist())
        checks["finite_native_steps"] = checks.get("finite_native_steps", True) and bool(torch.isfinite(reward).all()) and all(
            bool(torch.isfinite(x).all()) for x in observation.values())
        decoder = env.encoder_increment_actions
        delta = decoder.last_targets-decoder.last_reference
        peak = max(peak, float(torch.atan2(delta.sin(),delta.cos()).abs().max()))
        checks["bounded_encoder_relative_targets"] = peak <= .100002
        checks["no_extra_encoder_sampling"] = decoder.reference_feedback_tick <= env.getup_feedback.tick
        checks["capture_counts_actual_native_steps"] = env.native_capture_steps-native_before == (step+1)*env.decimation
        checks["no_short_probe_completion"] = not bool(d["support_completed_event"].any())
        checks["TB_metrics_present"] = all(k in info["log"] for k in (
            "ProneSupport/goal_completed_events", "ProneSupport/standing_completed_events",
            "ProneSupportReward/quiet_goal_per_second", "ProneSupport/native_hold_peak_s"))
    checks["no_reset_action_fixture_faults"] = not faults
    checks["same_controller_objects"] = objects == tuple(id(x) for x in (env.robot, env.sim, env.v5,
        env.actuator_response, env.identified_wheels, env.getup_feedback, env.domain, env.gimbal))
    return dict(checks={k:bool(v) for k,v in checks.items()}, all_checks_pass=all(checks.values()),
        new_PPO=0, behavior_qualified=False, physical_steps=400, actor_control_steps=40,
        target_encoder_displacement_peak_rad=peak, reset_fault_lanes=faults,
        new_reset_parameters_tested=len(indices), scope="new reset and six-axis wiring admission; not a recovery test")
