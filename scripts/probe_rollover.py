"""Zero-PPO native wiring checks for the explicitly scoped rollover skill."""
import math

import torch

from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL


@torch.inference_mode()
def probe(env, plan, *, stop_requested=lambda: False):
    objects = tuple(id(x) for x in (env.robot, env.sim, env.v5, env.actuator_response,
                                    env.identified_wheels, env.getup_feedback, env.domain, env.gimbal))
    count = env.num_envs
    env.reset(torch.arange(count, device=env.device))
    observation = env.get_observations()
    q0 = env.robot.data.joint_pos.torch[:, env.ids][:, list(env.v5.LEGS)].clone()
    reference_dt = env.getup_feedback.dt
    native_before = env.native_capture_steps
    # Explicit wiring fixture only: shorten prefixes to30ms. This does not
    # change the frozen training course and grants no behavioral admission.
    env.prefix_remaining[1::2] = .03
    env.prefix_initial[1::2] = .03
    env.prefix_remaining[::2] = env.prefix_initial[::2] = 0.
    checks = dict(
        policy35=observation["policy"].shape == (count,35),
        finite_birth=all(bool(torch.isfinite(v).all()) for v in observation.values()),
        zero_actor_previous_at_birth=bool((observation["policy"][:,22:28] == 0.).all()),
        inverted_train_split_only=all(env.poses[c["pose_index"]]["recovery"]["family"] == "roof_down"
            and env.poses[c["pose_index"]]["recovery"]["split"] == "train" for c in env.course_cases),
        all_four_leg_classes={env.poses[c["pose_index"]]["recovery"]["leg_class"] for c in env.course_cases}
            == set(plan["rollover_training"]["leg_classes"]),
        independent_encoder_births=bool((q0[:,0]+q0[:,2]).abs().max() > .1),
        varied_mechanical_knees=bool(env.route_state()["knee_deg"].std() > 1.),
        holdout_not_sampled=not bool(set(env.roof_train) & set(env.roof_holdout)),
        actual_feedback_5ms=reference_dt == .005,
        clocks=(env.dt == .001 and env.policy_dt == .01 and env.decimation == 10),
        original_PD=env.v5.settings["leg_kp"] == 160. and env.v5.settings["leg_kd"] == 2.5,
        fixed_PD_spring=bool((env.domain.kp == 1.).all() and (env.domain.kd == 1.).all()
                             and (env.domain.spring == 1.).all()),
        original_wheel_model=env.identified_wheels.enabled,
        physical_spring_and_friction=env.leg_friction.enabled and env.v5.gas_compensation is not None,
        completion_is_prone_not_standing=env.getup_cfg["stable_seconds"] == .01
            and plan["rollover_training"]["capture_hold_s"] == 1.,
        wheel_contact_not_a_capture_requirement=not plan["rollover_training"]["wheel_contact_required"],
        no_legacy_standing_gate=env.global_curriculum is None,
        record_only_closure=bool(env.getup_cfg["closure_gap_record_only"]))
    steps = 40
    prefix_seen = actor_seen = release_seen = False
    target_peak = 0.
    learning = bootstrap = None
    for index in range(steps):
        if stop_requested():
            raise InterruptedError("Native rollover probe stopped")
        action = torch.zeros(count, 6, device=env.device)
        # Exercise all axes without a checkpoint, optimizer or behavior claim.
        action[:, :4] = .03*math.sin(index*.3)
        action[:, 4:] = .02*math.sin(index*.5)
        observation, reward, done, info = env.step(action)
        d = info["diagnostics"]
        prefix, learning, bootstrap = (d["rollover_prefix_interval"], info["policy_learning_mask"],
                                       info["policy_bootstrap_mask"])
        prefix_seen |= bool(prefix.any())
        actor_seen |= bool(learning.any())
        released = d["rollover_prefix_release_event"] & ~done
        release_seen |= bool(released.any())
        checks["prefix_excluded_from_PPO"] = checks.get("prefix_excluded_from_PPO", True) and not bool((prefix & learning).any())
        checks["release_previous_actor_action_zero"] = checks.get("release_previous_actor_action_zero", True) and bool(
            (observation["policy"][released,22:28] == 0.).all())
        checks["release_next_observation_can_bootstrap"] = checks.get("release_next_observation_can_bootstrap", True) and bool(
            bootstrap[released].all())
        checks["finite_native_steps"] = checks.get("finite_native_steps", True) and bool(torch.isfinite(reward).all()) and all(
            bool(torch.isfinite(v).all()) for v in observation.values())
        decoder = env.encoder_increment_actions
        delta = decoder.last_targets-decoder.last_reference
        delta = torch.atan2(delta.sin(), delta.cos())
        target_peak = max(target_peak, float(delta.abs().max()))
        checks["bounded_encoder_increment"] = target_peak <= .100002
        checks["no_extra_feedback_observe"] = decoder.reference_feedback_tick <= env.getup_feedback.tick
        checks["only_actual_native_capture_clock"] = env.native_capture_steps-native_before == (index+1)*env.decimation
        checks["no_short_hold_completion"] = not bool(d["rollover_completed_event"].any())
        checks["whole_interval_force_records"] = bool(torch.isfinite(d["rollover_whole_impulse_peak_ns"]).all())
        checks["reward_terms_present"] = len(d["task_reward_terms"]) == 12
        checks["TB_metrics_present"] = all(tag in info["log"] for tag in (
            "Rollover/completed_events", "Rollover/zero_prefix_completed_events",
            "Rollover/native_hold_peak_s", "RolloverReward/orientation_per_second"))
    checks.update(native_prefix_fixture_executed=prefix_seen, pure_actor_fixture_executed=actor_seen,
        native_release_executed=release_seen, same_controller_history_at_release=env.prefix_release_integrity,
        same_native_objects=objects == tuple(id(x) for x in (env.robot, env.sim, env.v5, env.actuator_response,
            env.identified_wheels, env.getup_feedback, env.domain, env.gimbal)))
    return dict(checks={k:bool(v) for k,v in checks.items()}, all_checks_pass=all(checks.values()),
        new_PPO=0, behavior_qualified=False, physical_steps=steps*env.decimation,
        actor_control_steps=steps, target_encoder_displacement_peak_rad=target_peak,
        prefix_release_receipts=env.prefix_release_receipts,
        scope="shortened_native_prefix_fixture_only_not_a_rollover_success_test")
