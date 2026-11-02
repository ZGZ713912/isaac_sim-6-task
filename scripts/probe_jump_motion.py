"""Composed original-plant and mixed-jump native checks; no PPO or checkpoints."""
from __future__ import annotations

import torch

from evaluate_prone_motion import probe as base_probe
from train_scratch_standing import nominal_action, TrainingStopped
from wheeled_tasks.chassis.jump_motion import AIRBORNE, JUMP, BASE
from wheeled_tasks.chassis.standing_support import wheel_bottom_clearance


@torch.inference_mode()
def probe(env, plan, *, stop_requested=lambda: False):
    ids = torch.arange(env.num_envs, device=env.device)
    env.jump_sampling_enabled = False
    env.reset(ids)
    try:
        result = base_probe(env, plan, stop_requested=stop_requested)
    finally:
        env.jump_sampling_enabled = True
        env.reset(ids)
    jump = env.jump_training
    checks = result["checks"]
    checks["jump_all_three_sample_pools_present"] = all(bool((jump.mode == mode).any()) for mode in (BASE, AIRBORNE, JUMP))
    checks["jump_does_not_remove_base_recovery"] = bool(env.recovery_lane.any())
    aerial = jump.mode == AIRBORNE
    wheel = env.robot.data.body_link_pose_w.torch[:, env.wheel_ids]
    clearance = wheel_bottom_clearance(env.wheel_centers(), wheel[:, :, 3:], env.origins[:, 2:3])
    checks["airborne_birth_geometrically_above_ground"] = bool((clearance[aerial] >= .039).all())
    checks["airborne_birth_not_counted_as_self_propelled_liftoff"] = not bool(jump.liftoff_event.any())
    checks["jump_declared_six_second_episode_limits"] = bool((env.episode_limits[jump.active] == round(jump.cfg["episode_seconds"]/env.policy_dt)).all())
    expected_base = torch.tensor([round((plan["training_prone_seconds"] if row["recovery"] else
        plan["standing_seconds"] if row["role"] == "stand" else plan["motion_seconds"])/env.policy_dt)
        for row in env.course_cases], device=env.device)
    checks["jump_original_episode_limits_retained_for_base"] = bool((env.episode_limits[~jump.active] == expected_base[~jump.active]).all())
    action = nominal_action(env).expand(env.num_envs, -1).clone()
    counts_before = jump.total.clone()
    for _ in range(round(2./env.policy_dt)):
        if stop_requested():
            raise TrainingStopped()
        observation, reward, done, extra = env.step(action)
        physical = env.prone_motion_physical
        active = jump.active
        checks["jump_actor35_critic_unchanged_size"] = observation["policy"].shape == (env.num_envs, 35) and observation["critic"].shape == (env.num_envs, env.cfg["critic_dim"])
        checks["jump_finite_observations_rewards"] = checks.get("jump_finite_observations_rewards", True) and bool(
            torch.isfinite(observation["policy"]).all() and torch.isfinite(observation["critic"]).all() and torch.isfinite(reward).all())
        checks["jump_air_wheel_spin_cannot_become_standing_odometry"] = checks.get("jump_air_wheel_spin_cannot_become_standing_odometry", True) and bool((observation["policy"][active][:, [14, 15, 31]] == 0.).all())
        checks["jump_all_physical_samples_and_torque_limits_retained"] = checks.get("jump_all_physical_samples_and_torque_limits_retained", True) and bool(
            physical["motor_effort"].shape[0] == env.decimation
            and (physical["motor_effort"][:, :, [0, 1, 3, 4]].abs() <= 40.+1e-5).all()
            and (physical["motor_effort"][:, :, [2, 5]].abs() <= 4.5+1e-5).all())
        checks["jump_TensorBoard_reward_and_event_tags"] = all(k in extra["log"] for k in (
            "Jump/request_events_per_step", "Jump/liftoff_events_per_step", "Airborne/landing_events_per_step",
            "JumpReward/jump_push_velocity_per_second"))
        d = extra["diagnostics"]
        expected = ((sum(d["task_reward_terms"].values())*env.policy_dt+d["reward_potential_progress"]
            +3.*d["recovery_completed_event"])*extra["policy_learning_mask"]
            -torch.where(env.recovery_lane, env.getup_cfg["multi_pose_failure_cost"], 3.)*d["terminated"])
        # Auto-reset can change lane ownership after a terminal transition;
        # failure cost is3 for every non-recovery lane, and recovery is fixed
        # by the persistent allocation in this initial mixed course.
        checks["jump_dense_and_event_reward_paid_once"] = checks.get("jump_dense_and_event_reward_paid_once", True) and bool(torch.allclose(reward, expected, atol=1e-5))
    checks["airborne_birth_physical_samples_observed"] = bool(jump.total[0] > counts_before[0])
    result["jump_training"] = dict(counters=jump.summary(), learned_behavior=False,
        action="constant nominal targets for wiring only", physical_dt_s=env.dt,
        policy_dt_s=env.policy_dt, external_takeoff_force=False)
    result["all_checks_pass"] = all(checks.values())
    return result
