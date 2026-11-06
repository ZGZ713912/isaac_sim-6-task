#!/usr/bin/env python3
"""Six-motor RL, fresh by default with explicit interrupted-course recovery."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import train_scratch_standing as base


def configuration(output, plan, *, profile="nominal", strength=0., evaluation=False, num_envs=None, seed=None,
                  performance_backend="cuda_graph", device="cuda:0", sampling_backend="packed"):
    from wheeled_tasks.chassis.prone_motion import validate, timing, discounts
    from wheeled_algo.recovery_action_distribution import std_bounds
    plan = validate(plan)
    std_bounds(plan)
    rates, discount = timing(plan), discounts(plan)
    if performance_backend not in ("eager","cuda_graph"):
        raise ValueError("Unknown prone-motion performance backend")
    if sampling_backend not in ('reference', 'packed'):
        raise ValueError('Unknown prone-motion sampling backend')
    output.mkdir(parents=True, exist_ok=True)
    full_vehicle = plan.get("full_vehicle")
    args = SimpleNamespace(output=output,
        bank=ROOT/(plan["pose_bank"]["file"] if full_vehicle else "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json"),
        updates=plan["max_updates"], num_envs=num_envs or plan["num_envs"], rollout_steps=plan["rollout_steps"],
        seed=(plan["evaluation"]["seed"] if evaluation else plan["seed"]) if seed is None else seed,
        all_heights_after=1000,
        leg_target_rate_cost=None, device=device, fixed_target_probe=False, preflight_only=False,
        upright_task=None, standard_prone_task=None, precise_objective=False,
        normalize_observations=True, desync_episodes=False, standing_failures=False,
        wheel_velocity_increment=False, separate_leg_velocity_feedback=False, static_load_initialization=False,
        contact_break_cost=None, speed_weight=2., contact_weight=8., learning_rate=plan["learning_rate"],
        randomization_study=None if profile == "nominal" else ROOT/plan["randomization_contract"],
        randomization_profile=None if profile == "nominal" else profile, randomization_strength=strength,
        mechanism=None if plan["schema"] in ("wheel-leg-prone-motion-v3", "wheel-leg-prone-motion-v4", "wheel-leg-prone-motion-v5") else
            ROOT/"contracts/scratch_standing_base_mechanism_v2.json")
    if full_vehicle:
        for item in (plan["mechanism"], plan["pose_bank"]):
            if base.digest(ROOT/item["file"]) != item["sha256"]:
                raise ValueError("Full-vehicle mechanism or reset bank changed")
        args.mechanism = ROOT/plan["mechanism"]["file"]
    cfg, settings = base.prepare(args)
    if "native_environment_layout" in plan:
        cfg["native_environment_layout"] = plan["native_environment_layout"]
        settings["native_origin_mode"] = plan["native_environment_layout"]
    if "encoder_increment_actions" in plan:
        settings["cyclic_leg_targets"] = False
        settings["leg_action_clip"] = plan["encoder_increment_actions"]["leg_raw_clip"]
    if "closure_guard" in plan:
        guard=plan["closure_guard"]["peak_m"]
        # RobustGetupEnv owns this task's terminal mask. Keep the original
        # threshold for diagnostics even when its closure terminal is disabled.
        cfg["closure_gap_termination_m"]=guard
        settings["closure_gap_m"]=guard
        from wheeled_tasks.chassis.prone_motion import closure_record_only
        cfg["closure_gap_record_only"] = settings["closure_gap_record_only"] = closure_record_only(plan)
    if "solver_iterations" in plan:
        cfg.update(plan["solver_iterations"])
    settings.update(version=plan["schema"], prone_motion_task=deepcopy(plan),
        prone_geometry=json.loads((ROOT/plan["prone_geometry_contract"]).read_text()),
        quiet_training=False, quiet_critic_anchor=True, hold_motion_during_recovery=True,
        motion_fraction=.7, motion_push_fraction=0., stable_seconds=plan["completion_hold_seconds"],
        episode_seconds=plan["training_prone_seconds"], recovery_timeout_seconds=plan["rise_deadline_seconds"],
        reward_potential_gain=0., reward_gamma=discount["gamma"], sensor_policy_dt=rates["policy_dt"],
        boundary_xy_m=[45., 45.])
    if "standing_odometry" in plan:
        settings["standing_odometry"] = deepcopy(plan["standing_odometry"])
    if "native_wheel_encoder" in plan:
        settings["native_wheel_encoder"] = deepcopy(plan["native_wheel_encoder"])
    if "cold_evaluation_lifecycle" in plan:
        from wheeled_tasks.chassis.cold_evaluation_lifecycle import validate as validate_cold_lifecycle
        validate_cold_lifecycle(plan["cold_evaluation_lifecycle"])
    if full_vehicle:
        from wheeled_tasks.chassis.full_vehicle_training import authenticate
        authenticate(cfg, ROOT)
        if cfg["full_vehicle"] != full_vehicle:
            raise ValueError("Plan and mechanism select different full vehicles")
        if cfg.get("leg_motor_friction") != plan.get("leg_motor_friction"):
            raise ValueError("Plan and mechanism select different leg friction models")
        settings["prone_geometry"]["encoder_geometry"].update(
            table=plan["standing_odometry"]["geometry_file"],
            table_sha256=plan["standing_odometry"]["geometry_table_sha256"],
            model_spec_sha256=plan["standing_odometry"]["geometry_model_spec_sha256"])
    if evaluation and profile != "nominal":
        settings["nominal_fraction"] = 0.
        cfg["dynamics_randomization"]["enabled_fraction"] = 1.
    cfg.update(episode_seconds=plan["training_prone_seconds"], record_diagnostics=True,
               curriculum="prone_motion_long_v1", learning_rate=plan["learning_rate"], entropy_coef=.001,
               prone_motion_performance=performance_backend,
               prone_motion_sampling_backend=sampling_backend)
    cfg["stages"][0].update(vx_max=plan["forward_speed_m_s"], yaw_max=plan["spin_target_rpm"]*math.tau/60.)
    cfg["policy_architecture"]["scope"] = f"Fresh actor35, critic{cfg['critic_dim']}, six fully trainable motor outputs"
    cfg["v5_control"]["action_clip"] = 16.
    if plan["schema"] in ("wheel-leg-prone-motion-v3", "wheel-leg-prone-motion-v4", "wheel-leg-prone-motion-v5"):
        gas = cfg["v5_control"]["gas_spring_compensation"]
        if (plan.get("gas_spring_backend") != "dev_compensation_v2"
                or gas != plan["gas_spring_feedforward"]
                or gas["model_file"] != plan["gas_spring_model"]["file"]
                or gas["model_sha256"] != plan["gas_spring_model"]["sha256"]):
            raise ValueError("Unified training requires the authenticated dev gas fit and one compensation path")
    else:
        from wheeled_tasks.chassis.training_gas_spring import resolve
        model_spec = json.loads((ROOT/cfg["asset_directory"]/"model_spec.json").read_text())
        cfg["v5_control"]["identified_gas_spring"] = resolve(ROOT, plan["gas_spring_model"], model_spec)
        from wheeled_tasks.chassis.encoder_spring_feedforward import resolve as resolve_feedforward
        cfg["v5_control"]["encoder_spring_feedforward"] = resolve_feedforward(ROOT,
            plan["gas_spring_feedforward"], cfg["v5_control"]["identified_gas_spring"],
            ROOT/cfg["asset_directory"]/"model_spec.json")
    if plan.get("multi_pose_recovery") is not None:
        from wheeled_tasks.chassis.multi_pose_recovery import validate_bank,validate_native_admission
        bank_data = json.loads(args.bank.read_text())
        validate_bank(bank_data,plan["multi_pose_recovery"],
                      cfg["asset_manifest_sha256"],base.digest(ROOT/cfg["asset_directory"]/"model_spec.json"))
        if plan["multi_pose_recovery"].get("require_native_reset_admission"):
            validate_native_admission(bank_data,ROOT,rates)
        settings["multi_pose_failure_cost"] = plan["multi_pose_recovery"]["failure_cost"]
        # Near-ground feasible placements, not an assumption of static equilibria.
        settings.update(passive_min_seconds=.05,passive_max_seconds=.15,passive_hold_seconds=.05)
    from wheeled_tasks.chassis.training_wheel_response import configure
    cfg, wheel_control = configure(ROOT, cfg, plan["wheel_model"], settings)
    if wheel_control["timing"] != rates:
        raise ValueError("Task and wheel-control policy/physics timing differ")
    for key in ("wheel_static_friction", "wheel_dynamic_friction"):
        cfg["contact_material"][key] = plan["friction"]["wheel"]
    if "randomization_study" in settings:
        study = settings["randomization_study"]
        study["effective_wheel_ground_friction_range"] = [
            (v+plan["friction"]["wheel"])/2. for v in study["ground_friction_range"]]
        base.write_json(output/"randomization_study.resolved.json", study)
    design_path = output/"config/design.json"
    design = json.loads(design_path.read_text())
    row = design["stages"]["speed"]["roles"]["train"]
    if not full_vehicle and plan.get("gas_spring_backend") == "dev_compensation_v2":
        # Retain the original component evidence and its parent hash. Only the
        # environment declaration changed; the legacy step remains AST-identical.
        evidence = "model/gas_spring_compensation_unified_20261029/evidence_multirate_declaration_20261104.json"
        cfg["gas_spring_compensation_evidence"] = dict(file=evidence, sha256=base.digest(ROOT/evidence))
    if "native_environment_layout" in plan:
        row["contract"]["native_environment_layout"] = plan["native_environment_layout"]
    for key in row["contract"]:
        if key in cfg:
            row["contract"][key] = deepcopy(cfg[key])
    row["runtime"]["contact_material"] = deepcopy(cfg["contact_material"])
    row["contract"]["identified_wheel_response"] = deepcopy(cfg["identified_wheel_response"])
    row["runtime"]["identified_wheel_response"] = deepcopy(cfg["identified_wheel_response"])
    row["runtime"]["actuator_response"] = deepcopy(cfg["actuator_response"])
    row["runtime"]["modules"]["identified_wheel_response"] = True
    row["runtime"]["wheel_velocity_p"] = wheel_control["actuators"]["wheel"]["kd"]
    row["control"]["actuators.wheel"] = deepcopy(wheel_control["actuators"]["wheel"])
    row["control"]["timing"] = deepcopy(wheel_control["timing"])
    row["runtime"].update(rates)
    if "feedback_control_dt" in cfg:
        row["contract"]["feedback_control_dt"] = cfg["feedback_control_dt"]
    row["runtime"]["gas_spring_compensation"] = deepcopy(cfg["v5_control"].get("gas_spring_compensation"))
    if "leg_motor_friction" in cfg:
        row["runtime"]["leg_motor_friction"] = deepcopy(cfg["leg_motor_friction"])
        row["runtime"]["modules"]["leg_motor_friction"] = True
    design["scope"] = "fresh_prone_motion_long"
    base.write_json(design_path, design)
    cfg["design_preflight"]["sha256"] = base.digest(design_path)
    base.write_json(output/"config/contract.json", cfg)
    base.write_json(output/"task_contract.json", dict(schema="prone-motion-native-task-v1", plan=plan,
        settings=settings, actor_dim=35, critic_dim=cfg["critic_dim"], action_dim=6,
        full_vehicle=deepcopy(full_vehicle), gimbal_motion=deepcopy(cfg.get("gimbal_motion")),
        policy_hz=round(1./rates["policy_dt"]), physics_hz=round(1./rates["physics_dt"]),
        feedback_pd_hz=round(1./rates["pc_control_dt"]),
        timing=rates, rollout_steps=plan["rollout_steps"], discounts=discount,
        simulation_timing_approximation=deepcopy(plan.get("simulation_timing_approximation")),
        contact_sampling_ms=rates["physics_dt"]*1000.,
        fine_contact_tolerance_resolved=rates["physics_dt"]*1000. <= plan["evaluation"]["ignored_contiguous_support_loss_ms"],
        solver_iterations={key:cfg[key] for key in ("position_iterations", "velocity_iterations")},
        standing_odometry=deepcopy(plan.get("standing_odometry")),
        learning_interface=deepcopy(plan.get("learning_interface")),
        encoder_increment_actions=deepcopy(plan.get("encoder_increment_actions")),
        terminal_transition=deepcopy(plan.get("terminal_transition")),
        cold_evaluation_lifecycle=deepcopy(plan.get("cold_evaluation_lifecycle")),
        getup_translation=deepcopy(plan.get("getup_translation")),
        jump_training=deepcopy(plan.get("jump_training")),
        policy_sensor_delay_ms=[v*rates["policy_dt"]*1000. for v in settings["sensor_delay_policy_ticks"]],
        wheel_reduction=cfg["wheel_reduction"], wheel_action="trainable absolute output-shaft velocity",
        identified_wheel_response=deepcopy(cfg["identified_wheel_response"]),
        leg_motor_friction=deepcopy(cfg.get("leg_motor_friction")),
        multi_pose_recovery=deepcopy(plan.get("multi_pose_recovery")), recovery_impact=deepcopy(plan.get("recovery_impact")),
        gas_spring_feedforward=deepcopy(cfg["v5_control"].get("gas_spring_compensation") or
            cfg["v5_control"]["encoder_spring_feedforward"]),
        action_order=deepcopy(plan["encoder_increment_actions"]["policy_order"]) if "encoder_increment_actions" in plan else
            ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        wheel_velocity_scale_rad_s=10., wheel_velocity_clip_rad_s=90.,
        leg_position_scale_rad=None if "encoder_increment_actions" in plan else .25,
        leg_action_clip=settings["leg_action_clip"], profile=profile, strength=strength,
        full_randomized_holdout=evaluation and profile != "nominal", state_writes="reset only",
        mechanism_config_sha256=base.digest(output/"config/contract.json"), hardware_ready=False,
        training_origin=plan["initialization"], static_load_initialization=None))
    from train_chassis import preflight
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.task import choose_scene_groups
    choose_scene_groups(cfg["scene_groups"], args.num_envs)
    preflight(output/"config/contract.json")
    check = design_preflight(cfg, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
        stage="speed", role="train", seed=args.seed, num_envs=args.num_envs)
    base.write_json(output/"design_preflight.json", check)
    return cfg, settings, args.bank


def learner_config(plan, device="cuda:0"):
    from wheeled_algo.scratch_standing import agent_config
    from wheeled_algo.recovery_action_distribution import std_bounds
    from wheeled_tasks.chassis.prone_motion import discounts
    result = agent_config(plan["seed"], device, plan["rollout_steps"], plan["learning_rate"],
                          normalize_observations=True)
    result["actor"]["distribution_cfg"]["std_range"] = list(std_bounds(plan))
    ppo = plan.get('ppo',dict(hidden_dims=[256,128,64], epochs=3, mini_batches=4))
    result["actor"]["hidden_dims"] = list(ppo['hidden_dims'])
    result["critic"]["hidden_dims"] = list(ppo['hidden_dims'])
    if plan.get("standing_odometry",{}).get("schema") == "wheel-leg-local-standing-odometry-v2":
        for key in ("actor", "critic"):
            result[key]["class_name"] = "wheeled_algo.prone_observation:ProneMotionModel"
    if "learning_interface" in plan:
        from wheeled_algo.prone_initialization import validate as validate_interface
        validate_interface(plan["learning_interface"])
        if (len(ppo["hidden_dims"]) < 2 or ppo["hidden_dims"][0] < 16
                or any(size < 4 for size in ppo["hidden_dims"][1:])):
            raise ValueError("Pose-hold initialization needs first16 and later4 hidden units")
        for key in ("actor", "critic"):
            result[key]["class_name"] = "wheeled_algo.prone_observation:PhysicalScaleProneMotionModel"
    result['algorithm'].update(num_learning_epochs=ppo['epochs'],num_mini_batches=ppo['mini_batches'])
    result["algorithm"]["entropy_coef"] = .001
    result["algorithm"].update(discounts(plan))
    result.update(max_iterations=plan["max_updates"], save_interval=plan["snapshot_interval"], experiment_name="prone_motion_long")
    return result


def check_first_update(result, runner_cfg):
    """Validate native PPO work against this run's actual epoch/batch budget."""
    algorithm = runner_cfg['algorithm']
    expected = algorithm['num_learning_epochs']*algorithm['num_mini_batches']
    checks = dict(finite_recorded_values=result['all_recorded_values_finite'],
        configured_optimizer_calls=result['committed_optimizer_calls'] == expected,
        actor_changed=result['actor_changed_parameter_tensors'] > 0,
        critic_changed=result['critic_changed_parameter_tensors'] > 0)
    result.update(expected_optimizer_calls=expected, checks=checks)
    return all(checks.values())


def source_hashes(plan_path, plan):
    result = base.sources()
    names = ["src/wheeled_algo/recovery_action_distribution.py", "tests/test_recovery_action_distribution.py",
             "scripts/train_prone_motion.py", "scripts/evaluate_prone_motion.py", "scripts/train_prone_motion.sh",
             "scripts/play_prone_motion.py", "scripts/play_prone_motion.sh", "tests/test_prone_motion_cli.py",
             "scripts/benchmark_prone_motion.py", "src/wheeled_tasks/chassis/prone_motion_sampling.py",
             "tests/test_prone_motion_sampling.py",
             "tests/test_prone_motion_wheelbipe.py",
             "scripts/server_isaac_python.sh", "scripts/resolve_isaac_python.sh",
             "tests/test_prone_motion_python.py", "docs/SERVER_TRAINING_20261028.md",
             "tests/test_prone_motion.py", "docs/PRONE_MOTION_TRAINING_20261028.md",
             "contracts/scratch_standing_base_mechanism_v2.json", "contracts/scratch_standing_base_design_v2.json",
             "contracts/wheel_leg_prone_motion_v1.json", "contracts/wheel_leg_prone_motion_v2.json",
             "contracts/wheel_leg_prone_motion_v3.json",
             "contracts/wheel_leg_prone_motion_v4.json", "tests/test_standing_local_odometry.py",
             "src/wheeled_tasks/chassis/native_wheel_encoder.py", "src/wheeled_tasks/chassis/standard_getup_reward.py",
             "tests/test_standard_getup_learning.py", "scripts/diagnose_standard_getup.py",
             "tests/test_standard_getup_sequence.py", "docs/STANDARD_GETUP_SEQUENCE_20261103.md",
             "tests/test_standard_getup_standing_bridge.py", "docs/STANDARD_GETUP_STANDING_BRIDGE_20261103.md",
             "tests/test_closure_record_only.py", "docs/CLOSURE_RECORD_ONLY_20261103.md",
             "contracts/wheel_leg_prone_motion_v4_before_closure_record_only_20261103.json",
             "contracts/wheel_leg_prone_motion_v4_before_standing_bridge_20261103.json",
             "contracts/wheel_leg_prone_motion_v4_before_sequence_20261103.json",
             "contracts/wheel_leg_prone_motion_v4_before_getup_fix_20261102.json",
             "contracts/wheel_leg_prone_motion_v4_before_repair_20261102.json",
             "contracts/wheel_leg_prone_motion_v4_multi_pose_repair_20261102.json",
             "docs/STANDING_ODOMETRY_20261030.md",
             plan["randomization_contract"],
             plan["acceptance_contract"], plan["prone_geometry_contract"], "contracts/standing_geometry107_v1.json",
             plan["gas_spring_model"]["file"], "tests/test_training_gas_spring.py",
             "tests/test_encoder_spring_feedforward.py",
             "tests/test_prone_motion_performance.py", "tests/test_prone_motion_timing.py",
             "contracts/wheel_leg_prone_motion_control_policy100_pd200_v1.json",
             "contracts/wheel_leg_prone_motion_v4_1000hz_20261101.json",
             "docs/PRONE_MOTION_200HZ_20261101.md", "tests/test_prone_motion_200hz.py",
             "tests/test_unified_training.py", "tests/data/unified_dev_gas_reference.json",
             "tests/data/unified_dev_gas_reference.npz", "docs/WHEEL_LEG_UNIFIED_20261029.md",
             "src/wheeled_algo/continuous_rollout.py", "tests/test_continuous_rollout.py",
             "src/wheeled_algo/training_control.py", "tests/test_training_control.py", "docs/TRAINCTL.md",
             "model/gas_spring_compensation_unified_20261029/evidence_sampling_layout_20261102.json",
             "model/gas_spring_compensation_unified_20261029/evidence_multirate_declaration_20261104.json",
             "scripts/review_gas_spring_soft_range.py",
             "tests/test_prone_motion_resume.py", "docs/PRONE_MOTION_RESUME.md",
             str(plan_path.relative_to(ROOT))]
    if plan.get("simulation_timing_approximation", {}).get("model") == "physics1000hz_policy100hz_feedback_pd200hz_hold_v1":
        names.extend(["tests/test_fine_pd200.py", "scripts/build_standard_prone_timing_bank.py",
                      "docs/PRONE_MOTION_FINE_PD200_20261104.md",
                      "reports/standard_getup_multirate_repair_20261104/passive_002/report.json",
                      "reports/standard_getup_multirate_repair_20261104/passive_probe_plan.input.json"])
    if plan.get("paired_torque_saturation") is not None:
        names.extend(["tests/test_paired_torque_saturation.py", "docs/STANDARD_GETUP_SHAPE_PRIORITY_20261103.md"])
    if "terminal_transition" in plan:
        names.extend(["tests/test_hard_fault_terminal.py", "docs/HARD_FAULT_TERMINAL_20261103.md",
                      "contracts/wheel_leg_prone_motion_v4_physical_interface_v1.json"])
    if "cold_evaluation_lifecycle" in plan:
        names.extend(["tests/test_cold_evaluation_lifecycle.py", "docs/COLD_EVALUATION_LIFECYCLE_20261103.md",
                      "contracts/wheel_leg_prone_motion_v4_terminal_settlement_v1.json"])
    if "getup_translation" in plan:
        names.extend(["src/wheeled_tasks/chassis/getup_translation.py", "tests/test_getup_translation.py",
                      "docs/GETUP_TRANSLATION_20261103.md",
                      "contracts/wheel_leg_prone_motion_v4_cold_lifecycle_v1.json"])
    if "learning_interface" in plan:
        names.extend(["tests/test_prone_learning_interface.py", "docs/STANDARD_GETUP_LEARNING_INTERFACE_20261103.md",
                      "contracts/wheel_leg_prone_motion_v4_shape_priority_v1.json"])
    if "encoder_increment_actions" in plan:
        names.extend(["tests/test_recovery_actions.py", "docs/RECOVERY_ACTIONS.md"])
    if "trend_extension" in plan["gas_spring_model"]:
        names.append(plan["gas_spring_model"]["trend_extension"]["file"])
    if plan.get("gas_spring_backend") == "dev_compensation_v2":
        gas_cfg = json.loads(base.BASE_CONFIG.read_text())
        gas_evidence = json.loads((ROOT/gas_cfg["gas_spring_compensation_evidence"]["file"]).read_text())
        names.extend(gas_evidence["files_sha256"])
        names.append(gas_evidence["basis"]["file"])
        if "previous_binding" in gas_evidence:
            names.append(gas_evidence["previous_binding"]["file"])
    names.extend(item["file"] for item in plan["wheel_model"].values() if isinstance(item, dict) and "file" in item)
    if plan.get("full_vehicle"):
        directory = (ROOT/plan["mechanism"]["file"]).parent
        recipe = json.loads((ROOT/plan["full_vehicle"]["file"]).read_text())
        source = ROOT/recipe["source_bundle"]
        names.extend(str(p.relative_to(ROOT)) for folder in (directory, source)
            for p in folder.rglob("*") if p.is_file())
        if plan.get("leg_motor_friction"):
            model_directory = (ROOT/plan["leg_motor_friction"]["model_path"]).parent
            names.extend(str(p.relative_to(ROOT)) for p in model_directory.iterdir() if p.is_file())
            names.extend(["scripts/build_leg_friction_v202.py", "scripts/probe_leg_motor_friction.py", "tests/test_leg_motor_friction.py",
                "docs/LEG_MOTOR_FRICTION_20261031.md"])
            names.extend(str(p.relative_to(ROOT)) for p in source.parent.glob("*.json"))
        binding = recipe["parent_component_validation"]
        names.extend([binding["file"], "model/wheel_leg_v2.0.2/build_receipt.json", "scripts/build_full_vehicle_training.py", "scripts/train_chassis.py",
                      "scripts/sync_full_vehicle_knee_limits.py", "tools/full_robot_v202.py",
                      "src/wheeled_tasks/chassis/source_chassis_collision.py",
                      "model/wheel_leg_v2.0.2/chassis_collision/geometry.json",
                      "model/wheel_leg_v2.0.2/chassis_collision/robot.usda",
                      "tests/test_source_chassis_collision.py", "docs/CHASSIS_COLLISION_GEOMETRY.md",
                      "model/wheel_leg_v2.0.2/knee_limits_sync_receipt.json", "model/wheel_leg_v2.0.2/source_provenance.json",
                      "tests/test_full_vehicle_training.py", "docs/FULL_VEHICLE_TRAINING_20261030.md"])
        receipt = json.loads((ROOT/binding["file"]).read_text())
        names.extend(run["path"] for run in receipt["native_runs"])
    if plan.get("multi_pose_recovery") is not None:
        if plan["multi_pose_recovery"].get("dense_reward", {}).get("schema") == "wheel-leg-phase-recovery-reward-v1":
            names.extend(["tests/test_phase_recovery.py", "docs/SELF_RIGHTING_EXPERIMENTS.md"])
        if plan["multi_pose_recovery"].get("require_native_reset_admission"):
            names.extend(["scripts/build_native_admitted_recovery_bank.py", "scripts/probe_recovery_reset_admission.py",
                "scripts/build_standard_prone_repair_bank.py", "tests/test_standard_prone_repair.py",
                "tests/test_training_stability_repair.py", "docs/TRAINING_STABILITY_REPAIR_20261102.md"])
            names.extend(str(p.relative_to(ROOT)) for p in (ROOT/plan["pose_bank"]["file"]).parent.glob("*.json"))
            names.extend(str(p.relative_to(ROOT)) for p in (ROOT/"model/recovery_v202_20261102").glob("*.json"))
        names.extend([plan["pose_bank"]["file"], "model/recovery_v202_20261031/pose_bank.build.json",
            "scripts/build_multi_pose_recovery_bank.py", "scripts/probe_multi_pose_recovery.py",
            "contracts/wheel_leg_prone_motion_v4_standard_20261031.json", "scripts/evaluate_recovery_mastery.py",
            "scripts/validate_recovery_continuation.py", "tests/test_recovery_mastery.py",
            "tests/test_multi_pose_recovery.py", "tests/test_recovery_impact.py",
            "docs/MULTI_POSE_RECOVERY_20261031.md"])
        if plan.get("global_curriculum") is not None:
            if plan["global_curriculum"].get("recovery_focus") is not None:
                names.append("tests/test_recovery_discovery.py")
            if plan["multi_pose_recovery"].get("start_curriculum") is not None:
                names.append("tests/test_recovery_start_curriculum.py")
            names.extend(["contracts/wheel_leg_prone_motion_v4_recovery_20261031.json",
                "scripts/evaluate_global_curriculum.py","tests/test_global_curriculum.py",
                "scripts/probe_global_curriculum.py",
                "docs/GLOBAL_MASTERY_CURRICULUM_20261031.md"])
        reset_bank = json.loads((ROOT/plan["pose_bank"]["file"]).read_text())
        names.extend(reset_bank["multi_pose_recovery"]["source_mesh_sha256"])
    evidence = json.loads((ROOT/plan["wheel_model"]["evidence"]["file"]).read_text())
    names.extend(evidence["files_sha256"])
    names.extend(["tests/test_identified_wheel_response.py", "tests/test_training_wheel_response.py"])
    if plan.get("jump_training") is not None:
        names.extend(["scripts/probe_jump_motion.py", "scripts/train_jump_motion.sh",
                      "tests/test_jump_motion.py", "docs/JUMP_MOTION_TRAINING.md"])
    for name in names:
        result[name] = base.digest(ROOT/name)
    return result


def check_other_learners():
    names = {"train_009_curriculum.py", "train_standard_prone.py", "train_upright_locomotion.py",
             "train_scratch_standing.py", "train_fresh_unified.py", "train_prone_motion.py"}
    owners = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors="replace") for x in (proc/"cmdline").read_bytes().split(b"\0") if x]
            if argv and "python" in Path(argv[0]).name and any(Path(a).name in names for a in argv[1:3]):
                owners.append(dict(pid=int(proc.name), argv=argv[:3]))
        except OSError:
            pass
    if owners:
        raise RuntimeError("Refusing a concurrent wheel-leg learner. Let it finish or stop it first: "+str(owners))


def verify_native_probe(output):
    """Kit shutdown can mask an exception's exit code; require saved evidence."""
    result = json.loads((Path(output)/"progress.json").read_text())
    probes = result.get("native_probes", [])
    if (result.get("status") != "native_probe_completed_unqualified"
            or result.get("completed_updates") != 0 or result.get("optimizer_attempts") != 0
            or result.get("checkpoint_reads") != [] or result.get("source_mismatches_at_exit") != []
            or result.get("all_native_checks_pass") is not True or len(probes) != 2
            or [(c["profile"], c["strength"]) for c in result.get("contexts", [])]
                != [("nominal", 0.), ("requested", 1.)]
            or any(p.get("all_checks_pass") is not True or p.get("new_PPO") != 0
                or not p.get("checks") or not all(v is True for v in p["checks"].values()) for p in probes)):
        detail = "\nSaved native exception:\n"+result["error"] if result.get("error") else ""
        raise RuntimeError("Native probe did not finish with two verified domains; refusing PPO: "+str(output)+detail)
    return result


def stop_previous_009():
    """Only the user's explicit CLI flag permits retiring the known old run."""
    run = Path("/home/noir/Documents/workspace/isaac_wheeled_rl_train/runs/reproduce009_20261028/standing_tracking_clockfix_v3_u3072_e256")
    progress_file = run/"progress.json"
    value = json.loads(progress_file.read_text())
    pid = value["pid"]
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        if value["status"] not in ("stopped", "completed_basic_scope_unqualified", "completed_gate_failed_unqualified"):
            raise RuntimeError("Old009 disappeared without a reviewed final status; inspect "+str(progress_file))
        return
    argv = [s.decode() for s in (proc/"cmdline").read_bytes().split(b"\0") if s]
    ticks = int((proc/"stat").read_text().rsplit(")",1)[1].split()[19])
    expected_cwd = Path("/home/noir/.codex/worktrees/standard-prone-training/isaac_wheeled_rl_train")
    if (ticks != 13480232 or value["start_ticks"] != ticks or pid != 2929818
            or (proc/"cwd").resolve() != expected_cwd
            or len(argv) < 4 or Path(argv[1]).name != "train_009_curriculum.py"
            or "--output" not in argv or Path(argv[argv.index("--output")+1]).resolve() != run):
        raise RuntimeError("Old009 identity changed; refusing to signal any process")
    print(f"Identity verified. Sending one SIGTERM to old009 PID{pid}; waiting for its complete-update checkpoint.", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+600.
    while proc.exists() and time.monotonic() < deadline:
        time.sleep(.5)
    value = json.loads(progress_file.read_text())
    checkpoint = value.get("last_checkpoint", {})
    if (proc.exists() or value["status"] != "stopped" or checkpoint.get("updates") != value["completed_updates"]
            or Path(checkpoint.get("path", "")).name != "model_stopped.pt"
            or not (run/"model_stopped.pt").is_file()):
        raise RuntimeError("Old009 has not sealed and exited cleanly; no new learner will start")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    if str(pid) in gpu.stdout.split():
        raise RuntimeError("Old009 still owns a GPU context; refusing to start")
    print(f"Old009 sealed at{value['completed_updates']} updates. No weights will be reused.", flush=True)


def main():
    from wheeled_tasks.chassis.prone_motion_cli import train_parser, training_options
    parser = train_parser()
    args = parser.parse_args()
    try:
        plan_path, plan, output = training_options(args, ROOT)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.print_output:
        print(output)
        return
    from wheeled_tasks.chassis.prone_motion import domain_stage, snapshot_name
    from wheeled_algo.prone_motion_resume import (
        metadata, authenticate_sources, validate_checkpoint, restore, restore_rng, restore_curriculum,
    )
    resume_bundle = metadata(args.resume) if args.resume is not None else None
    if args.stop_existing_009:
        if args.preflight_only:
            parser.error("A CPU preflight must not signal an existing learner")
        stop_previous_009()
    if not args.preflight_only:
        check_other_learners()
    output.mkdir(parents=True, exist_ok=False)
    (output/"curriculum_plan.input.json").write_bytes(plan_path.read_bytes())
    base.write_json(output/"curriculum_plan.resolved.json", plan)
    run_options = dict(argv=sys.argv[1:], task=args.task, output=str(output), run_name=args.run_name,
        device=args.device, headless=args.headless, performance_backend=args.performance_backend,
        sampling_backend=args.sampling_backend,
        python_executable=sys.executable, python_prefix=sys.prefix,
        options=plan["runtime_cli"]["options"], base_plan=str(plan_path),
        resolved_plan_sha256=base.digest(output/"curriculum_plan.resolved.json"))
    base.write_json(output/"run_options.json", run_options)
    initial_profile, initial_strength = ("nominal", 0.) if args.native_probe else domain_stage(plan, 0)
    cfg, settings, bank = configuration(output, plan, profile=initial_profile, strength=initial_strength,
        performance_backend=args.performance_backend, device=args.device, sampling_backend=args.sampling_backend)
    hashes = source_hashes(plan_path, plan)
    resume_sources = authenticate_sources(resume_bundle, hashes, ROOT) if resume_bundle else None
    if args.preflight_only:
        base.write_json(output/"preflight.json", dict(source_hashes=hashes, checkpoint_reads=[], new_PPO=0,
            resume_source_validation=resume_sources,
            run_options=run_options,
            max_updates=plan["max_updates"], rollout_slots=plan["max_updates"]*plan["num_envs"]*plan["rollout_steps"],
            training_origin=plan["initialization"], initialized_behavior_qualified=False))
        return
    committed = base.archive_committed_sources(hashes, output)
    base.write_json(output/"source_hashes.json", hashes)
    (output/"pose_bank.json").write_bytes(bank.read_bytes())
    base.write_json(output/"input_hashes.json", dict(bank_sha256=base.digest(bank), plan_sha256=base.digest(plan_path),
        resolved_plan_sha256=run_options["resolved_plan_sha256"]))
    report = dict(schema="prone-motion-progress-v1", status="starting", pid=os.getpid(),
        start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        started_utc=datetime.now(timezone.utc).isoformat(), requested_updates=plan["max_updates"],
        completed_updates=0, optimizer_attempts=0, checkpoint_reads=[], contexts=[], cold_reviews=[],
        committed_source=committed, run_options=run_options, hardware_ready=False, full_capability_qualified=False)
    resume_state = None
    start_updates = 0
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        report["requested_stop_signal"] = signum
    launcher = env = runner = audit = original_load = domain_switch = mastery_hook = None
    global_course = None
    mastery = None
    if plan.get("multi_pose_recovery",{}).get("mastery_gate") is not None:
        from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
        mastery = RecoveryMastery(plan["multi_pose_recovery"]["mastery_gate"])
    if plan.get("global_curriculum") is not None:
        from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
        global_course = GlobalCurriculum(plan,mastery)
    active_directory = output
    start = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True if args.native_probe else args.headless, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        import numpy as np
        import torch
        from isaaclab.sim import SimulationContext
        from rsl_rl.runners import OnPolicyRunner
        if plan.get("jump_training") is not None:
            from wheeled_tasks.chassis.jump_motion_env import JumpMotionEnv as ProneMotionEnv
        else:
            from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.randomization_study import runtime_samples
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.native_ppo_audit import NativePPOAudit
        from wheeled_algo.continuous_rollout import RolloutDomainSwitch
        if plan.get("jump_training") is not None:
            from probe_jump_motion import probe
        else:
            from evaluate_prone_motion import probe
        torch.set_num_threads(4)
        torch.manual_seed(plan["seed"])
        random.seed(plan["seed"])
        np.random.seed(plan["seed"])
        base.torch = torch
        original_load = torch.load
        if resume_bundle is not None and not args.native_probe:
            if base.digest(resume_bundle["path"]) != resume_bundle["sha256"]:
                raise ValueError("Resume checkpoint changed after provenance validation")
            resume_state = original_load(resume_bundle["path"], map_location="cpu", weights_only=False)
            start_updates = validate_checkpoint(resume_state, resume_bundle)
            mastery, global_course = restore_curriculum(resume_state, plan)
            report.update(completed_updates=start_updates, optimizer_attempts=start_updates,
                checkpoint_reads=[dict(path=str(resume_bundle["path"]), sha256=resume_bundle["sha256"],
                                       updates=start_updates, purpose="explicit_interrupted_run_resume")],
                resume_source_validation=resume_sources)
            base.write_json(output/"progress.json", report)
        def forbidden_load(*a, **kw):
            report["checkpoint_reads"].append(str(a[0] if a else kw.get("f")))
            raise RuntimeError("Checkpoint/teacher/donor loads are forbidden")
        torch.load = forbidden_load

        def build(profile="nominal", strength=0., evaluation=False):
            nonlocal env, active_directory
            jump_counters = (env.jump_training.total.clone() if env is not None and
                getattr(env, "jump_training", None) is not None else None)
            changed = [name for name, expected in hashes.items() if base.digest(ROOT/name) != expected]
            if changed:
                raise RuntimeError("Frozen source or input changed; refusing a new native context: "+str(changed))
            if env is not None:
                env.close()
                if runner is not None:
                    runner.env = None
                env = None
                SimulationContext.clear_instance()
                gc.collect()
                torch.cuda.empty_cache()
            index = len(report["contexts"])
            runtime_seed = plan["evaluation"]["seed"] if evaluation else plan["seed"]+index*31
            active_directory = output/"contexts"/f"{index:03d}_{profile}_{strength:g}_{'eval' if evaluation else 'train'}"
            active_directory.mkdir(parents=True, exist_ok=False)
            cfg, settings, _ = configuration(active_directory, plan, profile=profile, strength=strength,
                evaluation=evaluation, num_envs=80 if args.native_probe else plan["num_envs"], seed=runtime_seed,
                performance_backend=args.performance_backend, device=args.device, sampling_backend=args.sampling_backend)
            manifest = json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text())
            env = ProneMotionEnv(cfg, manifest, load_chassis_control(ROOT/cfg["control_math_source"]), ROOT,
                stage_name="speed", num_envs=cfg["target_num_envs"], device=args.device,recovery_mastery=mastery,
                global_curriculum=global_course,
                seed=runtime_seed,
                level=1., run_role="train", bank=output/"pose_bank.json", settings=settings)
            env.actor_updates = report["completed_updates"]
            if getattr(env, "jump_training", None) is not None:
                counters = jump_counters if jump_counters is not None else (
                    resume_state["infos"]["jump_training_counters"] if resume_state is not None else None)
                if counters is not None:
                    env.jump_training.total.copy_(counters.to(env.device))
            env.reset(torch.arange(env.num_envs, device=env.device))
            if not args.headless and not args.native_probe:
                native_step = env.step
                last_render = 0.
                def visible_step(actions):
                    nonlocal last_render
                    result = native_step(actions)
                    now = time.monotonic()
                    if now-last_render >= 1./30.:
                        pose = env.robot.data.root_link_pose_w.torch[0, :3].cpu().tolist()
                        env.sim.set_camera_view((pose[0]+1.7, pose[1]-1.7, pose[2]+.8), pose)
                        env.sim.render()
                        last_render = now
                    return result
                env.step = visible_step
            if runner is not None:
                runner.env = env
            report["contexts"].append(dict(index=index, profile=profile, strength=strength,
                evaluation=evaluation, directory=str(active_directory), completed_updates=report["completed_updates"],
                seed=runtime_seed))
            base.write_json(active_directory/"native_startup.json", env.startup_report)
            base.write_json(active_directory/"domain_samples.json", runtime_samples(env))
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            base.write_json(output/"progress.json", report)
            print(f"[prone-motion] Native context ready: {profile}, strength={strength:g}, "
                  f"completed={report['completed_updates']}/{plan['max_updates']}", flush=True)

        build(*(("nominal", 0.) if args.native_probe else domain_stage(plan, start_updates, global_course)))
        if args.native_probe:
            reports = []
            report["native_probes"] = reports
            desired = [("nominal", 0.)]
            probe_runner = SimpleNamespace(env=env, device=args.device, logger=SimpleNamespace(writer=None),
                alg=SimpleNamespace(actor=object(), critic=object(), optimizer=object(),
                    storage=SimpleNamespace(step=0), act=lambda observations: observations))
            def probe_rebuild(profile, strength):
                build(profile, strength)
                probe_runner.env = env
            probe_switch = RolloutDomainSwitch(probe_runner, completed=lambda: 0,
                desired=lambda _: desired[0],
                current=lambda: (report["contexts"][-1]["profile"], report["contexts"][-1]["strength"]),
                rebuild=probe_rebuild)
            for profile, strength in (("nominal", 0.), ("requested", 1.)):
                if stop_requested:
                    raise base.TrainingStopped()
                desired[0] = (profile, strength)
                with torch.inference_mode():
                    observation = probe_runner.alg.act(probe_runner.env.get_observations())
                    expected = env.get_observations()
                fresh = all(torch.equal(observation[k], expected[k]) for k in expected.keys())
                reports.append(probe(env, plan, stop_requested=lambda: stop_requested))
                base.write_json(active_directory/"native_final.json", env.startup_report)
                reports[-1]["checks"]["continuous_domain_switch_uses_new_observation"] = fresh
                reports[-1]["all_checks_pass"] = all(reports[-1]["checks"].values())
                print(f"[prone-motion] Zero-PPO probe {profile}: "
                      f"{sum(reports[-1]['checks'].values())}/{len(reports[-1]['checks'])} checks", flush=True)
                base.write_json(output/"progress.json", report)
            probe_switch.close()
            report.update(status="native_probe_completed_unqualified", new_PPO=0, native_probes=reports,
                          all_native_checks_pass=all(x["all_checks_pass"] for x in reports))
            if not report["all_native_checks_pass"]:
                raise RuntimeError("Native prone-motion wiring probe failed")
            return
        runner_cfg = learner_config(plan, args.device)
        base.write_json(output/"agent_config.json", runner_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(output), device=args.device)
        if resume_state is None:
            from wheeled_algo.prone_initialization import initialize_prone_fresh
            report["initialization"] = initialize_prone_fresh(runner.alg, base.nominal_action(env), plan)
        else:
            report["initialization"] = restore(runner, resume_state, resume_bundle)
            report["initialization"]["rng_restored"] = restore_rng(resume_state, env)
            # Bootstrap resets happen while constructing the native scene.
            # Begin the resumed episodes from the saved sampler/case rotation.
            env.reset(torch.arange(env.num_envs, device=env.device))
            report["initialization"]["reset_generation_after_reinitialization"] = env.prone_reset_generation
            report["resume"] = dict(checkpoint=str(resume_bundle["path"]), sha256=resume_bundle["sha256"],
                start_updates=start_updates, remaining_updates=plan["max_updates"]-start_updates,
                original_run_last_recorded_updates=resume_bundle["progress"].get("completed_updates"),
                initial_domain=domain_stage(plan, start_updates, global_course), curriculum_restarted=False,
                recovery_mastery=mastery.state_dict() if mastery is not None else None,
                global_curriculum=global_course.state_dict() if global_course is not None else None)
            base.write_json(output/"resume_verification.json", dict(initialization=report["initialization"],
                resume=report["resume"], source_validation=resume_sources))
        base.write_json(output/"initialization.json", report["initialization"])
        original_actor = {k:v.detach().cpu().clone() for k,v in runner.alg.actor.named_parameters()}
        original_critic = {k:v.detach().cpu().clone() for k,v in runner.alg.critic.named_parameters()}
        mask = RecoveryRolloutMask(runner.alg, num_envs=plan["num_envs"])
        def identity():
            return tuple(id(v) for v in (runner.alg, runner.alg.actor, runner.alg.critic, runner.alg.optimizer, runner.alg.storage))
        identities = identity()
        native_update = runner.alg.update

        def save(path, infos=None):
            path = Path(path)
            # The stock runner requests model_0 after ONE completed update.
            # Our periodic hook owns exact250-update snapshots instead.
            number = path.stem.removeprefix("model_")
            if number.isdigit():
                name = snapshot_name(report["completed_updates"], plan)
                if name is None:
                    return
                path = output/name
                previous = report.get("last_checkpoint", {})
                if previous.get("updates") == report["completed_updates"] and previous.get("path") == str(path):
                    return
            state = runner.alg.save()
            state.update(iter=report["completed_updates"], infos=dict(schema="prone-motion-checkpoint-v1",
                training_origin="scratch", continuation_forbidden=False, resume_requires_explicit_cli=True,
                initialization=report["initialization"], resume=report.get("resume"),
                source_revision=committed, source_hashes=hashes, configuration=str(active_directory),
                resolved_plan=deepcopy(plan), run_options=run_options,
                recovery_mastery=mastery.state_dict() if mastery is not None else None,
                global_curriculum=global_course.state_dict() if global_course is not None else None,
                completed_updates=report["completed_updates"], bank_sha256=base.digest(output/"pose_bank.json"),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                env_rng=env.generator.get_state(), domain_rng=env.domain.rng.get_state(),
                recovery_sampler=env.recovery_sampler.state_dict() if hasattr(env, "recovery_sampler") else None,
                prone_reset_generation=env.prone_reset_generation,
                jump_training_counters=(env.jump_training.total.clone() if getattr(env, "jump_training", None) is not None else None),
                python_rng=random.getstate(), numpy_rng=np.random.get_state()))
            temporary = path.with_suffix(path.suffix+".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            report["last_checkpoint"] = dict(path=str(path), sha256=base.digest(path), updates=report["completed_updates"])

        def update():
            nonlocal audit
            report["optimizer_attempts"] += 1
            report["optimizer_update_in_flight"] = True
            base.write_json(output/"progress.json", report)
            losses = native_update()
            report["completed_updates"] += 1
            report["optimizer_update_in_flight"] = False
            context = report["contexts"][-1]
            if "runtime_substep_verification" not in context:
                context["runtime_substep_verification"] = deepcopy(
                    env.startup_report["control_timing"]["runtime_substep_verification"])
                base.write_json(active_directory/"native_startup.json", env.startup_report)
            if identity() != identities or not all(torch.isfinite(p).all() for m in (runner.alg.actor, runner.alg.critic) for p in m.parameters()):
                raise RuntimeError("Learner identity or finite parameter guard failed")
            if report["completed_updates"] == start_updates+1:
                audit.close()
                result = audit.report()
                result.update(actor_changed_parameter_tensors=sum(not torch.equal(original_actor[k], v.detach().cpu()) for k,v in runner.alg.actor.named_parameters()),
                    critic_changed_parameter_tensors=sum(not torch.equal(original_critic[k], v.detach().cpu()) for k,v in runner.alg.critic.named_parameters()))
                passed = check_first_update(result, runner_cfg)
                result.update(completed_updates=report["completed_updates"], start_updates=start_updates,
                              resumed=resume_state is not None)
                base.write_json(output/"first_update_audit.json", result)
                if not passed:
                    raise RuntimeError("First PPO audit failed")
                audit = None
            env.actor_updates = report["completed_updates"]
            report.update(status="training", elapsed_s=time.monotonic()-start, valid_rollout=mask.metrics,
                          last_losses={k:float(v) for k,v in losses.items()})
            if getattr(env, "jump_training", None) is not None:
                report["jump_training"] = env.jump_training.summary()
            if mastery is not None:
                report["recovery_mastery"] = mastery.state_dict()
            if global_course is not None:
                report["global_curriculum"] = global_course.state_dict()
            base.write_json(output/"progress.json", report)
            if stop_requested:
                save(output/"model_stopped.pt")
                raise base.TrainingStopped()
            name = snapshot_name(report["completed_updates"], plan)
            if name is not None:
                save(output/name)
                base.write_json(output/"progress.json", report)
            return losses

        runner.save, runner.alg.update = save, update
        save(output/(f"model_resumed_{start_updates:05d}.pt" if resume_state is not None else "model_initial.pt"))
        audit = NativePPOAudit(runner.alg)

        domain_switch = RolloutDomainSwitch(runner, completed=lambda: report["completed_updates"],
            desired=lambda completed: domain_stage(plan, completed,global_course),
            current=lambda: (report["contexts"][-1]["profile"], report["contexts"][-1]["strength"]),
            rebuild=build)
        if global_course is not None:
            from wheeled_algo.global_curriculum import GlobalCurriculumHook
            mastery_hook = GlobalCurriculumHook(runner,global_course,completed=lambda:report["completed_updates"],
                output=output,stop_requested=lambda:stop_requested)
        elif mastery is not None:
            from wheeled_algo.recovery_mastery import RecoveryMasteryHook
            mastery_hook = RecoveryMasteryHook(runner,mastery,completed=lambda:report["completed_updates"],
                output=output,stop_requested=lambda:stop_requested)
        if stop_requested:
            save(output/"model_stopped.pt")
            raise base.TrainingStopped()
        report["execution"] = dict(mode=plan["execution"], learn_calls=1,
            requested_updates=plan["max_updates"], snapshot_interval=plan["snapshot_interval"],
            start_updates=start_updates, remaining_updates=plan["max_updates"]-start_updates,
            intermediate_cold_reviews=mastery is not None, checkpoint_selection="manual after training")
        base.write_json(output/"progress.json", report)
        remaining_updates = plan["max_updates"]-start_updates
        print(f"[prone-motion] Starting at {start_updates}/{plan['max_updates']}, learn({remaining_updates}); "
              f"snapshots every{plan['snapshot_interval']} COMPLETE updates; "
              f"automatic recovery admission={mastery is not None}; final review is manual.", flush=True)
        from wheeled_algo.training_control import learn as controlled_learn
        controlled_learn(runner, remaining_updates, init_at_random_ep_len=False, name=output.name)
        if report["completed_updates"] != plan["max_updates"] or identity() != identities:
            raise RuntimeError("Continuous training ended before its declared full-update budget")
        if global_course is not None:
            mastery_hook.review(report["completed_updates"],force=True)
            report["global_curriculum"] = global_course.state_dict()
            report["recovery_mastery"] = mastery.state_dict()
        save(output/"model_final.pt")
        report.update(status="completed_unreviewed", elapsed_s=time.monotonic()-start)
    except base.TrainingStopped:
        report.update(status="stopped", elapsed_s=time.monotonic()-start)
        if args.native_probe:
            raise SystemExit(130)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_s=time.monotonic()-start)
        raise
    finally:
        report["source_mismatches_at_exit"] = [n for n,h in hashes.items() if base.digest(ROOT/n) != h]
        base.write_json(output/"progress.json", report)
        if original_load is not None:
            import torch
            torch.load = original_load
        if audit is not None:
            audit.close()
        if mastery_hook is not None:
            mastery_hook.close()
        if domain_switch is not None:
            domain_switch.close()
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush(); runner.logger.writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            try:
                launcher.app.close()
            finally:
                if report["status"] == "failed":
                    raise SystemExit(1)
                if args.native_probe and report["status"] == "stopped":
                    raise SystemExit(130)


if __name__ == "__main__":
    main()
