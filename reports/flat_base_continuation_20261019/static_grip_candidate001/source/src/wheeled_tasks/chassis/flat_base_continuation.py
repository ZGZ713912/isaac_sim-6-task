"""Matched 50 Hz actor-transfer experiments with an independent design source."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

PLAN = "contracts/flat_base_finetune_v1.json"
DROP = ("task_campaign", "integrated_production", "design_preflight", "budget_spec_file", "budget_spec_sha256",
        "step_jump_task", "actor_reserved_frame30", "recovery_training", "nonstandard_start", "release_stop",
        "locomotion_difficulty", "ordinary_posture_curriculum", "signal_delay", "landing_tracking",
        "behavior_pool_membership", "behavior_pool_fractions")


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_plan(root, plan_file=PLAN):
    root = Path(root)
    plan = json.loads((root / plan_file).read_text())
    if plan["schema"] not in tuple(f"flat-base-finetune-plan-v{n}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)):
        raise ValueError("Unknown flat actor transfer plan")
    for name in ("source_worker", "source_design", "control"):
        if digest(root / plan[name + "_file"]) != plan[name + "_sha256"]:
            raise ValueError(f"Frozen flat continuation source changed: {name}")
    asset = root / plan["asset_directory"]
    if (digest(asset / "manifest.json") != plan["asset_manifest_sha256"]
            or digest(asset / "model_spec.json") != plan["model_spec_sha256"]):
        raise ValueError("Actor transfer requires the exact ONNX-declared mechanical model")
    cpu = json.loads((root / plan["asset_cpu_evidence"]).read_text())
    if not cpu["passed"] or cpu["bundle_manifest_sha256"] != plan["asset_manifest_sha256"]:
        raise ValueError("Matched asset CPU validation is missing or belongs to another bundle")
    if digest(root / plan["actor_bundle"] / "source.onnx") != plan["source_onnx_sha256"]:
        raise ValueError("Original flat actor changed")
    if (plan["policy_dt"], plan["leg_target_winding"], plan["flat_context"]) != (
            .02, "paired_branch107_v1", [1, 0, 0, 0, 0, 0, 0]):
        raise ValueError("Flat transfer changed its declared50Hz paired-107 interface")
    from .flat_clock_transition import validate_flat_clock
    validate_flat_clock(root, plan)
    velocity_scale = plan.get("moving_velocity_cost_scale")
    if plan["schema"] in ("flat-base-finetune-plan-v16", "flat-base-finetune-plan-v17"):
        if type(velocity_scale) not in (int, float) or not math.isfinite(velocity_scale) or not 1. < velocity_scale <= 10.:
            raise ValueError("v16/v17 require a finite moving velocity cost scale in (1,10]")
    elif velocity_scale is not None:
        raise ValueError("Moving velocity cost scaling requires the explicit v16 plan")
    contact_material = plan.get("contact_material", {"enabled": False})
    if not isinstance(contact_material, dict) or type(contact_material.get("enabled", False)) is not bool:
        raise ValueError("contact_material must be an explicit enabled/disabled mapping")
    if contact_material.get("enabled"):
        if plan["schema"] != "flat-base-finetune-plan-v17":
            raise ValueError("Separate contact materials require the explicit v17 plan")
        from .contact_material import ContactMaterialDomain
        # Validate with the native sampler's isolated generator; this does not
        # consume the environment or learner RNG.
        ContactMaterialDomain(1, contact_material, plan["seed"])
    elif plan["schema"] == "flat-base-finetune-plan-v17":
        raise ValueError("v17 requires enabled separate wheel and ground materials")
    if abs(sum(group["fraction"] for group in plan["train_groups"]) - 1.) > 1e-10:
        raise ValueError("Flat training population must sum to one")
    names = [g["name"] for g in plan["train_groups"]]
    if len(names) != len(set(names)) or any(g["fraction"] <= 0 for g in plan["train_groups"]):
        raise ValueError("Flat training groups must be unique and nonempty")
    for name, seconds in plan.get("group_episode_seconds", {}).items():
        if name not in names or not .5 <= seconds <= plan["episode_seconds"]:
            raise ValueError("Group horizon must name an existing group within the global episode limit")
    stationary = plan.get("stationary_tracking", {})
    if stationary.get("position_cost") == "capped_square":
        floor = stationary["position_weight"] * stationary["position_cost_cap"] * plan["policy_dt"] / (1. - plan["gamma"])
        if stationary.get("failure_event_cost", 0.) + 1e-10 < floor:
            raise ValueError("Stationary failure event cannot be cheaper than avoiding the capped discounted position cost")
    if plan.get("feedback_training_scope") not in (None, "feedback_only"):
        raise ValueError("Unknown feedback training scope")
    if plan.get("feedback_training_scope") and plan.get("actor_adaptation") not in (
            "original-command-and-gated-feedback-residual-v1", "original-command-and-tracking-feedback-residual-v2",
            "original-command-and-separated-tracking-feedback-residual-v3",
            "original-command-and-continuous-low-feedback-residual-v4", "original-command-and-static-recall-feedback-residual-v5", "original-static-feedback-and-wheel-authority-residual-v6", "original-wheel-authority-with-static-grip-adaptation-v7"):
        raise ValueError("Feedback-only scope requires the declared feedback architecture")
    recall = plan.get("static_feedback_recall")
    if plan.get("actor_adaptation") in ("original-command-and-static-recall-feedback-residual-v5", "original-static-feedback-and-wheel-authority-residual-v6", "original-wheel-authority-with-static-grip-adaptation-v7"):
        if (not isinstance(recall, dict) or set(recall) != {"checkpoint", "sha256"}
                or not isinstance(recall["checkpoint"], str)
                or digest(root / recall["checkpoint"]) != recall["sha256"]):
            raise ValueError("Static feedback requires the exact declared recalled checkpoint")
    elif recall is not None:
        raise ValueError("Static feedback checkpoint is only valid for the declared v5 architecture")
    return plan


def _changes(plan, num_envs, *, evaluation=False, plan_file=PLAN):
    """Explicit new requirements, applied separately to worker and design parents."""
    marker = {"schema": "flat-base-actor-transfer-v1", "source_onnx_sha256": plan["source_onnx_sha256"],
              "model_spec_sha256": plan["model_spec_sha256"], "flat_context": plan["flat_context"],
              "plan_file": str(plan_file), "initialization": "exact_actor_mean_fresh_critic_std_optimizer"}
    values = dict(asset_directory=plan["asset_directory"], asset_manifest_sha256=plan["asset_manifest_sha256"],
        control_math_source=plan["control_file"], control_math_sha256=plan["control_sha256"],
        physics_dt=plan["physics_dt"], policy_dt=plan["policy_dt"], action_difference_reference_dt=.02,
        flat_base_continuation=marker, episode_seconds=plan["episode_seconds"],
        num_steps_per_env=plan["num_steps_per_env"], target_num_envs=num_envs,
        total_updates=plan["update_ceiling"], num_learning_epochs=plan["num_learning_epochs"],
        num_mini_batches=plan["num_mini_batches"], learning_rate=plan["learning_rate"],
        learning_rate_schedule="fixed", gamma=plan["gamma"], lam=plan["lam"],
        critic_learning_rate=plan["critic_learning_rate"], use_clipped_value_loss=plan["use_clipped_value_loss"],
        entropy_coef=plan["entropy_coef"], initial_noise_std=plan["initial_std_policy_order"][0],
        critic_warmup_updates=plan["critic_warmup_updates"], transfer_critic=False,
        ppo_minibatch_samples=num_envs * plan["num_steps_per_env"] // plan["num_mini_batches"],
        save_interval=plan["checkpoint_interval"], checkpoint_interval=plan["checkpoint_interval"],
        checkpoint_first_update=1, stages=[dict(name="speed", updates=plan["update_ceiling"],
            vx_max=plan.get("vx_max", .5), yaw_max=plan.get("yaw_max", 1.), push_max=0., terrain=["flat"])],
        usb_transport={"enabled": False}, command_transport={"enabled": False},
        signal_perturbations={"enabled": False}, dynamics_randomization={"enabled": False},
        contact_domain={"enabled": False}, contact_material=deepcopy(plan.get("contact_material", {"enabled": False})),
        step_assist={"enabled": False},
        new_asset_curriculum={"enabled": False}, jump_assist={"enabled": False},
        observation_noise_enabled=False, mass_randomization_enabled=False, diagnostic_trace=evaluation,
        response_migration="flat_base_paired107_actor_transfer_v1", evaluation_exact_cases=evaluation,
        curriculum="flat_actor_transfer_with_external_behavioral_comparison",
        scene_groups=[dict(name=g["name"], fraction=g["fraction"], terrain=["flat"]) for g in plan["train_groups"]],
        skill_specs={g["name"]: deepcopy(g["skill"]) for g in plan["train_groups"]})
    if "flat_floor_width_m" in plan and not evaluation:
        values["flat_floor_width_m"] = plan["flat_floor_width_m"]
    if plan.get("group_episode_seconds") and not evaluation:
        values["flat_group_episode_seconds"] = deepcopy(plan["group_episode_seconds"])
    if plan.get("stationary_tracking") and not evaluation:
        values["stationary_tracking"] = deepcopy(plan["stationary_tracking"])
    if plan.get("critic_position_anchor") and not evaluation:
        if plan.get("critic_dim") != 83 or not plan.get("stationary_tracking"):
            raise ValueError("Position-aware critic requires exactly two privileged anchor inputs")
        values.update(critic_dim=83, flat_critic_anchor_xy=True)
    if plan.get("actor_output_layer_only") and not evaluation:
        values["actor_output_layer_only"] = True
    if plan.get("actor_adaptation") and not evaluation:
        if (plan["actor_adaptation"] not in ("original-feedback-command-residual-v1",
                "original-command-and-gated-feedback-residual-v1", "original-command-and-tracking-feedback-residual-v2",
                "original-command-and-separated-tracking-feedback-residual-v3",
                "original-command-and-continuous-low-feedback-residual-v4", "original-command-and-static-recall-feedback-residual-v5",
                "original-static-feedback-and-wheel-authority-residual-v6", "original-wheel-authority-with-static-grip-adaptation-v7")
                or plan.get("actor_output_layer_only")):
            raise ValueError("Unsupported or conflicting flat actor adaptation")
        values["actor_adaptation"] = plan["actor_adaptation"]
    if plan.get("static_feedback_recall") and not evaluation:
        values["static_feedback_recall"] = deepcopy(plan["static_feedback_recall"])
    if plan.get("training_command_reference") and not evaluation:
        values["flat_rmcs_command_training"] = deepcopy(plan["training_command_reference"])
    return values


def write_bundle(root, directory, num_envs, *, evaluation=False, seed=None, plan_file=PLAN):
    root, directory = Path(root), Path(directory)
    plan = read_plan(root, plan_file)
    if not evaluation and num_envs not in plan["allowed_num_envs"]:
        raise ValueError("Training population differs from the pilot plan")
    source = json.loads((root / plan["source_worker_file"]).read_text())
    table = json.loads((root / plan["source_design_file"]).read_text())
    row = deepcopy(table["stages"]["speed"]["roles"]["train"])
    worker, expected = deepcopy(source), row["contract"]
    for key in DROP:
        worker.pop(key, None)
        expected.pop(key, None)
    changes = _changes(plan, num_envs, evaluation=evaluation, plan_file=plan_file)
    for mapping in (worker, expected):
        mapping.update(deepcopy(changes))
        mapping["v5_control"]["leg_target_winding"] = plan["leg_target_winding"]
        mapping["flat_base_continuation"]["plan_sha256"] = digest(root / plan_file)
        if not evaluation and "moving_velocity_cost_scale" in plan:
            # The dense velocity terms already vanish at zero planar demand.
            # Scale their densities once, keeping the command-visible gates,
            # stationary/height/yaw costs and tracking peaks unchanged.
            for key in ("broad_weight", "fine_weight"):
                mapping["dense_tracking"]["velocity"][key] *= plan["moving_velocity_cost_scale"]
    # Select cases from the separately pinned original catalogue, then apply
    # the explicit tighter flat requirements. No simulation result sets a bar.
    settings = deepcopy(source["evaluation"])
    settings = {key: value for key, value in settings.items() if key not in (
        "quick_case_names", "full_case_names", "promotion_case_names", "frozen_module_enablement",
        "regression_recovery", "protected_case_names")}
    settings.update(plan["acceptance"], protocol_id="flat-base-paired107-fixed-v1", seed=plan["evaluation_seed"],
        confirmation_seed=plan["confirmation_seed"], episodes_per_case=4, mechanical_checks=True,
        protect_anchor_cases=False, stable_case_layout=False, episode_seconds=16.,
        cases=[deepcopy(next(case for case in source["evaluation"]["cases"] if case["name"] == name))
               for name in plan["pilot_case_names"]])
    if plan.get("contact_material", {}).get("enabled"):
        material_id = hashlib.sha256(encode(plan["contact_material"])).hexdigest()[:12]
        settings["protocol_id"] += "-material-" + material_id
    for case in settings["cases"]:
        case.update({key: plan["acceptance"][key] for key in ("survival_rate_min", "height_mae_m_max",
                    "velocity_mae_m_s_max", "yaw_mae_rad_s_max")})
        case["stand_drift_m_max"] = plan["acceptance"]["stand_drift_m_max"]
        if case.get("stationary"):
            case["velocity_mae_m_s_max"] = plan["acceptance"]["stand_velocity_mae_m_s_max"]
    selected_seed = (plan["evaluation_seed"] if evaluation else plan["seed"]) if seed is None else seed
    if evaluation:
        settings["seed"] = selected_seed
        if num_envs != len(settings["cases"]) * settings["episodes_per_case"]:
            raise ValueError("Evaluation requires four equal replicas of each frozen case")
    for mapping in (worker, expected):
        mapping["evaluation"] = deepcopy(settings)
        # Dotted requirements are independent assertions, not extra config keys.
        if mapping is expected:
            mapping["evaluation.seed"] = settings["seed"]
            mapping["evaluation.confirmation_seed"] = settings["confirmation_seed"]
        if evaluation:
            mapping["scene_groups"] = [dict(name=c["name"], fraction=1 / len(settings["cases"]), terrain=["flat"])
                                       for c in settings["cases"]]
            mapping["episode_seconds"] = settings["episode_seconds"]
    control = json.loads((root / plan["control_file"]).read_text())
    quality = {key: deepcopy(worker["dense_tracking"][key]) for key in ("stand_still_lin_vel", "locomotion_posture")
               if key in worker["dense_tracking"]}
    row["invocation"] = dict(num_envs=num_envs, seed=dict(rule="fixed", value=selected_seed))
    row["control"] = {"actuators.wheel": deepcopy(control["actuators"]["wheel"]), "timing": deepcopy(control["timing"])}
    row["runtime"] = dict(physics_dt=plan["physics_dt"], policy_dt=plan["policy_dt"],
        pc_control_dt=control["timing"]["pc_control_dt"], decimation=control["timing"]["decimation"],
        leg_kp=160., leg_kd=2.5, wheel_velocity_p=.6, seed=selected_seed, action_difference_reference_dt=.02,
        precision_tracking=True, actuator_response=deepcopy(worker["actuator_response"]),
        locomotion_quality_rewards=quality,
        modules=dict(usb_transport=False, command_transport=False, signal_perturbations=False,
            dynamics_randomization=False, contact_domain=False,
            contact_material=bool(plan.get("contact_material", {}).get("enabled", False)),
            step_assist=False, actuator_response=True,
            stand_still_lin_vel="stand_still_lin_vel" in quality, locomotion_posture_floor="locomotion_posture" in quality))
    row["runtime"]["contact_material"] = deepcopy(worker["contact_material"])
    role = "quick" if evaluation else "train"
    design = dict(schema="chassis-design-expectations-v1", status="frozen", scope="flat_base_paired107_actor_transfer",
        review_basis=dict(plan_file=str(plan_file), plan_sha256=digest(root / plan_file),
            source_worker_sha256=plan["source_worker_sha256"], source_design_sha256=plan["source_design_sha256"]),
        stages={"speed": {"roles": {role: row}}})
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "design.json").write_bytes(encode(design))
    worker["design_preflight"] = dict(file=str((directory / "design.json").resolve()),
        sha256=digest(directory / "design.json"), stage="speed")
    (directory / "contract.json").write_bytes(encode(worker))
    return worker, selected_seed
