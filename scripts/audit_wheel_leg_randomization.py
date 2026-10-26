#!/usr/bin/env python3
"""Authenticate archived domain settings and resolve adjustable future studies.

Read-only CPU audit. No native simulation, optimizer or training checkpoint is
constructed. Historical raw receipts retain their original acceptance labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.randomization_study import resolve


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def consequences(task):
    strength = task["motor_strength_range"]
    kp = [a*b for a, b in zip(strength, task["leg_kp_scale_range"], strict=True)]
    kd = [a*b for a, b in zip(strength, task["leg_kd_scale_range"], strict=True)]
    mount = task.get("imu_mount")
    wobble = (math.radians(mount["wobble_degrees"][1])*2.*math.pi*mount["wobble_hz"][1]
              if mount else 0.)
    return dict(effective_unsaturated_leg_kp_scale_bounds=kp,
        effective_unsaturated_leg_kd_scale_bounds=kd,
        imu_wobble_single_axis_gyro_peak_bound_rad_s=wobble,
        imu_wobble_single_axis_gyro_peak_bound_deg_s=math.degrees(wobble),
        gyro_white_sigma_physical_rad_s=task["gyro_noise_rad_s"]*task["sensor_noise_scale"],
        encoder_white_sigma_rad=task["encoder_noise_rad"]*task["sensor_noise_scale"],
        leg_velocity_white_sigma_physical_rad_s=task["velocity_noise_rad_s"][0]*task["sensor_noise_scale"],
        wheel_velocity_white_sigma_physical_rad_s=task["velocity_noise_rad_s"][1]*task["sensor_noise_scale"],
        independent_policy_PD_bias_disagreement_bound_rad=task["encoder_bias_rad"]+task["pd_encoder_bias_rad"],
        bias_applied_inside_PD=True, white_noise_applied_inside_PD=False,
        note="Bounds describe distinct draws, not simultaneous realized extrema or a measured transfer function. P/D multiplication applies before saturation; native motor strength also scales the clipped torque.")


def audit():
    receipts = {}
    def tracked(path):
        path = Path(path)
        receipts[str(path.relative_to(ROOT))] = sha(path)
        return read(path)
    campaign = ROOT / "runs/real_robot_robustness_long_20261024/campaign001"
    plan = tracked(campaign / "plan.json")
    result = tracked(campaign / "result.json")
    keys = ("domain_scale", "gap_severity", "gap_actuator_timing_profile", "domains_enabled", "nominal_fraction",
        "motor_strength_range", "leg_kp_scale_range", "leg_kd_scale_range", "spring_strength_range",
        "encoder_bias_rad", "pd_encoder_bias_rad", "encoder_noise_rad", "gyro_bias_rad_s", "gyro_noise_rad_s",
        "gravity_noise", "velocity_noise_rad_s", "sensor_noise_scale", "imu_mount", "torque_delay_ms",
        "motor_tau_ms", "sensor_delay_policy_ticks", "pd_feedback_delay_ms", "pd_feedback_period_ms")
    stages = []
    for stage in range(1, 5):
        run = campaign / f"train_{stage:03d}"
        progress = tracked(run / "progress.json")
        task_contract = tracked(run / "task_contract.json")
        mechanism = tracked(run / "config/contract.json")
        if sha(run / "task_contract.json") != progress["task_contract_sha256"]:
            raise ValueError("Historical task contract authentication failed")
        source_checks = {}
        for name in ("src/wheeled_tasks/chassis/robust_getup.py", "src/wheeled_tasks/chassis/gap_robustness.py",
                     "src/wheeled_tasks/chassis/robust_getup_env.py", "src/wheeled_tasks/chassis/getup_feedback.py",
                     "src/wheeled_tasks/chassis/dynamics.py", "src/wheeled_tasks/chassis/contact_material.py",
                     "scripts/train_robust_getup.py"):
            path = run / "source" / name
            source_checks[name] = sha(path) == progress["source_sha256"][name]
            receipts[str(path.relative_to(ROOT))] = sha(path)
        if not all(source_checks.values()):
            raise ValueError("Historical frozen implementation changed")
        task = task_contract["settings"]
        startup = progress["startup"]
        domain = startup["robust_domains"]
        dynamics = startup["dynamics_randomization"]
        material = startup["contact_material"]
        # The live startup receipts store the final effective task, after all
        # overlays. Reject disagreement rather than reading CLI intent alone.
        if any(task[k] != domain["domains"][k] for k in keys):
            raise ValueError("Saved task and observed startup domain disagree")
        stages.append(dict(stage=stage, new_completed_PPO=progress["new_optimizer_updates"],
            final_settings={k: task[k] for k in keys}, consequences=consequences(task),
            dynamics=mechanism["dynamics_randomization"], material=mechanism["contact_material"],
            sampled=dict(randomized_lanes=domain["enabled_environments"], nominal_domain_lanes=domain["nominal_environments"],
                motor_strength=[domain["strength_min"], domain["strength_max"]],
                torque_delay_ms_max=domain["torque_delay_ms_max"], sensor_delay_ms_max=domain["sensor_delay_ms_max"],
                dynamics_metrics=dynamics["metrics"],
                ground_friction=material["sampled_ground_static_min_max"],
                effective_wheel_friction=material["effective_wheel_static_min_max"],
                ground_lanes_exactly_nominal=sum(v == .5 for v in material["ground_static_friction_per_environment"])),
            final_settings_match_startup=True, frozen_source_checks=source_checks))
    if sum(s["new_completed_PPO"] for s in stages) != result["new_optimizer_updates"]:
        raise ValueError("Completed historical stages do not match campaign consumption")
    history = [dict(learning_clock=r["learning_clock"], new_updates=r["new_updates"],
        training_severity=r["training_severity"], standing_pose_position_pass=r["standing"]["pose_and_position_pass"],
        motion_vx_mae_m_s=r["motion"]["vx_mae_m_s"], fallen_peak_leg_speed_rad_s=r["fallen"]["max_leg_speed_rad_s"],
        score=r["score"]) for r in result["reviews"]]
    paired = tracked(ROOT / "reports/robust_policy_repair_20261024/stopped1501_paired.json")
    if not all(paired["initial_exact"].values()) or paired["sensor_actuator_material_exact"] is not True:
        raise ValueError("Historical before/after pairing is incomplete")
    holdout = tracked(ROOT / "reports/robust_policy_repair_20261024/holdout_review001.json")
    contact_comparisons = {}
    for name in ("contact073_height_matched019", "contact075_zero_sensor_delay", "contact075_nominal_dynamics"):
        path = ROOT / "reports/robust_policy_repair_20261024" / (name + ".json")
        if not path.exists():
            continue
        d = tracked(path)
        rows = d["rows"]
        # This is only the contact-duration component under the new user
        # allowance; do not promote it to a full standing/robustness pass.
        contact_comparisons[name] = dict(cases=len(rows),
            contact_duration_component_within_2ms_count=sum(r["survived"] and r["whole"]["maximum_interruption_ms"] <= 2 for r in rows),
            maximum_interruption_ms=max(r["whole"]["maximum_interruption_ms"] for r in rows),
            mean_leg_angle_band_rms_mrad=sum(r["tail_leg_motion_3_95hz_rms_rad"]*1000 for r in rows)/len(rows),
            not_a_full_standing_pass=True,
            explicit_IMU_test_profiles_still_present=True)
    current = ROOT / "runs/scratch_standing_20261026/pilot009_leg_velocity_separated"
    task = tracked(current / "task_contract.json")["settings"]
    cfg = tracked(current / "config/contract.json")
    current_settings = dict(domains_enabled=task["domains_enabled"], domain_scale=task["domain_scale"],
        encoder_noise_rad=task["encoder_noise_rad"], gravity_noise=task["gravity_noise"],
        gyro_noise_rad_s=task["gyro_noise_rad_s"], imu_mount=task.get("imu_mount"),
        dynamics_enabled=cfg["dynamics_randomization"]["enabled"],
        ground_friction=cfg["contact_material"]["ground_static_friction_range"])
    return dict(schema="wheel-leg-randomization-audit-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
        current009=current_settings, old_campaign_stages=stages,
        actually_completed_severity_max=max(s["final_settings"]["gap_severity"] for s in stages),
        scheduled_but_unreached_severity=1., completed_new_PPO=result["new_optimizer_updates"],
        history=history, fixed_before_after_pairing=dict(seed=paired["seed"], eval_severity=paired["severity"],
            initial_exact=paired["initial_exact"], sensor_actuator_material_exact=True),
        second_seed_comparison=dict(seed=holdout["seed"], rows=[dict(label=r["label"],
            standing_pose_position_pass=r["standing_pass"], motion_vx_mae_m_s=r["groups"]["motion"]["vx_mae_m_s"])
            for r in holdout["rows"]]),
        archived_contact_duration_diagnostics=contact_comparisons,
        findings=[
            "Two legacy strength knobs: gap_severity overwrites gains, bias, IMU and rigid ranges after domain_scale; it does not multiply domain_scale.",
            "Regression already appears after256 new updates at severity0.6; severity0.8 cannot be the sole trigger.",
            "The fixed before/after evaluation uses the same severity1 domains and exactly paired initial state; learned feedback changed.",
            "Motor strength multiplies both P and D; inertia is scaled by both component mass and an independent inertia scalar.",
            "Moving IMU mount creates coherent angular-velocity disturbance far larger than the gyro white sigma; do not equate it with encoder/IMU white noise.",
            "Policy and PD encoder biases are independent, creating an additional measurement disagreement domain.",
            "Native extra torque/PD delay and motor lag were zero throughout; policy observation delay remained0/5ms.",
            "Nominal domain lanes still had randomized ground friction.25percent nominal did not mean25percent fully nominal scenes.",
            "There is no matched from-scratch training ablation across domain intensity/groups, so no measured optimal intensity or sole causal attribution is claimed."
        ], new_PPO=0, new_native_steps=0, checkpoint_loads=0, hardware_ready=False,
        source_receipts_sha256=receipts)


def write_events(report, directory):
    from rsl_rl.utils.logger import Logger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from wheeled_tasks.chassis.randomization_study import scalar_metrics
    run = ROOT / "runs/real_robot_robustness_long_20261024/campaign001/train_004"
    logger = Logger(log_dir=str(directory), cfg=read(run / "agent_config.json"),
        env_cfg=read(run / "config/contract.json"), num_envs=1, is_distributed=False,
        gpu_world_size=1, gpu_global_rank=0, device="cpu")
    logger.init_logging_writer()
    expected = {}
    def add(tag, value, step):
        logger.writer.add_scalar(tag, value, step)
        expected[tag, step] = value
    try:
        clock = 569
        for stage in report["old_campaign_stages"]:
            clock += stage["new_completed_PPO"]
            task = stage["final_settings"]
            for key in ("gap_severity", "domain_scale", "encoder_bias_rad", "gyro_noise_rad_s"):
                add("DomainAudit/LegacyConfig/" + key, task[key], clock)
            for key in ("effective_unsaturated_leg_kp_scale_bounds", "effective_unsaturated_leg_kd_scale_bounds"):
                for label, value in zip(("min", "max"), stage["consequences"][key], strict=True):
                    add(f"DomainAudit/LegacyConfig/{key}/{label}", value, clock)
            add("DomainAudit/LegacyConfig/imu_wobble_single_axis_gyro_peak_bound_rad_s",
                stage["consequences"]["imu_wobble_single_axis_gyro_peak_bound_rad_s"], clock)
        for row in report["history"]:
            for key in ("standing_pose_position_pass", "motion_vx_mae_m_s", "fallen_peak_leg_speed_rad_s"):
                add("DomainAudit/LegacyFixedEval/" + key, row[key], row["learning_clock"])
        for key, value in scalar_metrics(report["selected"]).items():
            add("DomainStudyPrepared/" + key, value, 0)
        logger.writer.flush()
    finally:
        logger.writer.close()
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    observed = {(tag, row.step): row.value for tag in events.Tags()["scalars"] for row in events.Scalars(tag)}
    if set(observed) != set(expected) or any(not math.isclose(value, observed[key], rel_tol=2e-6, abs_tol=2e-8)
                                           for key, value in expected.items()):
        raise ValueError("Domain audit RSL event readback differs")
    return dict(verified_scalar_values=len(expected), all_values_read_back=True,
        clock="legacy actual completed PPO clock; prepared candidate config uses0 and claims no training",
        files_sha256={str(p.resolve()): sha(p) for p in directory.glob("events.out.tfevents.*")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=ROOT / "contracts/wheel_leg_randomization_study_v1.json")
    p.add_argument("--profile")
    p.add_argument("--strength", type=float, default=1.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.output.with_suffix(".source.py")
    resolver = args.output.with_suffix(".resolver.py")
    study_copy = args.output.with_suffix(".study.json")
    events = args.output.with_suffix(".events")
    if any(path.exists() for path in (args.output, source, resolver, study_copy, events)):
        raise ValueError("Preserve earlier audit receipts")
    spec = read(args.study)
    report = audit()
    report.update(study_input_sha256=sha(args.study),
        resolved_profiles={name: resolve(spec, name, args.strength) for name in spec["profiles"]},
        selected=resolve(spec, args.profile, args.strength), audit_source_sha256=sha(__file__))
    resolver_path = ROOT / "src/wheeled_tasks/chassis/randomization_study.py"
    report["resolver_source_sha256"] = sha(resolver_path)
    report["RSL_events"] = write_events(report, events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    resolver.write_bytes(resolver_path.read_bytes())
    study_copy.write_bytes(args.study.read_bytes())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), completed_new_PPO=report["completed_new_PPO"],
        history=report["history"], selected=report["selected"],
        archived_contact_duration_diagnostics=report["archived_contact_duration_diagnostics"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
