#!/usr/bin/env python3
"""New actor, new critic and empty Adam. Checkpoint continuation is forbidden."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE_CONFIG = ROOT / "contracts/scratch_standing_base_mechanism_v2.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args):
    from wheeled_tasks.chassis.robust_getup import settings, rigid_body_domain
    from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins

    # Reuse the authenticated mechanism description only, never a model file.
    cfg = json.loads(BASE_CONFIG.read_text())
    previous_design = Path(cfg["design_preflight"]["file"])
    if not previous_design.is_absolute():
        previous_design = ROOT / previous_design
    if digest(previous_design) != cfg["design_preflight"]["sha256"]:
        raise ValueError("Mechanism design changed")
    design = json.loads(previous_design.read_text())
    task = settings(domains_enabled=False, domain_scale=0., motion_fraction=1.,
        motion_push_fraction=0., quiet_training=True, quiet_critic_anchor=True,
        quiet_action_rate_cost=0., quiet_motor_speed_cost=0., quiet_effort_delta_cost=0.,
        quiet_spectral_cost=False, standing_support_training=False,
        episode_seconds=8., curriculum_updates=max(1, args.all_heights_after),
        origin_updates=0, native_origin_mode="isolated_overlap_v1")
    task["version"] = "scratch-standing-v1"
    if args.precise_objective:
        task["version"] = "scratch-standing-v3-coupled"
    if args.standing_failures:
        task["version"] += "-failure-resets"
    if args.wheel_velocity_increment:
        task["version"] += "-wheel-increment-v1"
    if getattr(args, "separate_leg_velocity_feedback", False):
        task["version"] += "-separated-leg-velocity-v1"
    if args.static_load_initialization:
        task["version"] += "-static-load-init-v1"
    continuity_contract = None
    if args.contact_break_cost is not None:
        from wheeled_tasks.chassis.standing_contact_continuity import contract
        continuity_contract = contract(args.contact_break_cost)
        task["version"] += "-contact-events-v1"
    target_rate_contract = None
    if getattr(args, "leg_target_rate_cost", None) is not None:
        from wheeled_tasks.chassis.standing_leg_target_rate import contract as target_contract
        target_rate_contract = target_contract(args.leg_target_rate_cost)
        task["version"] += "-leg-target-rate-v1"
    changes = dict(target_num_envs=args.num_envs, num_steps_per_env=args.rollout_steps,
        ppo_minibatch_samples=args.num_envs * args.rollout_steps // 4,
        episode_seconds=8., record_diagnostics=False, full_rl_square_ground=True,
        full_rl_isolated_origins=True, flat_floor_width_m=96., flat_half_length_m=48.,
        dynamics_randomization=rigid_body_domain(False, 0.))
    # These fields select the existing 35D flat context and privileged XY
    # channels. Replace the old transfer provenance and unrelated short groups.
    changes.update(flat_base_continuation=dict(schema="scratch-standing-context-v1",
        flat_context=[1, 0, 0, 0, 0, 0, 0], initialization="none; fresh native PPO"),
        flat_group_episode_seconds={}, curriculum="scratch_standing_foundation",
        critic_warmup_updates=0, resume_critic_warmup_updates=0, transfer_critic=False)
    material = deepcopy(cfg["contact_material"])
    for key in ("ground_static_friction_range", "ground_dynamic_friction_range"):
        material[key] = [.5, .5]
    nominal_material = deepcopy(material)
    changes["contact_material"] = material
    study_path = getattr(args, "randomization_study", None)
    study = None
    if study_path is not None:
        from wheeled_tasks.chassis.randomization_study import resolve, apply
        study_path = Path(study_path).resolve()
        study = resolve(json.loads(study_path.read_text()), getattr(args, "randomization_profile", None),
                        getattr(args, "randomization_strength", 1.))
        study.update(input_sha256=digest(study_path), input_file=str(study_path))
        apply(changes, task, study)
        write_json(args.output / "randomization_study.resolved.json", study)
        (args.output / "randomization_study.input.json").write_bytes(study_path.read_bytes())
    upright = None
    if getattr(args, "upright_task", None) is not None:
        from wheeled_tasks.chassis.upright_locomotion import resolve
        spec_path = args.upright_task.resolve()
        upright = resolve(json.loads(spec_path.read_text()), args.upright_mode, args.upright_friction_profile)
        upright.update(input_sha256=digest(spec_path), input_file=str(spec_path))
        friction = upright["friction_profiles"][upright["friction_profile"]]
        for name in ("wheel_static_friction", "wheel_dynamic_friction"):
            material[name] = friction["wheel"]
        if study is None:
            for name in ("ground_static_friction_range", "ground_dynamic_friction_range"):
                material[name] = [friction["ground"], friction["ground"]]
        ground_range = material["ground_static_friction_range"]
        upright["effective_wheel_ground_friction_range"] = [(v + friction["wheel"])/2. for v in ground_range]
        if study is not None:
            study["effective_wheel_ground_friction_range"] = deepcopy(upright["effective_wheel_ground_friction_range"])
            study["upright_material_override"] = dict(wheel_friction=friction["wheel"],
                original_ground_range_preserved=True, provenance=upright["friction_provenance"])
            task["randomization_study"] = deepcopy(study)
            write_json(args.output / "randomization_study.resolved.json", study)
        task.update(version="upright-locomotion-v1", upright_locomotion=upright,
            reward_potential_gain=0., recovery_leg_speed_cost=0.,
            recovery_family_weights=[1., 0., 0., 0., 0., 0., 0.],
            motion_push_fraction=0., hold_motion_during_recovery=False,
            episode_seconds=upright["episode_seconds"], curriculum_updates=1)
        changes.update(episode_seconds=upright["episode_seconds"], record_diagnostics=True,
            curriculum="upright_" + upright["mode"], contact_material=material)
        (args.output / "upright_task.input.json").write_bytes(spec_path.read_bytes())
    prone = None
    if getattr(args, "standard_prone_task", None) is not None:
        spec_path = args.standard_prone_task.resolve()
        prone = json.loads(spec_path.read_text())
        if prone.get("schema") != "wheel-leg-standard-prone-v1" or prone["fold_target_deg"] != 40.:
            raise ValueError("Need the explicit physical-knee40deg standard prone task")
        prone.update(input_sha256=digest(spec_path), input_file=str(spec_path),
                     friction_profile=args.upright_friction_profile)
        friction = prone["friction_profiles"][args.upright_friction_profile]
        for name in ("wheel_static_friction", "wheel_dynamic_friction"):
            material[name] = friction["wheel"]
        if study is None:
            for name in ("ground_static_friction_range", "ground_dynamic_friction_range"):
                material[name] = [friction["ground"], friction["ground"]]
        prone["effective_wheel_ground_friction_range"] = [
            (v + friction["wheel"])/2. for v in material["ground_static_friction_range"]]
        if study is not None:
            study["effective_wheel_ground_friction_range"] = deepcopy(prone["effective_wheel_ground_friction_range"])
            study["standard_prone_material_override"] = dict(wheel_friction=friction["wheel"],
                original_ground_range_preserved=True, provenance=prone["friction_provenance"])
            task["randomization_study"] = deepcopy(study)
            write_json(args.output / "randomization_study.resolved.json", study)
        task.update(version="standard-prone-in-place-v1", standard_prone=prone,
            reward_potential_gain=3., reward_gamma=.9974905699336811, recovery_leg_speed_cost=0.,
            motion_fraction=0., motion_push_fraction=0., hold_motion_during_recovery=True,
            stable_seconds=prone["stable_seconds"], episode_seconds=prone["episode_seconds"], curriculum_updates=1)
        changes.update(episode_seconds=prone["episode_seconds"], record_diagnostics=True,
            curriculum="standard_prone_in_place", contact_material=material)
        (args.output / "standard_prone_task.input.json").write_bytes(spec_path.read_bytes())
    static_prior = None
    if args.static_load_initialization:
        from wheeled_algo.standing_load_initialization import load_prior
        reference = ROOT / "reports/robust_policy_repair_20261024/scratch_nominal_static_load001.json"
        native_pair = ROOT / "reports/robust_policy_repair_20261024/scratch_static_load_native_pair001.json"
        # The offline holding-torque prior is authenticated only at the nominal
        # mechanism. It initializes a trainable bias, not compensation for the
        # sampled domains. A changed friction range must not silently be claimed
        # as part of the old native validation.
        nominal_mechanism = dict(cfg, **changes)
        nominal_mechanism["contact_material"] = nominal_material
        static_prior = load_prior(reference, native_pair, mechanism=nominal_mechanism, bank_path=args.bank, root=ROOT)
        if study is not None or upright is not None:
            static_prior["domain_scope"] = dict(nominal_contact_material=nominal_material,
                training_contact_material=deepcopy(material),
                randomized_domain_native_validation=False,
                initialization_only="nominal load prior; all six outputs remain trainable")
        (args.output / "static_load_reference.json").write_bytes(reference.read_bytes())
        (args.output / "static_load_native_pair.json").write_bytes(native_pair.read_bytes())
    row = design["stages"]["speed"]["roles"]["train"]
    for key, value in changes.items():
        cfg[key] = deepcopy(value)
        if key in row["contract"]:
            row["contract"][key] = deepcopy(value)
    validate_isolated_origins(cfg, args.num_envs)
    row["runtime"]["seed"] = args.seed
    row["runtime"]["contact_material"] = deepcopy(material)
    row["runtime"]["modules"]["dynamics_randomization"] = changes["dynamics_randomization"]["enabled"]
    row["invocation"] = dict(num_envs=args.num_envs, seed=dict(rule="fixed", value=args.seed))
    design["scope"] = ("standard_prone_mechanism_construction" if prone is not None else
                       "upright_locomotion_mechanism_construction" if upright is not None else
                       "scratch_standing_mechanism_construction")
    design["review_basis"] = dict(mechanism_config=str(BASE_CONFIG), sha256=digest(BASE_CONFIG),
        checkpoint_initialization="forbidden", task_contract=str(args.output / "task_contract.json"))
    write_json(args.output / "config/design.json", design)
    cfg["design_preflight"] = dict(file=str(args.output / "config/design.json"),
        sha256=digest(args.output / "config/design.json"), stage="speed")
    write_json(args.output / "config/contract.json", cfg)
    write_json(args.output / "task_contract.json", dict(schema="scratch-standing-task-v1", settings=task,
        training_origin="seeded random actor and critic; empty Adam; no checkpoint reads",
        bank=str(args.bank), bank_sha256=digest(args.bank),
        mechanism_config_sha256=digest(args.output / "config/contract.json"),
        policy_hz=200, physics_hz=1000, action_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        wheel_reduction=deepcopy(cfg["wheel_reduction"]),
        leg_position_scale_rad=.25, wheel_velocity_scale_rad_s=10.,
        state_writes="reset only", initial_pitch_range_deg=[-.5, .5],
        curriculum=dict(mid_height_until_update=args.all_heights_after, then_heights_m=[.23, .305, .42]),
        current_scope=("explicit domain study standing foundation; robustness and recovery are not qualified"
                       if study is not None else "nominal standing foundation; robustness and recovery are not qualified"),
        randomization_study=study,
        precise_standing_objective=args.precise_objective, normalized_observations=args.normalize_observations,
        desynchronized_initial_episode_lengths=args.desync_episodes,
        standing_failure_resets=args.standing_failures,
        wheel_action_parameterization=("encoder_velocity_plus_learned_increment_v1"
            if args.wheel_velocity_increment else "learned_absolute_velocity_v1"),
        static_load_initialization=static_prior,
        standing_contact_continuity_contract=continuity_contract,
        standing_leg_target_rate_contract=target_rate_contract, hardware_ready=False))
    if getattr(args, "separate_leg_velocity_feedback", False):
        contract_path = args.output / "task_contract.json"
        value = json.loads(contract_path.read_text())
        value["leg_velocity_feedback_architecture"] = dict(schema="separated-neural-leg-velocity-v1",
            neutral_raw_features=[16, 17, 18, 19], leg_policy_outputs=[0, 1, 2, 3],
            native_full_feedback_outputs=[4, 5], physical_PD_velocity_feedback_hz=1000,
            final_Gaussian_mean_before_sampling=True, action_filter=False, state_writes="reset only",
            scope="Fresh trainable six-output actor; no checkpoint, teacher or running target hold.")
        write_json(contract_path, value)
    if upright is not None:
        contract_path = args.output / "task_contract.json"
        value = json.loads(contract_path.read_text())
        value.update(schema="upright-locomotion-task-v1", upright_locomotion=upright,
            recovery_training_enabled=False, automatic_task_or_height_curriculum=False,
            curriculum=dict(mode=upright["mode"], reset_height_m=.305, automatic_changes=False),
            current_scope="305mm upright " + upright["mode"] + "; recovery and full capability qualification excluded",
            terminal_evaluation="30s cold upright cases; all failures and raw contact samples retained")
        write_json(contract_path, value)
    if prone is not None:
        contract_path = args.output / "task_contract.json"
        value = json.loads(contract_path.read_text())
        value.update(schema="standard-prone-task-v1", standard_prone=prone,
            recovery_training_enabled=True, automatic_task_or_height_curriculum=False,
            curriculum=dict(reset_pose=prone["reset_pose_name"], target_height_m=.305, automatic_changes=False),
            initial_pitch_range_deg=[90., 90.], current_scope="One standard prone family; guided in-place six-motor RL",
            terminal_evaluation="Actual recovery completion, then30s standing, with all failures and raw1ms samples retained")
        write_json(contract_path, value)
    if upright is not None or prone is not None:
        contract_path = args.output / "task_contract.json"
        value = json.loads(contract_path.read_text())
        value["final_objective"] = {
            name: dict(file=relative, sha256=digest(ROOT / relative)) for name, relative in (
                ("acceptance", "contracts/wheel_leg_acceptance_requirements_v2.json"),
                ("curriculum", "contracts/wheel_leg_separated_curriculum_v1.json"),
                ("randomization", "contracts/wheel_leg_randomization_study_v2.json"))}
        for item in value["final_objective"].values():
            target = args.output / "final_objectives" / Path(item["file"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / item["file"]).read_bytes())
        write_json(contract_path, value)
    return cfg, task


def sources():
    files = list((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    for name in ("train_upright_locomotion.py", "evaluate_upright_locomotion.py",
                 "review_standing_acceptance.py", "train_standard_prone.py", "evaluate_standard_prone.py"):
        path = ROOT / "scripts" / name
        if path.is_file():
            files.append(path)
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def archive_committed_sources(source_hashes, output, *, root=ROOT):
    """Fail before Isaac startup unless every frozen source matches one commit."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for name, expected in source_hashes.items():
        result = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True)
        if (result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected
                or digest(root / name) != expected):
            raise ValueError("Training requires committed source matching the snapshot: " + name)
    archive = subprocess.check_output(["git", "archive", "--format=tar", revision, "--", *sorted(source_hashes)], cwd=root)
    path = output / "committed_source.tar"
    path.write_bytes(archive)
    receipt = dict(schema="scratch-committed-source-v1", commit=revision, source_count=len(source_hashes),
                   all_source_bytes_match_commit=True, archive=str(path), archive_sha256=digest(path))
    write_json(output / "source_revision.json", receipt)
    return receipt


def nominal_action(env, height=.305):
    from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL
    index = getattr(env, "initialization_pose_index", env.height_pose[height])
    pose = env.bank_joints[index, env.ids]
    delta = pose - env.v5.nominal
    result = delta.clone().zero_()
    result[list(env.v5.LEGS)] = torch.atan2(delta[list(env.v5.LEGS)].sin(), delta[list(env.v5.LEGS)].cos()) / .25
    return result[list(POLICY_FROM_CONTROL)]


def evaluate(env, actor, output, writer, completed, *, fixed_target=False, heights=(.23, .305, .42)):
    import numpy as np
    count = 7 * len(heights)
    pitches = (0., -.25, .25, -.5, .5, -1., 1.)
    requested = [dict(height=height, pitch_deg=pitch) for height in heights for pitch in pitches]
    if env.num_envs < count:
        raise ValueError("Evaluation requires at least 21 native environments")
    cases = [dict(pose_index=env.height_pose[requested[i % count]["height"]], recovery=False,
                  command=[0., 0., requested[i % count]["height"]], pitch_deg=requested[i % count]["pitch_deg"])
             for i in range(env.num_envs)]
    env.evaluation_cases = cases
    env.cfg["record_diagnostics"] = True
    env.reset(torch.arange(env.num_envs, device=env.device))
    env.fixed_cases = cases
    initial_root = env.robot.data.root_link_pose_w.torch[:count, :3].clone()
    horizon = round(8. / env.policy_dt)
    traces = {key: [] for key in ("force", "clearance", "dq", "torque", "q", "valid", "position", "gravity", "velocity", "gap")}
    observation = env.get_observations()
    with torch.inference_mode():
        constant = torch.stack([nominal_action(env, case["command"][2]) for case in cases])
        for _ in range(horizon):
            valid = ~env.finished[:count].clone()
            action = constant if fixed_target else actor(observation)
            observation, _, _, info = env.step(action)
            contacts = env.standing_contact_monitor.last
            physical = env.physical_standing_last
            diagnostic = info["diagnostics"]
            values = dict(force=contacts["force"][:, :count], clearance=contacts["clearance"][:, :count],
                dq=physical["dq"][:, :count], torque=physical["torque"][:, :count],
                q=diagnostic["motor_position"][:count, list(env.v5.LEGS)], valid=valid,
                position=diagnostic["local_position"][:count], gravity=diagnostic["gravity"][:count],
                velocity=diagnostic["velocity"][:count], gap=diagnostic["gap"][:count])
            for key, value in values.items():
                traces[key].append(value.cpu())
    arrays = {key: torch.stack(values).numpy() for key, values in traces.items()}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "native_trace.npz", **arrays)
    rows = []
    for i, case in enumerate(requested):
        valid = arrays["valid"][:, i]
        force = arrays["force"][:, :, i][valid]
        contact = (np.linalg.norm(force, axis=-1) > 2.).all(-1)
        speed = arrays["dq"][:, :, i][valid]
        q = arrays["q"][:, i][valid]
        late = q[600:] if len(q) > 600 else q
        frequency = np.fft.rfftfreq(len(late), env.policy_dt)
        spectrum = np.fft.rfft(late - late.mean(0), axis=0)
        spectrum[(frequency < 3.) | (frequency > 95.)] = 0.
        highpass = np.fft.irfft(spectrum, n=len(late), axis=0)
        position = arrays["position"][:, i][valid]
        tilt = np.arccos(np.clip(-arrays["gravity"][:, i, 2][valid], -1., 1.))
        velocity = arrays["velocity"][:, i][valid]
        gap = arrays["gap"][:, i][valid]
        row = dict(height_m=case["height"], pitch_deg=case["pitch_deg"],
            full_horizon=len(q) == horizon, physical_ms=int(contact.size),
            missing_physical_ms=int((~contact).sum()), dual_support_fraction=float(contact.mean()),
            first20ms_continuous=bool(contact.reshape(-1)[:20].all()),
            maximum_wheel_clearance_m=float(arrays["clearance"][:, :, i][valid].max()),
            leg_speed_peak_rad_s=float(np.abs(speed).max()), leg_speed_rms_rad_s=float(np.sqrt(np.mean(speed**2))),
            leg_vibration_3_95hz_rms_mrad=float(np.sqrt(np.mean(highpass**2))*1000.),
            height_mae_m=float(np.abs(position[:, 2] - case["height"]).mean()),
            tilt_peak_deg=float(np.degrees(tilt).max()), closure_gap_peak_m=float(gap.max()),
            forward_speed_rms_m_s=float(np.sqrt(np.mean(velocity[:, 0]**2))))
        row["continuous_support_pass"] = bool(row["full_horizon"] and row["missing_physical_ms"] == 0)
        row["standing_pass"] = bool(row["continuous_support_pass"] and row["tilt_peak_deg"] < 10.
            and row["height_mae_m"] < .01 and row["forward_speed_rms_m_s"] < .03
            and row["leg_speed_peak_rad_s"] < 3. and row["closure_gap_peak_m"] < .003)
        rows.append(row)
    result = dict(schema="scratch-standing-native-evaluation-v1", completed_updates=completed,
        protocol="nominal dynamics; seven held-out pitch starts per height; no startup grace; every 1 ms",
        fixed_target_probe=fixed_target, learned_policy=not fixed_target,
        checkpoint_loaded=False, initial_root_positions=initial_root.cpu().tolist(),
        case_count=count, continuous_support_pass=sum(row["continuous_support_pass"] for row in rows),
        standing_pass=sum(row["standing_pass"] for row in rows), cases=rows,
        trace_sha256=digest(output / "native_trace.npz"), hardware_ready=False)
    write_json(output / "result.json", result)
    for height in heights:
        selected = [row for row in rows if row["height_m"] == height]
        for metric in ("dual_support_fraction", "missing_physical_ms", "leg_speed_peak_rad_s",
                       "leg_vibration_3_95hz_rms_mrad", "height_mae_m", "forward_speed_rms_m_s"):
            writer.add_scalar(f"EvalScratchStanding/{height:.3f}m/{metric}",
                              sum(row[metric] for row in selected)/len(selected), completed)
    writer.add_scalar("EvalScratchStanding/continuous_support_pass", result["continuous_support_pass"], completed)
    writer.flush()
    return result


class TrainingStopped(Exception):
    pass


def main(*, upright_entry=False, prone_entry=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json")
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=261001)
    parser.add_argument("--all-heights-after", type=int, default=512)
    parser.add_argument("--randomization-study", type=Path,
                        help="Opt-in JSON with independent domain groups; new actor/critic/empty Adam only")
    parser.add_argument("--randomization-profile",
                        help="Named profile inside the explicit study; no legacy severity overlay")
    parser.add_argument("--randomization-strength", type=float, default=1.,
                        help="Scale the profile's group radii once within0..1; timing remains explicit")
    parser.add_argument("--speed-weight", type=float, default=2.)
    parser.add_argument("--contact-weight", type=float, default=8.)
    parser.add_argument("--contact-break-cost", type=float,
                        help="Training reward cost per actual1ms interruption of dual-wheel support; fixed evaluation unchanged")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--leg-target-rate-cost", type=float,
                        help="Training density cost on applied four-leg target deltas; no action filtering")
    parser.add_argument("--precise-objective", action="store_true")
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--desync-episodes", action="store_true")
    parser.add_argument("--standing-failures", action="store_true")
    parser.add_argument("--wheel-velocity-increment", action="store_true",
                        help="Fresh actor uses raw wheel encoder velocity plus a learned correction; final action ABI stays absolute")
    parser.add_argument("--separate-leg-velocity-feedback", action="store_true",
                        help="Fresh actor separates neural leg-target means from raw leg-speed inputs; wheels and1kHz PD retain velocity feedback")
    parser.add_argument("--static-load-initialization", action="store_true",
                        help="Initialize fresh trainable leg bias from the authenticated nominal static load, never from a policy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-target-probe", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--upright-task", type=Path,
        default=ROOT / "contracts/wheel_leg_upright_locomotion_v1.json" if upright_entry else None)
    parser.add_argument("--standard-prone-task", type=Path,
        default=ROOT / "contracts/wheel_leg_standard_prone_v1.json" if prone_entry else None)
    parser.add_argument("--upright-mode", choices=("balance", "basic_motion"), default="balance")
    parser.add_argument("--upright-friction-profile", choices=("baseline", "legacy_low_grip"), default="baseline")
    parser.add_argument("--native-probe-steps", type=int, default=0,
        help="Zero-PPO diagnostic of upright mechanics and task wiring; requires --updates 0")
    if upright_entry or prone_entry:
        parser.set_defaults(normalize_observations=True, wheel_velocity_increment=True,
                            static_load_initialization=not prone_entry)
    if prone_entry:
        parser.set_defaults(rollout_steps=256)
    args = parser.parse_args()
    if args.upright_task is not None and args.standard_prone_task is not None:
        parser.error("Upright balance and standard-prone recovery are separate training tasks")
    if args.standard_prone_task is not None and args.static_load_initialization:
        parser.error("Standard-prone initialization uses its own reset geometry, not a standing-load bias")
    if args.standard_prone_task is not None and args.rollout_steps < 256:
        parser.error("Standard-prone first rollout needs256 steps to include control after native passive settling")
    if args.upright_task is not None or args.standard_prone_task is not None:
        if (args.precise_objective or args.standing_failures
                or args.contact_break_cost is not None or args.leg_target_rate_cost is not None
                or args.fixed_target_probe):
            parser.error("Upright task owns its rewards, failures and contact review")
        if args.upright_mode == "basic_motion" and args.num_envs < 35:
            parser.error("Basic-motion cold evaluation needs at least35 environments")
    if args.native_probe_steps < 0 or args.native_probe_steps > 2000:
        parser.error("Native wiring probe must contain 1..2000 steps")
    if args.native_probe_steps and ((args.upright_task is None and args.standard_prone_task is None) or args.updates):
        parser.error("Native wiring probe requires an explicit task and --updates 0")
    if args.randomization_study is None and (args.randomization_profile is not None or args.randomization_strength != 1.):
        parser.error("Profile and strength require an explicit randomization study JSON")
    if args.updates < 0 or args.num_envs < 21 or args.num_envs > 256 or args.rollout_steps < 8:
        parser.error("Need nonnegative updates,21..256 isolated environments and rollout>=8")
    if args.fixed_target_probe and args.updates:
        parser.error("A fixed-target physics probe must have zero PPO updates")
    if args.standing_failures and not args.precise_objective:
        parser.error("Standing failure resets require the precise standing objective")
    if args.separate_leg_velocity_feedback and not args.wheel_velocity_increment:
        parser.error("Separated leg feedback requires the wheel-increment actor ABI")
    if args.contact_break_cost is not None:
        if not args.precise_objective:
            parser.error("Contact event cost requires the precise standing objective")
        from wheeled_tasks.chassis.standing_contact_continuity import contract
        try:
            contract(args.contact_break_cost)
        except ValueError as error:
            parser.error(str(error))
    if args.leg_target_rate_cost is not None:
        if not args.precise_objective:
            parser.error("Leg-target variation cost requires the precise standing objective")
        from wheeled_tasks.chassis.standing_leg_target_rate import contract as target_contract
        try:
            target_contract(args.leg_target_rate_cost)
        except ValueError as error:
            parser.error(str(error))
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, task = prepare(args)
    source_hashes = sources()
    committed_source = (None if args.preflight_only or args.native_probe_steps
                        else archive_committed_sources(source_hashes, args.output))
    for name, expected in source_hashes.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Source changed while taking the scratch snapshot")
        target = args.output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (args.output / "pose_bank.json").write_bytes(args.bank.read_bytes())
    write_json(args.output / "source_hashes.json", source_hashes)
    if args.preflight_only:
        print(json.dumps(dict(status="prepared", checkpoint_reads=0, output=str(args.output))))
        return
    global torch
    launcher = env = runner = audit = None
    start = time.monotonic()
    report = dict(schema="scratch-standing-progress-v1", status="starting",
        training_entry=("standard-prone-v1" if args.standard_prone_task is not None else
                        "upright-locomotion-v1" if args.upright_task is not None else "scratch-standing"),
        pid=os.getpid(), start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        started_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        completed_updates=0, optimizer_attempts=0, checkpoint_reads=[],
        committed_source=committed_source,
        requested_updates=args.updates, hardware_ready=False, args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write_json(args.output / "progress.json", report)
    original_load = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        report["requested_stop_signal"] = signum

    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.chassis.scratch_standing_env import ScratchStandingEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install
        from wheeled_algo.scratch_standing import agent_config, initialize_fresh, tensor_hash
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.native_ppo_audit import NativePPOAudit

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        original_load = torch.load

        def forbidden_load(*positional, **keywords):
            report["checkpoint_reads"].append(str(positional[0] if positional else keywords.get("f")))
            raise RuntimeError("Scratch training forbids every torch.load call")

        torch.load = forbidden_load
        manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
        environment_class = ScratchStandingEnv
        if args.upright_task is not None:
            from wheeled_tasks.chassis.upright_locomotion_env import UprightLocomotionEnv
            environment_class = UprightLocomotionEnv
        if args.standard_prone_task is not None:
            from wheeled_tasks.chassis.standard_prone_env import StandardProneEnv
            environment_class = StandardProneEnv
        env = environment_class(cfg, manifest, load_chassis_control(ROOT / cfg["control_math_source"]), ROOT,
            stage_name="speed", num_envs=args.num_envs, device=args.device, seed=args.seed, level=1.,
            run_role="train", bank=args.bank, settings=task, all_heights_after=args.all_heights_after)
        if args.upright_task is not None or args.standard_prone_task is not None:
            # Retain raw 1ms observers; this task owns every reward term.
            report["physical_reward"] = install(env, speed_weight=0., contact_weight=0., torque_weight=0.)
        else:
            report["physical_reward"] = install(env, speed_weight=args.speed_weight, contact_weight=args.contact_weight)
        if args.native_probe_steps:
            if args.standard_prone_task is not None:
                from evaluate_standard_prone import native_probe
            else:
                from evaluate_upright_locomotion import native_probe
            report["native_probe"] = native_probe(env, args.native_probe_steps, args.output)
            report.update(status="native_probe_completed_unqualified", elapsed_s=time.monotonic()-start,
                          new_PPO=0, learned_policy=False)
            return
        if args.precise_objective:
            from wheeled_tasks.chassis.precise_standing_reward import install as install_precision
            report["precise_reward"] = install_precision(env)
        if args.standing_failures:
            from wheeled_tasks.chassis.standing_failure_termination import install as install_failures
            report["standing_failure_resets"] = install_failures(env)
        if args.contact_break_cost is not None:
            from wheeled_tasks.chassis.standing_contact_continuity import install as install_contact_events
            report["standing_contact_continuity_contract"] = install_contact_events(env, args.contact_break_cost)
            expected = json.loads((args.output / "task_contract.json").read_text())["standing_contact_continuity_contract"]
            if report["standing_contact_continuity_contract"] != expected:
                raise ValueError("Actual contact-event objective differs from the frozen training contract")
        if args.leg_target_rate_cost is not None:
            from wheeled_tasks.chassis.standing_leg_target_rate import install as install_target_rate
            report["standing_leg_target_rate_contract"] = install_target_rate(env, args.leg_target_rate_cost)
            expected = json.loads((args.output / "task_contract.json").read_text())["standing_leg_target_rate_contract"]
            if report["standing_leg_target_rate_contract"] != expected:
                raise ValueError("Actual leg-target cost differs from the frozen training contract")
        runner_cfg = agent_config(args.seed, args.device, args.rollout_steps, args.learning_rate,
            normalize_observations=args.normalize_observations,
            wheel_velocity_increment=args.wheel_velocity_increment,
            separate_leg_velocity_feedback=args.separate_leg_velocity_feedback)
        write_json(args.output / "agent_config.json", runner_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(args.output), device=args.device)
        initial = nominal_action(env)
        bias_source = ("authenticated standard-prone reset joint geometry; all six outputs trainable"
                       if args.standard_prone_task is not None else None)
        if args.static_load_initialization:
            from wheeled_algo.standing_load_initialization import initial_action
            prior = json.loads((args.output / "task_contract.json").read_text())["static_load_initialization"]
            initial = initial_action(initial, prior)
            bias_source = prior["source"]
        report["initialization"] = initialize_fresh(runner.alg, initial, output_bias_source=bias_source)
        if args.static_load_initialization:
            report["initialization"]["static_load_initialization"] = prior
        report["initialization"]["actor_class"] = runner_cfg["actor"]["class_name"]
        report["initialization"]["wheel_velocity_increment"] = args.wheel_velocity_increment
        report["initialization"]["separate_leg_velocity_feedback"] = args.separate_leg_velocity_feedback
        write_json(args.output / "initialization.json", report["initialization"])
        if task.get("randomization_study") is not None:
            from wheeled_tasks.chassis.randomization_study import runtime_samples
            write_json(args.output / "randomization_study.initial_samples.json", runtime_samples(env))
        initial_actor = {key: value.detach().cpu().clone() for key, value in runner.alg.actor.state_dict().items()}
        initial_critic = {key: value.detach().cpu().clone() for key, value in runner.alg.critic.state_dict().items()}
        mask = RecoveryRolloutMask(runner.alg, num_envs=args.num_envs)
        native_update = runner.alg.update

        def save(path, infos=None):
            path = Path(path)
            state = runner.alg.save()
            state.update(iter=report["completed_updates"], infos=dict(schema="scratch-standing-checkpoint-v1",
                initialization=report["initialization"], completed_updates=report["completed_updates"],
                training_origin="scratch", continuation_forbidden=True,
                committed_source=committed_source,
                settings=task, source_hashes=source_hashes, bank_sha256=digest(args.bank),
                actor_class=runner_cfg["actor"]["class_name"],
                agent_config_sha256=digest(args.output / "agent_config.json"),
                standing_contact_continuity_contract=report.get("standing_contact_continuity_contract"),
                standing_leg_target_rate_contract=report.get("standing_leg_target_rate_contract"),
                mechanism_config_sha256=digest(args.output / "config/contract.json"),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                env_rng=env.generator.get_state(), domain_rng=env.domain.rng.get_state()))
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            report["last_checkpoint"] = dict(path=str(path), sha256=digest(path), updates=report["completed_updates"])

        def update():
            nonlocal audit
            report["optimizer_attempts"] += 1
            report["optimizer_update_in_flight"] = True
            write_json(args.output / "progress.json", report)
            losses = native_update()
            report["completed_updates"] += 1
            report["optimizer_update_in_flight"] = False
            for model in (runner.alg.actor, runner.alg.critic):
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise RuntimeError("Nonfinite fresh PPO parameter")
            if report["completed_updates"] == 1:
                if task.get("randomization_study") is not None:
                    from wheeled_tasks.chassis.randomization_study import scalar_metrics
                    for key, value in scalar_metrics(task["randomization_study"]).items():
                        runner.logger.writer.add_scalar("DomainStudyConfig/" + key, value, 0)
                    runner.logger.writer.flush()
                audit.close()
                audit_result = audit.report()
                audit_result.update(actor_changed_tensors=sum(not torch.equal(initial_actor[key], value.cpu()) for key, value in runner.alg.actor.state_dict().items()),
                    critic_changed_tensors=sum(not torch.equal(initial_critic[key], value.cpu()) for key, value in runner.alg.critic.state_dict().items()),
                    actor_changed_parameter_tensors=sum(not torch.equal(initial_actor[key], value.detach().cpu()) for key, value in runner.alg.actor.named_parameters()),
                    critic_changed_parameter_tensors=sum(not torch.equal(initial_critic[key], value.detach().cpu()) for key, value in runner.alg.critic.named_parameters()),
                    actor_sha256=tensor_hash(runner.alg.actor.state_dict()), critic_sha256=tensor_hash(runner.alg.critic.state_dict()))
                if (not audit_result["all_recorded_values_finite"] or audit_result["committed_optimizer_calls"] != 12
                        or not audit_result["actor_changed_parameter_tensors"] or not audit_result["critic_changed_parameter_tensors"]):
                    raise RuntimeError("First fresh native PPO update failed its audit")
                write_json(args.output / "first_update_audit.json", audit_result)
                audit = None
            env.actor_updates = report["completed_updates"]
            report.update(status="training", elapsed_s=time.monotonic()-start,
                last_losses={key: float(value) for key, value in losses.items()}, valid_rollout=mask.metrics,
                last_physical_metrics={key: float(value) for key, value in env.physical_standing_last["metrics"].items()})
            write_json(args.output / "progress.json", report)
            if stop_requested:
                save(args.output / "model_stopped.pt")
                raise TrainingStopped()
            return losses

        runner.save, runner.alg.update = save, update
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        save(args.output / "model_initial.pt")
        if args.updates:
            audit = NativePPOAudit(runner.alg)
            runner.learn(args.updates, init_at_random_ep_len=args.desync_episodes)
        else:
            runner.logger.init_logging_writer()
        if task.get("randomization_study") is not None:
            from wheeled_tasks.chassis.randomization_study import scalar_metrics
            for key, value in scalar_metrics(task["randomization_study"]).items():
                runner.logger.writer.add_scalar("DomainStudyConfig/" + key, value, report["completed_updates"])
        save(args.output / "model_final.pt")
        report["status"] = "evaluating"
        write_json(args.output / "progress.json", report)
        runner.alg.eval_mode()
        if args.standard_prone_task is not None:
            from evaluate_standard_prone import evaluate as evaluate_prone
            report["evaluation"] = evaluate_prone(env, runner.alg.actor, args.output / "evaluation",
                runner.logger.writer, report["completed_updates"])
        elif args.upright_task is not None:
            from evaluate_upright_locomotion import evaluate as evaluate_upright
            report["evaluation"] = evaluate_upright(env, runner.alg.actor, args.output / "evaluation",
                runner.logger.writer, report["completed_updates"])
        else:
            report["evaluation"] = evaluate(env, runner.alg.actor, args.output / "evaluation", runner.logger.writer,
                                             report["completed_updates"], fixed_target=args.fixed_target_probe)
        report.update(status="completed_unqualified", elapsed_s=time.monotonic()-start)
    except TrainingStopped:
        report.update(status="stopped", elapsed_s=time.monotonic()-start)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(), elapsed_s=time.monotonic()-start)
        raise
    finally:
        write_json(args.output / "progress.json", report)
        if original_load is not None:
            torch.load = original_load
        if audit is not None:
            audit.close()
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
