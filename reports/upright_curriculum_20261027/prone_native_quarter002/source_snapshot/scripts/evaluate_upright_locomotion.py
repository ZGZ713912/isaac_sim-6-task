"""Cold upright engineering review; not a full capability qualification."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from review_standing_acceptance import support_review
from train_scratch_standing import digest, nominal_action, write_json
from wheeled_tasks.chassis.upright_locomotion import training_case


def native_probe(env, steps, output):
    """A constant mechanics target tests wiring, never a learned policy."""
    action = nominal_action(env).expand(env.num_envs, -1).clone()
    checks = dict(native_rates=env.dt == .001 and env.policy_dt == .005,
                  no_passive_births=not bool(env.passive.any()),
                  no_recovery_births=not bool(env.recovery_lane.any()))
    observations, rows = [], []
    for _ in range(steps):
        observation, reward, done, extra = env.step(action)
        state = extra["diagnostics"]
        terms = state["upright_reward_terms"]
        expected = sum(terms.values()) * env.policy_dt - env.upright_task["termination"]["failure_cost"] * state["terminated"]
        checks["independent_reward"] = checks.get("independent_reward", True) and bool(torch.allclose(reward, expected))
        checks["no_recovery_progress_or_phase"] = checks.get("no_recovery_progress_or_phase", True) and (
            not bool(state["reward_potential_progress"].any()) and not bool(state["reward_recovery_active"].any()))
        checks["every_physical_contact_sample"] = checks.get("every_physical_contact_sample", True) and (
            env.standing_contact_monitor.last["force"].shape == (5, env.num_envs, 2, 3))
        checks["finite_observations"] = checks.get("finite_observations", True) and all(
            bool(torch.isfinite(value).all()) for value in observation.values())
        checks["terminal_transitions_do_not_bootstrap"] = checks.get("terminal_transitions_do_not_bootstrap", True) and not bool(
            (extra["policy_bootstrap_mask"] & state["terminated"]).any())
        observations.append(state["gravity"].cpu().numpy())
        rows.append(dict(terminated=int(state["terminated"].sum()),
            tilt_events=int(state["upright_failures"]["tilt"].sum()),
            height_events=int(state["upright_failures"]["height"].sum()),
            nonwheel_contact_events=int(state["upright_failures"]["nonwheel_contact"].sum()),
            nonwheel_force_peak_n=float(state["nonwheel_force_peak_n"].max())))
    material = env.contact_material.summary()
    declared = env.upright_task["effective_wheel_ground_friction_range"]
    actual = material["effective_wheel_static_min_max"]
    checks["effective_friction_within_declared_range"] = declared[0] - 1e-6 <= actual[0] <= actual[1] <= declared[1] + 1e-6
    np.savez_compressed(Path(output) / "native_probe_trace.npz", gravity=np.stack(observations))
    result = dict(schema="upright-native-wiring-probe-v1", new_PPO=0, learned_policy=False,
        control="constant reset geometry targets; no actor or optimizer constructed",
        all_wiring_checks_pass=all(checks.values()), checks=checks,
        physics_steps=steps * env.decimation, num_envs=env.num_envs,
        contact_material=material, actual_task=env.upright_task, transitions=rows,
        behavior_qualified=False, hardware_ready=False)
    write_json(Path(output) / "native_startup.json", env.startup_report)
    write_json(Path(output) / "native_probe.json", result)
    if not all(checks.values()):
        raise RuntimeError("Upright native wiring probe failed: " + str(checks))
    return result


def evaluate(env, actor, output, writer, completed):
    spec, count = env.upright_task, 7
    evaluation = spec["evaluation"]
    cases = []
    lanes = [0] if spec["mode"] == "balance" else [0, 4, 6, 8, 9]
    for lane in lanes:
        for pitch in evaluation["pitch_deg"]:
            case = training_case(spec, lane, env.height_pose[.305])
            case["pitch_deg"] = pitch
            cases.append(case)
    count = len(cases)
    if count > env.num_envs:
        raise ValueError("Upright cold evaluation needs at least " + str(count) + " environments")
    env.evaluation_cases = [deepcopy(cases[i % count]) for i in range(env.num_envs)]
    env.reset(torch.arange(env.num_envs, device=env.device))
    env.fixed_cases = env.evaluation_cases
    env.max_episode_length = horizon = round(evaluation["seconds"] / env.policy_dt)
    initial = env.robot.data.root_link_pose_w.torch[:count, :3].clone() - env.origins[:count]
    observation = env.get_observations()
    fields = ("position", "velocity", "omega", "gravity", "height", "commands", "time", "terminated", "valid",
              "force", "clearance", "dq", "torque", "gap", "nonwheel_force")
    traces = {key: [] for key in fields}
    with torch.inference_mode():
        for _ in range(horizon):
            valid = ~env.finished[:count].clone()
            observation, _, _, extra = env.step(actor(observation))
            state, contact = extra["diagnostics"], env.standing_contact_monitor.last
            physical = env.physical_standing_last
            values = dict(position=state["local_position"][:count], velocity=state["velocity"][:count],
                omega=state["omega"][:count], gravity=state["gravity"][:count], height=state["height"][:count],
                commands=state["commands"][:count], time=state["time"][:count],
                terminated=state["terminated"][:count], valid=valid,
                force=contact["force"][:, :count], clearance=contact["clearance"][:, :count],
                dq=physical["dq"][:, :count], torque=physical["torque"][:, :count], gap=state["gap"][:count],
                nonwheel_force=state["nonwheel_force_peak_n"][:count])
            for key, value in values.items():
                traces[key].append(value.cpu().numpy())
    arrays = {key: np.stack(value) for key, value in traces.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "native_trace.npz", **arrays, initial_root_local=initial.cpu().numpy())
    rows = []
    for lane, case in enumerate(cases):
        valid = arrays["valid"][:, lane]
        velocity, omega = arrays["velocity"][valid, lane], arrays["omega"][valid, lane]
        command = arrays["commands"][valid, lane]
        supported = (np.linalg.norm(arrays["force"][valid, :, lane], axis=-1) > 2.).all(-1).reshape(-1)
        support = support_review(supported, standing_start_sample=0,
            ignored_max_ms=evaluation["ignored_contiguous_support_loss_ms"])
        xy = arrays["position"][valid, lane, :2] - initial[lane, :2].cpu().numpy()
        active = np.abs(command[:, :2]).max(-1) > .01
        tracking = active if case["role"] != "balance" else np.ones(len(velocity), dtype=bool)
        row = dict(role=case["role"], pitch_deg=case["pitch_deg"],
            full_horizon=int(valid.sum()) == horizon, terminated=bool(arrays["terminated"][valid, lane].any()),
            tilt_peak_deg=float(np.degrees(np.arccos(np.clip(-arrays["gravity"][valid, lane, 2], -1., 1.))).max()),
            height_mae_m=float(np.abs(arrays["height"][valid, lane] - command[:, 2]).mean()),
            xy_excursion_max_m=float(np.linalg.norm(xy, axis=-1).max()),
            speed_rms_m_s=float(np.sqrt(np.mean(np.sum(velocity[:, :2]**2, axis=-1)))),
            vx_mae_m_s=(float(np.abs(velocity[tracking, 0] - command[tracking, 0]).mean()) if tracking.any() else None),
            yaw_mae_rad_s=(float(np.abs(omega[tracking, 2] - command[tracking, 1]).mean()) if tracking.any() else None),
            requested_motion_observed=bool(active.any()) if case["role"] != "balance" else True,
            closure_gap_peak_m=float(arrays["gap"][valid, lane].max()), support=support,
            raw_wheel_clearance_peak_m=float(arrays["clearance"][valid, :, lane].max()),
            leg_speed_peak_rad_s=float(np.abs(arrays["dq"][valid, :, lane]).max()))
        row["baseline_pass"] = bool(row["full_horizon"] and not row["terminated"] and support["support_pass"]
            and row["tilt_peak_deg"] <= evaluation["tilt_peak_deg"] and row["height_mae_m"] <= evaluation["height_mae_m"]
            and row["closure_gap_peak_m"] <= .003 and row["requested_motion_observed"])
        if case["role"] == "balance":
            row["baseline_pass"] &= (row["xy_excursion_max_m"] <= evaluation["standing_xy_max_m"]
                                     and row["speed_rms_m_s"] <= evaluation["standing_speed_rms_m_s"])
        else:
            row["baseline_pass"] &= bool(row["vx_mae_m_s"] is not None and row["yaw_mae_rad_s"] is not None
                and row["vx_mae_m_s"] <= evaluation["motion_vx_mae_m_s"]
                and row["yaw_mae_rad_s"] <= evaluation["motion_yaw_mae_rad_s"])
        rows.append(row)
    result = dict(schema="upright-cold-engineering-review-v1", completed_updates=completed,
        standing_start="explicit initialized-standing event at reset; no grace or moved start",
        case_count=count, baseline_pass=sum(row["baseline_pass"] for row in rows), cases=rows,
        source_scope=evaluation["scope"], full_capability_qualified=False, hardware_ready=False,
        trace_sha256=digest(output / "native_trace.npz"))
    write_json(output / "result.json", result)
    for lane, row in enumerate(rows):
        for key in ("baseline_pass", "tilt_peak_deg", "height_mae_m", "xy_excursion_max_m", "speed_rms_m_s"):
            writer.add_scalar(f"EvalUpright/case_{lane:02d}/{key}", row[key], completed)
    writer.add_scalar("EvalUpright/baseline_pass", result["baseline_pass"], completed)
    writer.flush()
    return result
