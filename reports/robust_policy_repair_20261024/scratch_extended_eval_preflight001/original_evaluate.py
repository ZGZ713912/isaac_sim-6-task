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
