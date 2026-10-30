#!/usr/bin/env python3
"""Capture a bounded fixed-case command path, then query actors on identical inputs.

Counterfactual commands are evaluated offline and are never sent to the simulator.
This is a zero-PPO diagnostic, with no acceptance grade or training-contract change.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--comparison-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1000)
    args = parser.parse_args()
    if not 201 <= args.ticks <= 1000:
        parser.error("diagnostic duration must be 201..1000 ticks at 200 Hz")
    root = args.source_root.resolve()
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from train_chassis import preflight
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    evaluation = json.loads((args.evaluation / "evaluation.json").read_text())
    assert evaluation["status"] == "evaluated" and evaluation["trace_recording"]
    candidate = evaluation["candidates"][args.candidate_index]
    checkpoint_path = Path(candidate["checkpoint"])
    contract, manifest = preflight(args.evaluation / "contract.json")
    assert digest(args.evaluation / "contract.json") == evaluation["contract_sha256"]
    assert digest(checkpoint_path) == candidate["checkpoint_sha256"]
    assert contract["actor_dim"] == 35 and contract["policy_dt"] == .005
    for name, expected in evaluation["source_sha256"].items():
        assert digest(root / name) == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="v6-local-command-path-diagnosis-v1", status="starting",
        started_at=datetime.now(timezone.utc).isoformat(), ppo_updates=0,
        acceptance_evaluation=False, counterfactual_commands_applied_to_simulator=False,
        source_root=str(root), script_sha256=digest(__file__),
        evaluation=str(args.evaluation), evaluation_sha256=digest(args.evaluation / "evaluation.json"),
        checkpoint=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
        comparison_checkpoint=str(args.comparison_checkpoint),
        comparison_checkpoint_sha256=digest(args.comparison_checkpoint), ticks=args.ticks,
        scope="Original nominal fixed cases, first five seconds, with extra measurement only; offline single-column command counterfactuals hold all other input entries constant.")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "contract.json").write_bytes((args.evaluation / "contract.json").read_bytes())
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.policy_transfer import transfer_actor_state
        torch.set_num_threads(4)
        settings = contract["evaluation"]
        torch.manual_seed(settings["seed"])
        count = len(settings["cases"]) * settings["episodes_per_case"]
        assert count == 32 and settings == evaluation["settings"]
        runtime = fixed_suite_contract(contract)
        env = FixedCaseEnv(runtime, manifest, load_chassis_control(root / contract["control_math_source"]), root,
            stage_name=contract["enabled_stages"][0], num_envs=count, device="cuda:0",
            seed=settings["seed"], level=1., run_role="quick")
        runtime_initial = deepcopy(env.cfg)
        cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
        cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        cfg.seed, cfg.device = settings["seed"], "cuda:0"
        runner = OnPolicyRunner(env, deepcopy(cfg.to_dict()), log_dir=None, device="cuda:0")
        representative_ids = [env.scene_groups.index(case["name"]) for case in settings["cases"]]
        # Native reset is not a simulator restart. Reproduce the evaluator's
        # preceding actors and full horizons before asserting trajectory identity.
        report["evaluation_prefix_reproduction"] = []
        for index in range(args.candidate_index):
            predecessor = evaluation["candidates"][index]
            path = Path(predecessor["checkpoint"])
            assert digest(path) == predecessor["checkpoint_sha256"]
            saved = torch.load(path, map_location="cpu", weights_only=True)
            source = checkpoint_contract_path(path)
            assert digest(source) == saved["infos"]["contract_sha256"]
            transferred, _ = transfer_actor_state(saved["actor_state_dict"], json.loads(source.read_text()), contract)
            runner.alg.actor.load_state_dict(transferred, strict=True)
            prefix_actor = runner.alg.actor.as_onnx(verbose=False).to("cuda:0").eval()
            observations = env.reset_suite()
            prefix_alive = torch.ones(count, dtype=torch.bool, device="cuda:0")
            samples = []
            with torch.no_grad():
                for tick in range(env.max_episode_length + 1):
                    observations, _, done, extras = env.step(prefix_actor(observations["policy"]))
                    d = extras["diagnostics"]
                    value = torch.cat((d["velocity"], d["omega"], d["height"][:, None],
                        d["position"], d["motor_effort"]), -1)
                    samples.append(value[representative_ids].cpu().numpy().copy())
                    if tick % 4 == 0:
                        env.robot.data.body_link_pose_w.torch[representative_ids].cpu().numpy()
                    prefix_alive &= ~done
                    if not bool(prefix_alive.any()):
                        break
            expected = np.load(args.evaluation / f"candidate_{index:02d}_traces.npz", allow_pickle=False)["values"]
            actual = np.asarray(samples)
            result = dict(candidate_index=index, ticks=len(actual), bitwise_equal=bool(np.array_equal(actual, expected)))
            report["evaluation_prefix_reproduction"].append(result)
            assert result["bitwise_equal"], "Preceding evaluation trajectory differs"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        source_contract = checkpoint_contract_path(checkpoint_path)
        assert digest(source_contract) == state["infos"]["contract_sha256"]
        actor_state, migration = transfer_actor_state(state["actor_state_dict"],
            json.loads(source_contract.read_text()), contract)
        report["observation_migration"] = migration
        runner.alg.actor.load_state_dict(actor_state, strict=True)
        before = {key: value.clone() for key, value in runner.alg.actor.state_dict().items()}
        critic_before = {key: value.clone() for key, value in runner.alg.critic.state_dict().items()}
        actor = runner.alg.actor.as_onnx(verbose=False).to("cuda:0").eval()
        observations = env.reset_suite()
        alive = torch.ones(count, dtype=torch.bool, device="cuda:0")
        names = ("active", "policy_input", "pre_step_commands", "actor_mean_policy", "values",
            "commands", "episode_ticks", "done", "motor_position", "motor_velocity", "actions",
            "wheel_target_velocity", "leg_target_position", "requested_motor_effort", "motor_effort_bounds")
        traces = {name: [] for name in names}
        terminal_records, max_input_command_error = [], 0.
        with torch.no_grad():
            for tick in range(args.ticks):
                actor_input = observations["policy"]
                policy_input = actor_input.clone()
                pre_command = env.commands.clone()
                expected = torch.stack((pre_command[:, 0], pre_command[:, 1], pre_command[:, 2] * 5), -1)
                error = float((policy_input[:, [0, 2, 3]] - expected).abs().max())
                max_input_command_error = max(max_input_command_error, error)
                assert error == 0., "Visible command columns differ from current commands"
                mean = actor(actor_input)
                from wheeled_tasks.chassis.scut_observation import CONTROL_FROM_POLICY
                feedback_q = (env._pc_feedback_transport.feedback[:, :6] if env._pc_feedback_transport is not None
                    else env.robot.data.joint_pos.torch[:, env.ids])
                decoded_legs = env.v5.decode(mean[:, CONTROL_FROM_POLICY], feedback_q)[0]
                before_alive = alive.clone()
                observations, _, done, extras = env.step(mean)
                d = extras["diagnostics"]
                frame = dict(active=before_alive, policy_input=policy_input, pre_step_commands=pre_command,
                    actor_mean_policy=mean, values=torch.cat((d["velocity"], d["omega"], d["height"][:, None],
                        d["position"], d["motor_effort"]), -1),
                    leg_target_position=decoded_legs,
                    requested_motor_effort=env.v5.requested_motor_effort,
                    motor_effort_bounds=env.v5.current_motor_bounds)
                for name in names:
                    traces[name].append((frame[name] if name in frame else d[name]).cpu().numpy().copy())
                if tick % 4 == 0:
                    # Preserve the original evaluator's body-state read cadence.
                    env.robot.data.body_link_pose_w.torch[representative_ids].cpu().numpy()
                for lane in (before_alive & done).nonzero().flatten().cpu().tolist():
                    terminal_records.append(dict(lane=lane, case=env.scene_groups[lane],
                        episode_tick=int(d["episode_ticks"][lane]),
                        reasons=[key for key, value in d["reasons"].items() if bool(value[lane])]))
                alive &= ~done
        traces = {name: np.asarray(values) for name, values in traces.items()}
        np.savez_compressed(args.output / "traces.npz", **traces)
        original = np.load(args.evaluation / f"candidate_{args.candidate_index:02d}_traces.npz", allow_pickle=False)
        comparisons = {}
        for name in ("values", "commands", "episode_ticks", "motor_velocity", "actions", "wheel_target_velocity"):
            left, right = traces[name][:, representative_ids], original[name][:args.ticks]
            comparisons[name] = dict(bitwise_equal=bool(np.array_equal(left, right)),
                max_absolute_error=float(np.max(np.abs(left.astype(float) - right.astype(float)))))
        report["measurement_neutrality_vs_original_evaluation"] = comparisons
        assert all(item["bitwise_equal"] for item in comparisons.values())
        mask = traces["active"] & (traces["episode_ticks"] * .005 > settings["warmup_seconds"])
        mask &= np.arange(args.ticks)[:, None] % 40 == 0
        inputs = torch.from_numpy(traces["policy_input"][mask])
        native_means = torch.from_numpy(traces["actor_mean_policy"][mask])
        report["counterfactual_sample_count"] = len(inputs)
        model_results = {}
        for label, path in (("current", checkpoint_path), ("previous", args.comparison_checkpoint)):
            model = deepcopy(runner.alg.actor).cpu()
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            path_contract = checkpoint_contract_path(path)
            assert digest(path_contract) == checkpoint["infos"]["contract_sha256"]
            transferred, _ = transfer_actor_state(checkpoint["actor_state_dict"],
                json.loads(path_contract.read_text()), contract)
            model.load_state_dict(transferred, strict=True)
            cpu_actor = model.as_onnx(verbose=False).cpu().eval()
            with torch.no_grad():
                real_output = cpu_actor(inputs)
                if label == "current":
                    report["cpu_gpu_actor_max_error"] = float((real_output - native_means).abs().max())
                    assert report["cpu_gpu_actor_max_error"] < 1e-5
                axes = {}
                for axis, column, scale, grid in (("velocity", 0, 1., [-.5, 0., .5]),
                        ("yaw", 2, 1., [-1., 0., 1.]), ("height", 3, 5., [.23, .305, .43])):
                    values = []
                    for command in grid:
                        changed = inputs.clone()
                        changed[:, column] = command * scale
                        keep = [i for i in range(35) if i != column]
                        assert torch.equal(changed[:, keep], inputs[:, keep])
                        output = cpu_actor(changed)
                        clipped = output.clamp(-env.v5.action_bounds[[0, 1, 3, 4, 2, 5]].cpu(),
                            env.v5.action_bounds[[0, 1, 3, 4, 2, 5]].cpu())
                        values.append(dict(command=command, mean_action_policy_order=output.mean(0).tolist(),
                            leg_target_offset_mean_rad=(clipped[:, :4] * env.v5.settings["leg_position_scale"]).mean(0).tolist(),
                            wheel_target_mean_rad_s=(clipped[:, 4:] * env.v5.settings["wheel_velocity_scale"]).mean(0).tolist()))
                    axes[axis] = values
                model_results[label] = dict(checkpoint_sha256=digest(path), axes=axes)
        report.update(counterfactual_actors=model_results, input_command_max_absolute_error=max_input_command_error,
            runtime_contract_unchanged=env.cfg == runtime_initial,
            unchanged_actor=all(torch.equal(value, runner.alg.actor.state_dict()[key]) for key, value in before.items()),
            unchanged_critic=all(torch.equal(value, runner.alg.critic.state_dict()[key]) for key, value in critic_before.items()),
            optimizer_state_empty=not runner.alg.optimizer.state, terminal_records=terminal_records,
            first_episode_survivors=int(alive.sum()), seed=settings["seed"], representative_ids=representative_ids,
            scene_groups=env.scene_groups, policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
            interpretation="A nonzero offline command response only excludes complete input insensitivity on these states; it does not establish correct tracking or identify a unique learning cause.")
        assert report["unchanged_actor"] and report["unchanged_critic"] and report["optimizer_state_empty"]
        assert report["runtime_contract_unchanged"]
        with SummaryWriter(str(checkpoint_path.parent)) as writer:
            clock = state["infos"]["stage_actor_update"]
            writer.add_scalar("LocalDiagnostics/CommandPath/input_command_max_absolute_error", max_input_command_error, clock)
            writer.add_scalar("LocalDiagnostics/CommandPath/measurement_neutrality_exact", 1., clock)
            writer.add_scalar("LocalDiagnostics/CommandPath/first_5s_survivors", int(alive.sum()), clock)
        report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
            trace_sha256=digest(args.output / "traces.npz"))
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
        raise
    finally:
        (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
