#!/usr/bin/env python3
"""Measure small action interventions in isolated native processes, without PPO.

Each invocation starts the same fixed-case scene and changes one policy action
axis only after a one-second common prefix. This is a plant diagnostic, not an
acceptance exam or a training-contract revision.
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


def intervention(tick, axis, sign):
    """Policy-order offset: 100 ms rise, 900 ms plateau, 100 ms fall."""
    if axis not in range(6) or sign not in (-1, 0, 1) or tick < 0:
        raise ValueError("Expected one policy axis, sign -1/0/1 and nonnegative tick")
    scale = max(0., min(1., (tick - 199) / 20., (420 - tick) / 20.))
    values = [0.] * 6
    values[axis] = .1 * sign * scale
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--axis", type=int, choices=range(6), required=True)
    parser.add_argument("--sign", type=int, choices=(-1, 0, 1), required=True)
    parser.add_argument("--reward-credit", action="store_true",
        help="Record native reward components and existing critic observations without optimization")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from train_chassis import preflight
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    evaluation = json.loads((args.evaluation / "evaluation.json").read_text())
    assert evaluation["status"] == "evaluated"
    contract, manifest = preflight(args.evaluation / "contract.json")
    assert digest(args.evaluation / "contract.json") == evaluation["contract_sha256"]
    for name, expected in evaluation["source_sha256"].items():
        assert digest(root / name) == expected, name
    candidate = evaluation["candidates"][args.candidate_index]
    checkpoint = Path(candidate["checkpoint"])
    assert digest(checkpoint) == candidate["checkpoint_sha256"]
    assert contract["actor_dim"] == 35 and contract["policy_dt"] == .005
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    report = dict(schema="v6-local-actuation-probe-v1", status="starting",
        started_at=datetime.now(timezone.utc).isoformat(), ppo_updates=0,
        acceptance_evaluation=False, source_root=str(root), script_sha256=digest(__file__),
        evaluation=str(args.evaluation), evaluation_sha256=digest(args.evaluation / "evaluation.json"),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        axis_policy_order=args.axis, sign=args.sign, peak_policy_offset=.1 * args.sign,
        reward_credit_recording=args.reward_credit,
        ticks=600, prefix_ticks=200, seed=contract["evaluation"]["seed"],
        waveform="ticks200..219 rise .05..1,220..399 plateau,400..419 fall1..0.05,420+ zero",
        scope="Independent-process cold start; pair initial arrays and all200 pre-intervention frames. Original evaluation prefix is not replayed, so identity to its trajectories is not claimed. Closed-loop actor remains active; actual intervention is recorded.")
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.policy_transfer import transfer_actor_state
        from wheeled_tasks.chassis.scut_observation import CONTROL_FROM_POLICY
        torch.set_num_threads(4)
        settings = contract["evaluation"]
        torch.manual_seed(settings["seed"])
        count = len(settings["cases"]) * settings["episodes_per_case"]
        assert count == 32 and settings == evaluation["settings"]
        env = FixedCaseEnv(fixed_suite_contract(contract), manifest,
            load_chassis_control(root / contract["control_math_source"]), root,
            stage_name=contract["enabled_stages"][0], num_envs=count,
            device="cuda:0", seed=settings["seed"], level=1., run_role="quick")
        runtime_before = deepcopy(env.cfg)
        cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
        cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        cfg.seed, cfg.device = settings["seed"], "cuda:0"
        runner = OnPolicyRunner(env, deepcopy(cfg.to_dict()), log_dir=None, device="cuda:0")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source_contract = checkpoint_contract_path(checkpoint)
        assert digest(source_contract) == saved["infos"]["contract_sha256"]
        transferred, _ = transfer_actor_state(saved["actor_state_dict"],
            json.loads(source_contract.read_text()), contract)
        runner.alg.actor.load_state_dict(transferred, strict=True)
        actor_before = {key: value.clone() for key, value in runner.alg.actor.state_dict().items()}
        critic_before = {key: value.clone() for key, value in runner.alg.critic.state_dict().items()}
        assert not runner.alg.optimizer.state

        def forbidden(*unused, **keywords):
            raise RuntimeError("Optimization is forbidden in this zero-PPO probe")

        runner.alg.update = runner.alg.optimizer.step = forbidden
        actor = runner.alg.actor.as_onnx(verbose=False).to("cuda:0").eval()
        observations = env.reset_suite()
        initial = dict(joint_position=env.robot.data.joint_pos.torch.cpu().numpy().copy(),
            joint_velocity=env.robot.data.joint_vel.torch.cpu().numpy().copy(),
            body_pose=env.robot.data.body_link_pose_w.torch.cpu().numpy().copy(),
            cpu_rng=torch.get_rng_state().cpu().numpy().copy(),
            cuda_rng=torch.cuda.get_rng_state().cpu().numpy().copy(),
            env_rng=env.generator.get_state().cpu().numpy().copy())
        np.savez_compressed(args.output / "initial.npz", **initial)
        alive = torch.ones(count, dtype=torch.bool, device="cuda:0")
        records, terminals = {}, []
        clipped_count = requested_count = 0
        with torch.no_grad():
            for tick in range(600):
                policy_input = observations["policy"].clone()
                critic_input = observations["critic"].clone() if args.reward_credit else None
                mean = actor(policy_input)
                offset = mean.new_tensor(intervention(tick, args.axis, args.sign))[None].expand_as(mean)
                offset = offset * alive[:, None]
                sent = mean + offset
                was_alive = alive.clone()
                observations, native_reward, done, extras = env.step(sent)
                d = extras["diagnostics"]
                bounds = env.v5.current_motor_bounds
                requested = env.v5.requested_motor_effort
                active = was_alive[:, None]
                clipped_count += int((((sent[:, CONTROL_FROM_POLICY] - d["actions"]).abs() > 1e-6) & active).sum())
                requested_count += int(((requested.abs() > bounds + 1e-6) & active).sum())
                row = dict(active=was_alive, policy_input=policy_input, actor_mean_policy=mean,
                    applied_offset_policy=offset, sent_action_policy=sent,
                    values=torch.cat((d["velocity"], d["omega"], d["height"][:, None],
                        d["position"], d["motor_effort"]), -1),
                    requested_motor_effort=requested, motor_effort_bounds=bounds)
                for name in ("commands", "episode_ticks", "done", "motor_position", "motor_velocity",
                        "actions", "wheel_target_velocity", "gap", "gravity"):
                    row[name] = d[name]
                if args.reward_credit:
                    components = extras["native_reward_components"]
                    row.update(critic_input_pre=critic_input,
                        critic_input_post=observations["critic"],
                        reward=native_reward, terminated=d["terminated"],
                        timeouts=extras["time_outs"], success=d["success"],
                        observed_event_reward=native_reward - sum(components.values()) * env.policy_dt)
                    for name, value in components.items():
                        row["reward_density__" + name] = value
                for name, value in row.items():
                    records.setdefault(name, []).append(value.cpu().numpy().copy())
                if tick % 4 == 0:
                    env.robot.data.body_link_pose_w.torch.cpu().numpy()
                for lane in (was_alive & done).nonzero().flatten().cpu().tolist():
                    terminals.append(dict(lane=lane, case=env.scene_groups[lane], tick=tick,
                        reasons=[name for name, value in d["reasons"].items() if bool(value[lane])]))
                alive &= ~done
        arrays = {name: np.asarray(value) for name, value in records.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(args.output / "traces.npz", **arrays)
        unchanged_actor = all(torch.equal(value, actor_before[key]) for key, value in runner.alg.actor.state_dict().items())
        unchanged_critic = all(torch.equal(value, critic_before[key]) for key, value in runner.alg.critic.state_dict().items())
        assert unchanged_actor and unchanged_critic and not runner.alg.optimizer.state
        assert env.cfg == runtime_before
        report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
            physics_steps=600 * env.decimation, policy_steps=600, num_envs=count,
            scene_groups=env.scene_groups, terminal_first_episodes=terminals,
            survivors=int(alive.sum()), actor_unchanged=unchanged_actor, critic_unchanged=unchanged_critic,
            optimizer_state_empty=True, runtime_contract_unchanged=True,
            clipped_action_coordinates=clipped_count, bounded_torque_coordinates=requested_count,
            initial_sha256=digest(args.output / "initial.npz"), traces_sha256=digest(args.output / "traces.npz"))
        if args.reward_credit:
            report["reward_credit_scope"] = (
                "Per-lane native rewards and already returned critic observations only. "
                "Observed discrete reward is raw reward minus continuous densities times dt. "
                "Post-step observations may be reset after done; do not bootstrap those states. "
                "The runner critic is unchanged and unused; any saved-model values are computed offline.")
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
