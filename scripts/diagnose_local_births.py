#!/usr/bin/env python3
"""Observe first training births at a checkpoint clock; no PPO or acceptance grade."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=250)
    parser.add_argument("--release-ramp-seconds", type=float, choices=(0., .1), default=0.,
                        help="Diagnostic-only post-hold effort ramp; production contract stays unchanged")
    args = parser.parse_args()
    if not 1 <= args.ticks <= 1000:
        parser.error("first-birth probe is bounded to 1..1000 policy ticks")
    from train_chassis import digest, preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
    contract_path = checkpoint_contract_path(args.checkpoint)
    contract, manifest = preflight(contract_path)
    count = contract["target_num_envs"]
    assert count == 512 and contract["policy_dt"] == .005
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="v6-local-first-birth-diagnosis-v1", status="starting",
        started_at=datetime.now(timezone.utc).isoformat(), ppo_updates=0,
        acceptance_evaluation=False, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), contract_sha256=digest(contract_path),
        script_sha256=digest(Path(__file__)), num_envs=count, ticks=args.ticks,
        noise_seed=290933, seed=290927, release_ramp_seconds=args.release_ramp_seconds,
        recipe="release_ramp_100ms" if args.release_ramp_seconds else "native",
        scope="One fresh batch at checkpoint curriculum clock. First births only; later auto-resets excluded. Not a native rollout replay.",
        sampler="unchanged deterministic actor mean plus independent standard Gaussian times learned native std")
    (args.output / "contract.json").write_bytes(contract_path.read_bytes())
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, enable_cameras=False, device="cuda:0",
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_tasks.chassis.env import ChassisEnv
        torch.set_num_threads(4)
        torch.manual_seed(report["seed"])
        env = ChassisEnv(contract, manifest,
            load_chassis_control(ROOT / contract["control_math_source"]), ROOT,
            stage_name="speed", num_envs=count, device="cuda:0", level=1., seed=report["seed"])
        cfg = json.loads((args.checkpoint.parent / "agent_config.json").read_text())
        runner = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cuda:0")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        assert checkpoint["infos"]["contract_sha256"] == digest(contract_path)
        runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        actor = runner.alg.actor.as_onnx(verbose=False).to("cuda:0").eval()
        scale = runner.alg.actor.distribution.std_param.detach().clone()
        clock = checkpoint["infos"]["stage_actor_update"]
        env.stage_actor_update = clock
        env.global_actor_update = checkpoint["infos"]["global_actor_update"]
        env.training_transitions = checkpoint["infos"]["training_transitions"]
        env.generator.set_state(checkpoint["infos"]["env_rng_state"])
        env.reset(torch.arange(count, device="cuda:0"))
        observations = env.get_observations()
        births = env.nonstandard_start
        profile = births.profile_ids.clone()
        selected = births.selected.clone()
        initial_input = observations["policy"].cpu().numpy().copy()
        if args.release_ramp_seconds:
            original_apply = births.apply_torque

            def apply_with_ramp(effort):
                result = original_apply(effort)
                elapsed = (births.episode_steps - births.hold_steps + 1) * env.dt
                factor = (elapsed / args.release_ramp_seconds).clamp(0., 1.)
                factor = torch.where(births.selected, factor, torch.ones_like(factor))
                return result * factor[:, None]

            births.apply_torque = apply_with_ramp
        alive = torch.ones(count, device="cuda:0", dtype=torch.bool)
        max_gap = torch.zeros(count, device="cuda:0")
        terminal_ticks = torch.zeros(count, device="cuda:0", dtype=torch.long)
        generator = torch.Generator(device="cuda:0").manual_seed(report["noise_seed"])
        records, gap_trace, state_trace = [], [], []
        with torch.no_grad():
            for tick in range(args.ticks):
                normal = torch.randn((count, 6), generator=generator, device="cuda:0")
                if tick == 0:
                    import hashlib
                    report["first_standard_normal_sha256"] = hashlib.sha256(normal.cpu().numpy().tobytes()).hexdigest()
                observations, _, done, extras = env.step(actor(observations["policy"]) + normal * scale)
                d = extras["diagnostics"]
                gap_trace.append(torch.where(alive, d["gap"], torch.nan).cpu().numpy().copy())
                state = torch.cat([d[k] for k in ("motor_position", "motor_velocity", "motor_effort",
                                                 "gravity", "omega", "position")], -1)
                state_trace.append(torch.where(alive[:, None], state, torch.nan).cpu().numpy().copy())
                max_gap = torch.maximum(max_gap, torch.where(alive, d["gap"], 0.))
                ended = alive & done
                terminal_ticks[ended] = d["episode_ticks"][ended]
                for lane in ended.nonzero().flatten().cpu().tolist():
                    pid = int(profile[lane])
                    records.append(dict(lane=lane, scene_group=env.scene_groups[lane],
                        profile_id=pid, profile_name=births.profile_names[pid] if pid >= 0 else "nominal",
                        episode_tick=int(d["episode_ticks"][lane]), gap_m=float(d["gap"][lane]),
                        reasons=[k for k, v in d["reasons"].items() if bool(v[lane])],
                        birth_elapsed_seconds=float(d["nonstandard_start_birth_elapsed_seconds"][lane]),
                        holding=bool(d["nonstandard_start_holding"][lane]),
                        motor_velocity=d["motor_velocity"][lane].cpu().tolist(),
                        motor_effort=d["motor_effort"][lane].cpu().tolist(),
                        gravity=d["gravity"][lane].cpu().tolist()))
                alive &= ~done
                report["completed_ticks"] = tick + 1
                if not alive.any():
                    break
        report.update(actor_unchanged=all(torch.equal(v, runner.alg.actor.state_dict()[k]) for k, v in before.items()),
            curriculum_actor_update=clock, learned_std_policy_order=scale.cpu().tolist(),
            first_birth_profiles=[births.profile_names[int(pid)] if int(pid) >= 0 else "nominal" for pid in profile.cpu()],
            selected_birth_count=int(selected.sum()), unfinished_first_episodes=int(alive.sum()),
            terminal_records=records, max_closure_gap_m=float(max_gap.max()))
        report["optimizer_state_empty"] = not runner.alg.optimizer.state
        assert report["actor_unchanged"] and report["optimizer_state_empty"]
        np.savez_compressed(args.output / "births.npz", initial_policy_input=initial_input,
            initial_profile_id=profile.cpu().numpy(), selected=selected.cpu().numpy(),
            terminal_ticks=terminal_ticks.cpu().numpy(), max_gap_m=max_gap.cpu().numpy(),
            unfinished=alive.cpu().numpy(), first_birth_gap_m=np.asarray(gap_trace),
            first_birth_state=np.asarray(state_trace))
        report["state_trace_columns"] = ["motor_position6", "motor_velocity6", "motor_effort6",
                                          "gravity3", "omega3", "position3"]
        with SummaryWriter(str(args.checkpoint.parent)) as writer:
            writer.add_scalar(f"LocalDiagnostics/FirstBirths/{report['recipe']}/closure_events",
                sum("closure_gap" in r["reasons"] for r in records), clock)
            writer.add_scalar(f"LocalDiagnostics/FirstBirths/{report['recipe']}/unfinished_first_episodes", int(alive.sum()), clock)
        report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
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
