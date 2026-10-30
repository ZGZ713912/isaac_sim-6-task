#!/usr/bin/env python3
"""Paired first-episode Gaussian probes; no PPO, no acceptance decision."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RECIPES = {
    "uniform_020": [.2] * 6,
    "legs_005_wheels_050": [.05] * 4 + [.5] * 2,
    "legs_005_wheels_100": [.05] * 4 + [1.] * 2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=4, choices=(4,))
    parser.add_argument("--recipe", choices=tuple(RECIPES), required=True,
                        help="One recipe per fresh simulator process; compare saved initial/noise hashes")
    args = parser.parse_args()
    from train_chassis import digest, preflight
    from wheeled_tasks.chassis.local_evaluation import write_bundle
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    contract_path = write_bundle(ROOT, ROOT / "runs/local_exploration_probes" / stamp)
    contract, manifest = preflight(contract_path)
    settings = contract["evaluation"]
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="v6-local-exploration-probe-v1", status="starting",
        started_at=datetime.now(timezone.utc).isoformat(), ppo_updates=0,
        acceptance_evaluation=False, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), contract_sha256=digest(contract_path),
        settings=settings, replicas=args.replicas, noise_seed=290931,
        recipe_policy_order="left_leg1,left_leg2,right_leg1,right_leg2,left_wheel,right_wheel",
        recipes=RECIPES, source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        results={})
    (args.output / "contract.json").write_bytes(contract_path.read_bytes())
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    launcher = env = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, enable_cameras=False, device="cuda:0",
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.agents.v40_ppo_cfg import V40PPORunnerCfg
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        torch.set_num_threads(4)
        torch.manual_seed(settings["seed"])
        count = len(settings["cases"]) * args.replicas
        env = FixedCaseEnv(fixed_suite_contract(contract), manifest,
            load_chassis_control(ROOT / contract["control_math_source"]), ROOT,
            stage_name="speed", num_envs=count, device="cuda:0", seed=settings["seed"],
            level=1., run_role="quick")
        cfg = handle_deprecated_rsl_rl_cfg(V40PPORunnerCfg(), "5.5.1")
        cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        cfg.seed, cfg.device = settings["seed"], "cuda:0"
        runner = OnPolicyRunner(env, deepcopy(cfg.to_dict()), log_dir=None, device="cuda:0")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["infos"]["successful_updates_total"] != 0:
            raise ValueError("Exploration probe requires an authentic original initial checkpoint")
        runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        actor = runner.alg.actor.as_onnx(verbose=False).to("cuda:0").eval()
        model_before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        for name, std in ((args.recipe, RECIPES[args.recipe]),):
            report["active_recipe"] = name
            observations = env.reset_suite()
            initial = observations["policy"].cpu().numpy().copy()
            signature = hashlib.sha256(initial.tobytes()).hexdigest()
            generator = torch.Generator(device="cuda:0").manual_seed(report["noise_seed"])
            scale = torch.tensor(std, device="cuda:0")
            metrics = EpisodeMetrics(env.scene_groups, "cuda:0", env.policy_dt, settings["warmup_seconds"])
            alive = torch.ones(count, dtype=torch.bool, device="cuda:0")
            terminal_ticks = torch.zeros(count, device="cuda:0", dtype=torch.int64)
            gaps = torch.zeros(count, device="cuda:0")
            wheel_square = torch.zeros(count, 2, device="cuda:0")
            action_square = torch.zeros(count, 6, device="cuda:0")
            exposure = torch.zeros(count, device="cuda:0")
            with torch.no_grad():
                for tick in range(env.max_episode_length + 1):
                    normal = torch.randn((count, 6), generator=generator, device="cuda:0")
                    if tick == 0:
                        noise_signature = hashlib.sha256(normal.cpu().numpy().tobytes()).hexdigest()
                    actions = actor(observations["policy"]) + normal * scale
                    observations, _, done, extras = env.step(actions)
                    report["last_completed_tick"] = tick + 1
                    d = extras["diagnostics"]
                    metrics.observe(d, alive)
                    gaps = torch.maximum(gaps, torch.where(alive, d["gap"], 0.))
                    wheel_square += d["wheel_target_velocity"].square() * alive[:, None]
                    action_square += d["actions"].square() * alive[:, None]
                    exposure += alive
                    terminal_ticks.copy_(torch.where(alive & done, d["episode_ticks"], terminal_ticks))
                    alive &= ~done
                    if not alive.any():
                        break
            cases = metrics.report()["groups"]
            item = dict(initial_observation_sha256=signature, standard_normal_first_tick_sha256=noise_signature,
                paired_comparison="Compare initial observation and noise hashes across fresh processes before inference",
                unfinished_episodes=int(alive.sum()),
                policy_ticks=tick + 1, terminal_time_mean_s=float(terminal_ticks.float().mean()) * env.policy_dt,
                closure_gap_max_m=float(gaps.max()),
                closure_terminated_episodes=sum(v["termination_reasons"].get("closure_gap", 0) for v in cases.values()),
                nonwheel_terminated_episodes=sum(v["termination_reasons"].get("nonwheel_contact", 0) for v in cases.values()),
                cases=cases)
            np.savez_compressed(args.output / (name + ".npz"), initial_policy_input=initial,
                terminal_ticks=terminal_ticks.cpu().numpy(), max_gap_m=gaps.cpu().numpy(),
                wheel_target_rms_rad_s=(wheel_square / exposure[:, None].clamp_min(1)).sqrt().cpu().numpy(),
                control_action_rms=(action_square / exposure[:, None].clamp_min(1)).sqrt().cpu().numpy())
            report["results"][name] = item
            print("EXPLORATION_PROBE", name, item["terminal_time_mean_s"], item["closure_terminated_episodes"], flush=True)
        report["actor_unchanged"] = all(torch.equal(v, runner.alg.actor.state_dict()[k]) for k, v in model_before.items())
        if not report["actor_unchanged"]:
            raise ValueError("Exploration probe changed the actor")
        event_dir = args.checkpoint.parent.parent if args.checkpoint.parent.name == "initial_state" else args.checkpoint.parent
        with SummaryWriter(str(event_dir)) as writer:
            for name, item in report["results"].items():
                for metric in ("terminal_time_mean_s", "closure_gap_max_m", "closure_terminated_episodes", "nonwheel_terminated_episodes"):
                    writer.add_scalar(f"LocalDiagnostics/AxisExploration/{name}/{metric}", item[metric], 0)
        report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
            interpretation="Untrained stochastic first episodes only; no learned behavior, acceptance, or long-run admission.")
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
