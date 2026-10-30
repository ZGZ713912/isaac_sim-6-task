"""Frozen-checkpoint standing survival diagnostic; no optimizer or training edits."""
from pathlib import Path
import importlib.metadata
import json
import os
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
FROZEN = BASE / "frozen_source"
sys.path.insert(0, str(FROZEN / "scripts"))
sys.path.insert(0, str(FROZEN / "src"))
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
from tensordict import TensorDict
import play_prone_motion as play
from train_scratch_standing import digest, write_json
from wheeled_tasks.chassis import prone_motion as task


def main():
    output = BASE / "probe002"
    output.mkdir(exist_ok=False)
    args = play.parser().parse_args([
        "--task", "WheelLeg-ProneMotion-v3",
        "--checkpoint", str(BASE / "inputs/model_03250.pt"),
        "--run", str(BASE / "inputs"),
        "--scene", "stand", "--num_envs", "96",
        "--seconds", "30", "--seed", "131",
        "--domain_randomization", "off", "--headless",
    ])
    bundle = play.load_bundle(args)
    cfg, settings, control = play.prepare(args, bundle, output)
    receipt = dict(
        status="starting", checkpoint=str(bundle["checkpoint"]),
        checkpoint_sha256=digest(bundle["checkpoint"]),
        completed_updates=bundle["state"]["iter"], seed=131,
        source_commit=bundle["info"]["source_revision"]["commit"],
        source_scope="344 verified frozen source files and committed resources",
        scope="Independent local standing-only snapshot diagnostic, not live training episode records",
        new_PPO=0, optimizer_loaded=False, remote_training_modified=False,
        num_envs=96, horizon_s=30., birth_height_m=.305,
        current_curriculum_targets=True, domain="nominal",
        action_modes=["Gaussian sample from saved distribution", "deterministic mean"],
        actor_normalization="saved statistics held fixed",
        full_capability_qualified=False,
        runtime_versions={n: importlib.metadata.version(n) for n in ("torch", "isaacsim")},
    )
    write_json(output / "receipt.json", receipt)
    app = env = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        manifest = json.loads((FROZEN / cfg["asset_directory"] / "manifest.json").read_text())
        env = ProneMotionEnv(cfg, manifest, control, FROZEN, stage_name="speed",
            num_envs=args.num_envs, device=args.device, seed=bundle["seed"], level=1.,
            run_role="playback", bank=bundle["bank"], settings=settings)
        env.actor_updates = bundle["state"]["iter"]
        cases = []
        for lane in range(env.num_envs):
            height = (.23, .305, .42)[lane % 3]
            cases.append(dict(role="stand", recovery=False, birth_height=.305,
                pose_index=env.height_pose[.305], pitch_deg=0.,
                target=task.target("stand", env.actor_updates, bundle["plan"], height=height),
                command=[0., 0., .305],
                action_mode="sample" if lane < 48 else "mean"))
        env.evaluation_cases = cases
        env.reset(torch.arange(env.num_envs, device=env.device))
        horizon = round(bundle["plan"]["standing_seconds"] / env.policy_dt)
        env.episode_limits.fill_(horizon)
        observation = env.get_observations()
        dummy = TensorDict(dict(policy=torch.zeros(env.num_envs, 35),
            critic=torch.zeros(env.num_envs, 83)), batch_size=[env.num_envs])
        actor = play.build_actor(dummy, bundle, args.device)
        torch.manual_seed(bundle["seed"])
        torch.cuda.manual_seed_all(bundle["seed"])
        initial_xy = (env.robot.data.root_link_pose_w.torch[:, :2] - env.origins[:, :2]).cpu().numpy().copy()
        receipt.update(status="running", policy_dt=env.policy_dt, physics_dt=env.dt,
            cases=cases, ticks=0, pid=os.getpid())
        write_json(output / "receipt.json", receipt)
        write_json(output / "native_startup.json", env.startup_report)
        trace = {}
        started = time.monotonic()
        with torch.inference_mode():
            for tick in range(horizon):
                valid = ~env.finished.clone()
                mean = actor(observation)
                sampled = actor(observation, stochastic_output=True)
                action = mean.clone()
                action[:48] = sampled[:48]
                if not torch.isfinite(action).all():
                    raise RuntimeError("Nonfinite snapshot action")
                observation, _, done, extras = env.step(action)
                state = extras["diagnostics"]
                values = {"valid": valid, "done": done}
                for key in ("time", "terminated", "stable", "height", "velocity", "omega", "gravity",
                            "local_position", "commands", "mechanical", "closure_gap", "impact_bad",
                            "boundary_bad", "course_anchor_error_m", "nonwheel_force_peak_n"):
                    values[key] = state[key]
                for key, value in state["task_failures"].items():
                    values["task_failure_" + key] = value
                for key, value in values.items():
                    trace.setdefault(key, []).append(value.cpu().numpy().copy())
                if (tick + 1) % 500 == 0:
                    receipt.update(ticks=tick + 1, wall_s=time.monotonic() - started,
                        finished_lanes=int(env.finished.sum()))
                    write_json(output / "receipt.json", receipt)
                    print(json.dumps({k: receipt[k] for k in ("ticks", "wall_s", "finished_lanes")}), flush=True)
                if env.finished.all():
                    break
        arrays = {k: np.stack(v) for k, v in trace.items()}
        np.savez_compressed(output / "trace.npz", initial_xy=initial_xy, **arrays)
        rows = []
        for lane, case in enumerate(cases):
            n = int(arrays["valid"][:, lane].sum())
            s = slice(0, n)
            terminated = bool(arrays["terminated"][s, lane].any())
            tilt = np.degrees(np.arccos(np.clip(-arrays["gravity"][s, lane, 2], -1., 1.)))
            distance = np.linalg.norm(arrays["local_position"][s, lane, :2] - initial_xy[lane], axis=-1)
            run = longest = 0
            for stable in arrays["stable"][s, lane]:
                run = run + 1 if stable else 0
                longest = max(longest, run)
            reason_keys = [k for k in arrays if k.startswith("task_failure_") or k in
                ("mechanical", "closure_gap", "impact_bad", "boundary_bad")]
            rows.append(dict(lane=lane, action_mode=case["action_mode"], target_height_m=case["target"][2],
                survival_observed_s=n * env.policy_dt, terminated=terminated,
                survived_full30=n == horizon and not terminated,
                right_censored_at30=n == horizon and not terminated,
                longest_strict_stable_s=longest * env.policy_dt,
                xy_max_m=float(distance.max()), xy_final_m=float(distance[-1]),
                tilt_max_deg=float(tilt.max()), tilt_mean_deg=float(tilt.mean()),
                height_mae_m=float(np.abs(arrays["height"][s, lane] - arrays["commands"][s, lane, 2]).mean()),
                termination_reasons=[k for k in reason_keys if arrays[k][s, lane].any()]))
        summary = {}
        for mode in ("sample", "mean"):
            selected = [r for r in rows if r["action_mode"] == mode]
            times = np.array([r["survival_observed_s"] for r in selected])
            summary[mode] = dict(n=len(selected), full30_count=sum(r["survived_full30"] for r in selected),
                median_observed_s=float(np.median(times)), mean_observed_s=float(times.mean()),
                min_observed_s=float(times.min()), p10_observed_s=float(np.quantile(times, .1)),
                p90_observed_s=float(np.quantile(times, .9)),
                max_xy_m=float(max(r["xy_max_m"] for r in selected)),
                median_xy_max_m=float(np.median([r["xy_max_m"] for r in selected])),
                longest_strict_stable_s=float(max(r["longest_strict_stable_s"] for r in selected)))
            summary[mode]["by_target_height"] = {}
            for height in sorted({r["target_height_m"] for r in selected}):
                group = [r for r in selected if r["target_height_m"] == height]
                summary[mode]["by_target_height"][str(height)] = dict(n=len(group),
                    full30_count=sum(r["survived_full30"] for r in group),
                    median_observed_s=float(np.median([r["survival_observed_s"] for r in group])),
                    mean_height_mae_m=float(np.mean([r["height_mae_m"] for r in group])))
        receipt.update(status="completed_unqualified", ticks=len(arrays["valid"]),
            wall_s=time.monotonic() - started, summary=summary, rows=rows,
            trace_sha256=digest(output / "trace.npz"),
            interpretation="30-second survivors are right-censored; survival alone is not stable-standing qualification")
        write_json(output / "receipt.json", receipt)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        receipt.update(status="failed", error=traceback.format_exc())
        write_json(output / "receipt.json", receipt)
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.app.close()


if __name__ == "__main__":
    main()
