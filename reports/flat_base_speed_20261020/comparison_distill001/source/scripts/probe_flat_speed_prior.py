#!/usr/bin/env python3
"""Paired zero-PPO native test of an encoder wheel-error prior.

Each arm shares birth jitter and contact coefficients. This diagnostic cannot
qualify a release, alter the fixed acceptance suite, or save trained weights.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("wheel_error", "reachable_reference", "lean_reference", "state_reference", "progressive_reference", "checkpoint_compare"), default="wheel_error")
    parser.add_argument("--lean-gains", type=float, nargs=4)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-variants", type=Path, nargs="+", default=[])
    parser.add_argument("--include-reference-teacher", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--seed", type=int, default=191119)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False, parents=True)
    parent = ROOT / "runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt"
    assert digest(parent) == "39f7f2d364d82905741478fd254c7fdde3d9c101a613cfba1ba79d14a483fd73"
    source = ROOT / "runs/flat_base_20261019/ground025_static1338_maximum_envelope001/config/contract.json"
    original, manifest = preflight(source)
    design = json.loads((ROOT / original["design_preflight"]["file"]).read_text())
    row = deepcopy(design["stages"]["speed"]["roles"]["quick"])
    gains, heights, friction, replicas, horizon, seed = (0., .25, .5, 1.), (.305, .42), (.2, .5), 2, 8., args.seed
    if args.mode == "reachable_reference":
        gains, horizon = (0., .15, .3, .6), 12.
    selected = ("stand_305mm", "forward_05", "backward_05", "rotate_1", "rotate_1_reverse",
        "forward_2", "backward_2", "forward_3", "backward_3", "rotate_4", "rotate_4_reverse")
    if args.mode == "lean_reference":
        gains, horizon = (0., .01, .02, .04), 12.
        if args.lean_gains is not None:
            if any(not -.04 <= x <= .04 for x in args.lean_gains) or args.lean_gains[0] != 0.:
                raise ValueError("Require baseline zero and gains within +/-.04")
            gains = tuple(args.lean_gains)
        selected = ("forward_2", "backward_2", "forward_3", "backward_3", "rotate_4", "rotate_4_reverse")
    if args.mode == "checkpoint_compare":
        if args.candidate is None or not args.candidate.is_file():
            raise ValueError('Checkpoint comparison requires a sealed candidate')
        if any(not p.is_file() for p in args.candidate_variants):
            raise ValueError('Every comparison variant must be a saved checkpoint')
        gains, horizon = tuple(float(i) for i in range(3+len(args.candidate_variants)+int(args.include_reference_teacher))), 12.
    if args.mode == "state_reference":
        gains, horizon = (0.,.25,.5,.75,1.),12.
        selected = ("forward_2", "backward_2", "forward_3", "backward_3", "rotate_4", "rotate_4_reverse")
    if args.mode == "progressive_reference":
        gains, horizon = (0.,1.,2.,3.),12.
        selected = ("forward_2", "backward_2", "forward_3", "backward_3", "rotate_4", "rotate_4_reverse")
    if args.seconds is not None:
        if not 8. <= args.seconds <= 32.:
            raise ValueError('Diagnostic horizon must stay in [8,32] seconds')
        horizon = args.seconds

    def project(settings):
        settings = deepcopy(settings)
        catalogue = {c["name"]: c for c in settings["cases"]}
        cases = []
        for gain in gains:
            for height in heights:
                for mu in friction:
                    for name in selected:
                        case = deepcopy(catalogue[name])
                        case.update(name=f"g{gain:g}_h{height:g}_mu{mu:g}_{name}", reset_seed_key=name,
                            episode_seconds=horizon, warmup_seconds=4.,
                            contact_material_profile=dict(ground_static_friction=mu, ground_dynamic_friction=mu),
                            speed_prior_gain=gain, original_case=name)
                        case["command"][2] = height
                        case["skill"]["command"][2] = height
                        cases.append(case)
        settings.update(cases=cases, episode_seconds=horizon, episodes_per_case=replicas,
            seed=seed, warmup_seconds=4., diagnostic_only=True, qualification_eligible=False,
            protocol_id="flat-speed-encoder-prior-paired-diagnostic-v1")
        return settings

    settings = project(original["evaluation"])
    expected = project(row["contract"]["evaluation"])
    assert settings == expected
    count = len(settings["cases"]) * replicas
    config = deepcopy(original)
    for mapping, evaluation in ((config, settings), (row["contract"], expected)):
        mapping.update(evaluation=evaluation, episode_seconds=horizon, target_num_envs=count,
            ppo_minibatch_samples=count * mapping["num_steps_per_env"] // mapping["num_mini_batches"],
            scene_groups=[dict(name=c["name"], fraction=1 / len(evaluation["cases"]), terrain=["flat"])
                          for c in evaluation["cases"]])
        mapping.pop("rmcs_maximum_height_replay", None)
    row["invocation"].update(num_envs=count)
    row["invocation"]["seed"]["value"] = seed
    row["contract"]["evaluation.seed"] = seed
    row["runtime"]["seed"] = seed
    design["stages"]["speed"]["roles"]["quick"] = row
    design["scope"] = "zero_PPO_" + args.mode + "_diagnostic"
    design_path = args.output / "design.json"
    design_path.write_bytes(encode(design))
    config["design_preflight"].update(file=str(design_path), sha256=digest(design_path))
    (args.output / "contract.json").write_bytes(encode(config))
    control = load_chassis_control(ROOT / config["control_math_source"])
    static = design_preflight(config, control, ROOT, stage="speed", role="quick", seed=seed, num_envs=count)
    report = dict(status="prepared", started_at=datetime.now(timezone.utc).isoformat(),
        diagnostic_only=True, qualification_eligible=False, new_optimizer_updates=0,
        checkpoint_sha256=digest(parent), source_contract_sha256=digest(source),
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__), ROOT / "scripts/flat_speed_prior.py", ROOT / "src/wheeled_tasks/chassis/env.py"]},
        cpu_affinity=sorted(os.sched_getaffinity(0)), policy_hz=50, physics_hz=1000,
        gains=list(gains), heights=list(heights), ground_friction=list(friction), replicas=replicas,
        probe_mode=args.mode,
        seed=seed,
        static_preflight=static, qualified=False, checkpoint_saved=False)
    (args.output / "report.json").write_bytes(encode(report))
    app, env = None, None
    start = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False,
            kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from wheeled_algo.command_residual import mean_from_checkpoint
        from wheeled_tasks.chassis.eval_env import FixedCaseEnv
        from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
        from wheeled_tasks.chassis.evaluation import grade_fixed_suite
        from flat_rmcs_command_reference import install_reference
        from flat_speed_prior import correct, reachable_reference, lean_reference, state_reference, progressive_reference
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        state = torch.load(parent, map_location="cpu", weights_only=True)
        actor = mean_from_checkpoint(state, ROOT / "models/wheel_leg_flat_base_20261018/source.onnx").to("cuda:0").eval()
        actors = None
        if args.mode == 'checkpoint_compare':
            paths = {0:parent,1:args.candidate.resolve().parent/'initial_state.pt',2:args.candidate.resolve()}
            paths.update({3+i:p.resolve() for i,p in enumerate(args.candidate_variants)})
            actors = {}
            report['checkpoint_comparison'] = {}
            for index,path in paths.items():
                checkpoint = torch.load(path,map_location='cpu',weights_only=True)
                actors[index] = mean_from_checkpoint(checkpoint,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx').to('cuda:0').eval()
                role=('parent','initialized','trained')[index] if index<3 else f'variant_{index-2}'
                report['checkpoint_comparison'][str(index)] = dict(role=role,
                    path=str(path),sha256=digest(path),successful_updates=checkpoint['infos']['successful_updates_total'],
                    actor_updates=checkpoint['infos']['actor_updates_total'],
                    supervised_head_fit=checkpoint['infos'].get('supervised_head_fit'))
            if args.include_reference_teacher:
                index=len(actors)
                def teacher_mean(policy):
                    arm=torch.where(policy[:,0:1]<-.75,0.,1.)
                    return actor(progressive_reference(policy,arm))
                actors[index]=teacher_mean
                report['checkpoint_comparison'][str(index)]=dict(role='reference_teacher',
                    path=str(parent),sha256=digest(parent),successful_updates=state['infos']['successful_updates_total'],
                    actor_updates=state['infos']['actor_updates_total'],new_PPO_updates=0,
                    reference=dict(reverse_arm=0.,yaw_positive_arm=1.,yaw_negative_arm=1.,
                        source_sha256=digest(ROOT/'scripts/flat_speed_prior.py')))
        env = FixedCaseEnv(config, manifest, control, ROOT, stage_name="speed", num_envs=count,
            device="cuda:0", seed=seed, level=1., run_role="quick")
        reference = install_reference(env, config["rmcs_command_reference_replay"])
        report["startup"] = env.startup_report
        obs = env.reset_suite()
        gain = torch.tensor([[env.cases[g]["speed_prior_gain"]] for g in env.scene_groups], device=env.device)
        actor_lanes = {index:(gain[:,0]==index).nonzero(as_tuple=True)[0] for index in actors} if actors else None
        metrics = EpisodeMetrics(env.scene_groups, env.device, .02, 4., height_range_m=[.23, .43])
        alive = torch.ones(count, dtype=torch.bool, device=env.device)
        traces = {key: [] for key in ("active", "observations", "actions", "episode_ticks", "done", "commands",
            "velocity", "omega", "height", "position", "gravity", "gap", "motor_velocity", "motor_effort")}
        births = env.robot.data.root_link_pose_w.torch.clone()
        birth_joints = env.robot.data.joint_pos.torch.clone()
        # Explicit profile materials and reset_seed_key make paired arms equal
        # before any policy action, even when native articulation order differs.
        paired = {}
        birth_error = 0.
        for g in env.cases:
            key = (env.cases[g]["command"][2], env.cases[g]["contact_material_profile"]["ground_static_friction"], env.cases[g]["original_case"])
            ids = sorted([i for i, n in enumerate(env.scene_groups) if n == g],
                         key=lambda i: env.clone_indices[i] % replicas)
            local = births[ids].clone()
            local[:, :3] -= env.origins[ids]
            local = torch.cat((local, birth_joints[ids], env.fixed_reset_samples[ids]), -1)
            if key in paired:
                # Float32 world origin quantization is recorded, not called exact.
                birth_error = max(birth_error, float((local-paired[key]).abs().max()))
                assert torch.allclose(local, paired[key], rtol=0., atol=3e-5)
            else:
                paired[key] = local
        report["paired_birth_pose_max_tolerance_m"] = 3e-5
        report["paired_birth_root_joints_samples_max_abs_error"] = birth_error
        report["status"] = "running"
        (args.output / "report.json").write_bytes(encode(report))
        with torch.inference_mode():
            for tick in range(round(horizon / .02) + 1):
                policy = obs["policy"]
                if actors:
                    actions = torch.zeros(count,6,device=env.device)
                    for index,mean in actors.items():
                        ids=actor_lanes[index]
                        actions[ids]=mean(policy[ids])
                elif args.mode == "wheel_error":
                    actions = correct(policy, actor(policy), gain)
                elif args.mode == "state_reference":
                    actions = actor(state_reference(policy,gain))
                elif args.mode == "progressive_reference":
                    actions = actor(progressive_reference(policy,gain))
                else:
                    reference_obs = (lean_reference(policy, gain) if args.mode == "lean_reference"
                                     else reachable_reference(policy, gain))
                    actions = actor(reference_obs)
                obs, _, done, extras = env.step(actions)
                diagnostic = extras["diagnostics"]
                metrics.observe(diagnostic, alive)
                for key in traces:
                    value = {"active": alive, "observations": policy, "actions": actions}.get(key)
                    if value is None:
                        value = diagnostic[key]
                    traces[key].append(value.cpu().numpy().copy())
                alive &= ~done.bool()
                if tick % 50 == 0:
                    print("SPEED_PRIOR", json.dumps(dict(tick=tick, active=int(alive.sum()),
                        elapsed_s=round(time.monotonic()-start, 2))), flush=True)
                if not bool(alive.any()):
                    break
        result = grade_fixed_suite(metrics.report(), settings, replicas)
        result.update(passed=False, diagnostic_only=True, qualification_eligible=False)
        report.update(status="evaluated", result=result, elapsed_s=time.monotonic()-start)
        np.savez_compressed(args.output / "traces.npz", **{k: np.asarray(v) for k, v in traces.items()},
            groups=np.asarray(env.scene_groups))
        with SummaryWriter(str(args.output)) as writer:
            for name, case in result["cases"].items():
                for key in ("vx_mae_m_s", "yaw_mae_rad_s", "survival_rate", "height_mae_m", "closure_gap_max_m"):
                    writer.add_scalar("FlatSpeedPriorDiagnostic/" + name + "/" + key, case[key], 1338)
        print("SPEED_PRIOR_DONE", json.dumps({k: dict(vx=v["vx_mae_m_s"], yaw=v["yaw_mae_rad_s"],
            full=v["full_horizon_episodes"]) for k, v in result["cases"].items()}), flush=True)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_bytes(encode(report))
        if env is not None:
            env.close()
        if app is not None:
            app.app.close()
    return 0 if report["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
