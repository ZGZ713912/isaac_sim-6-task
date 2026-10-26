"""Record exact115 native PhysX body poses without blocking the display process."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).parent)]
from gui118_getup import MODEL_SHA, NAMES, prepare, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    cfg, task, manifest, control, mean, info = prepare(args.output, 211322, num_envs=8)
    receipt = dict(schema="native115-pose-recording-v1", status="starting", pid=os.getpid(),
        checkpoint_sha256=MODEL_SHA, source_sha256=sha(__file__), viewer_preparation_sha256=sha(Path(__file__).with_name("gui118_getup.py")),
        policy_hz=200, physics_hz=1000, source_seed=211322, domain_scale=task["domain_scale"],
        qualification=None, optimizer_updates=0, physics_parameters_changed=False,
        scope="Eight newly recorded native case lanes for unscored viewing; not prior single-lane domain pairing", clips=[])
    write(args.output / "recording.json", receipt)
    app = env = None
    interrupted = False
    def stop(_sig, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(dict(headless=True, device="cuda:0", enable_cameras=False))
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from isaaclab.sim.utils.stage import create_new_stage
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from train_robust_getup import fixed_cases
        torch.set_num_threads(4)
        create_new_stage()
        env = RobustGetupEnv(cfg, manifest, control, ROOT, stage_name="speed", num_envs=8,
            device="cuda:0", seed=211322, level=1., run_role="playback",
            bank=Path(__file__).parent / "pose_bank002.json", settings=task)
        assert env.recovery is None and env.sim.cfg.use_fabric
        templates = {case["name"]: case for case in fixed_cases(env.bank, 35, 211322)}
        cases = [deepcopy(templates[name]) for name in NAMES[:8]]
        env.fixed_cases = cases
        env.reset(torch.arange(8, device=env.device))
        receipt.update(status="recording", cases=cases, body_names=env.robot.body_names,
            asset_directory=str(ROOT / cfg["asset_directory"]), usd=manifest["usd"],
            origins=env.origins.cpu().tolist(), started_unix=time.time(), last_simulated_seconds=0.)
        write(args.output / "recording.json", receipt)
        mean = mean.cuda().eval()
        rows, saved = [], set()
        keys = ("time", "gravity", "height", "motor_position", "motor_velocity", "motor_effort",
                "actions", "gap", "passive", "stable", "recovered", "followup_success", "terminated", "local_position")
        with SummaryWriter(str(args.output), filename_suffix=".gui120_native") as writer, torch.inference_mode():
            start = time.monotonic()
            for tick in range(2400):
                if interrupted:
                    break
                obs = env.get_observations()["policy"]
                _, _, done, extras = env.step(mean(obs))
                d = extras["diagnostics"]
                rows.append({key: d[key].cpu().numpy() for key in keys}
                    | {"body_link_pose_w": env.robot.data.body_link_pose_w.torch.cpu().numpy()})
                terminal = done.cpu().numpy()
                for lane in range(8):
                    if lane in saved or not (terminal[lane] or tick == 2399):
                        continue
                    saved.add(lane)
                    path = args.output / (NAMES[lane] + ".npz")
                    np.savez_compressed(path, **{key: np.stack([row[key][lane:lane+1] for row in rows]) for key in rows[0]})
                    receipt["clips"].append(dict(case=NAMES[lane], trajectory=str(path), sha256=sha(path),
                        lane=lane, seconds=float(d["time"][lane]), native_terminated=bool(d["terminated"][lane]),
                        ever_recovered=bool(any(row["recovered"][lane] for row in rows)),
                        window_complete=tick == 2399, checkpoint_sha256=MODEL_SHA))
                    print("GUI120_CLIP " + json.dumps(receipt["clips"][-1]), flush=True)
                    write(args.output / "recording.json", receipt)
                if tick % 40 == 0:
                    receipt.update(last_simulated_seconds=float(d["time"].max()), heartbeat_unix=time.time(), wall_seconds=time.monotonic()-start)
                    write(args.output / "recording.json", receipt)
                    for lane,name in enumerate(NAMES[:8]):
                        writer.add_scalar("GetupRecorded/"+name+"/recovered",float(d["recovered"][lane]),tick)
                        writer.add_scalar("GetupRecorded/"+name+"/closure_gap_m",float(d["gap"][lane]),tick)
                if len(saved) == 8:
                    break
        receipt.update(status="interrupted" if interrupted else "completed", finished_unix=time.time())
        write(args.output / "recording.json", receipt)
    except BaseException:
        receipt.update(status="failed", error=traceback.format_exc())
        write(args.output / "recording.json", receipt)
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.app.close()


if __name__ == "__main__":
    main()
