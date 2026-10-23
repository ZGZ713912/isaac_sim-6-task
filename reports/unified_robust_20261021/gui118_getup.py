"""Unscored single-robot native viewer for the composed115 pure-RL actor."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
SOURCE = ROOT / "runs/unified_robust_20261022/composed_static_original115"
MODEL = ROOT / "runs/unified_robust_20261022/composed_getup115/model_migrated.pt"
MODEL_SHA = "11c4451ab73c4935051fc0187a5fc3a70a3534cee67f22d393500f623bd9f121"
NAMES = ("left_symmetric", "left_asymmetric", "right_symmetric", "right_asymmetric",
         "inverted_symmetric", "inverted_asymmetric", "front_symmetric", "back_symmetric",
         "stand_low", "stand", "stand_high")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def prepare(output, seed):
    """Project the runtime and its independently frozen train expectations."""
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from export_robust_getup import load_mean

    if sha(MODEL) != MODEL_SHA:
        raise ValueError("The current115 candidate changed")
    mean, info = load_mean(MODEL)
    report = json.loads((SOURCE / "robust/report.json").read_text())
    if report["checkpoint_sha256"] != MODEL_SHA or report["task_contract_sha256"] != sha(SOURCE / "task_contract.json"):
        raise ValueError("Source native receipt does not authenticate this actor/task")
    cfg = json.loads((SOURCE / "config/contract.json").read_text())
    task = json.loads((SOURCE / "task_contract.json").read_text())["settings"]
    binding = cfg["design_preflight"]
    if sha(binding["file"]) != binding["sha256"]:
        raise ValueError("Source mechanism design changed")
    design = json.loads(Path(binding["file"]).read_text())
    expected = deepcopy(design["stages"]["speed"]["roles"]["train"])
    control = load_chassis_control(ROOT / cfg["control_math_source"])
    original = design_preflight(cfg, control, ROOT, stage="speed", role="train", seed=211322, num_envs=48)

    def project(value):
        value = deepcopy(value)
        value["scene_groups"] = [dict(name="stand_mid", fraction=1., terrain=["flat"])]
        value["target_num_envs"] = 1
        value["flat_group_episode_seconds"] = {}
        value["evaluation"]["seed"] = seed
        if "evaluation.seed" in value:
            value["evaluation.seed"] = seed
        return value

    cfg, expected["contract"] = project(cfg), project(expected["contract"])
    expected["invocation"] = dict(num_envs=1, seed=dict(rule="fixed", value=seed))
    expected["runtime"]["seed"] = seed
    design = {k: v for k, v in design.items() if k != "stages"}
    design.update(run_scope="interactive_unscored", qualification=None,
        source_contract_sha256=sha(SOURCE / "config/contract.json"), source_role="train",
        allowed_changes=["scene_groups", "target_num_envs", "flat_group_episode_seconds", "evaluation.seed", "invocation", "runtime.seed"],
        stages={"speed": {"roles": {"playback": expected}}})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "playback_expectations.json", design)
    cfg["design_preflight"] = dict(file=str(output / "playback_expectations.json"),
        sha256=sha(output / "playback_expectations.json"), stage="speed")
    proof = design_preflight(cfg, control, ROOT, stage="speed", role="playback", seed=seed, num_envs=1)
    write(output / "playback_contract.json", cfg)
    write(output / "task_contract.json", dict(settings=task, source_sha256=sha(SOURCE / "task_contract.json")))
    write(output / "CPU_preflight.json", dict(source=original, playback=proof, checkpoint_sha256=MODEL_SHA,
        no_optimizer=True, qualification=None, policy_hz=200, physics_hz=1000,
        physics_parameters_changed=False, sensor_and_actuator_domains_changed=False))
    manifest = json.loads((ROOT / cfg["asset_directory"] / "manifest.json").read_text())
    return cfg, task, manifest, control, mean, info


class View:
    def __init__(self, env, args):
        import carb.input
        import carb.windowing
        import omni.appwindow
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        self.env, self.args = env, args
        self.paused = False
        self.reset_requested = False
        self.selected = NAMES.index(args.case)
        self.follow = True
        self.finished_wall = None
        self.auto = True
        window = omni.appwindow.get_default_app_window()
        carb.windowing.acquire_windowing_interface().set_window_title(window.get_window(),
            "Pure RL getup115 | left / right / inverted180 | P pause | R replay | N next")
        self.input = carb.input.acquire_input_interface()
        self.keyboard = window.get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.key)
        stage = env.sim.stage
        UsdLux.DomeLight.Define(stage, "/World/GetupDome").CreateIntensityAttr(1000.)
        colors = ((.22, .56, .85), (.94, .52, .17))
        for body in env.robot.body_names:
            visual = stage.GetPrimAtPath(env.env_paths[0] + "/Robot/" + body + "/Visual")
            color = (.4, .44, .49) if body == "base_link" else colors[int(body.startswith("R"))]
            material = UsdShade.Material.Define(stage, "/World/GetupMaterials/" + body)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.5)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)
        self.viewport = get_active_viewport()
        self.viewport.resolution = (1280, 720)
        self.window = ui.Window("PURE RL GETUP / ACTUAL MOTOR STATE", width=530, height=490)
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label("115 candidate | six motors controlled by RL after passive settle\nBlue: left leg. Orange: right leg. No scripted recovery.", word_wrap=True, height=44)
                self.selector = ui.ComboBox(self.selected, *NAMES, height=24)
                self.selector.model.add_item_changed_fn(self.choose)
                with ui.HStack(height=28):
                    ui.Button("Replay (R)", clicked_fn=self.replay)
                    ui.Button("Pause / resume (P)", clicked_fn=self.pause)
                    ui.Button("Next (N)", clicked_fn=self.next)
                with ui.HStack(height=24):
                    checkbox = ui.CheckBox(width=24)
                    checkbox.model.set_value(True)
                    checkbox.model.add_value_changed_fn(lambda m: setattr(self, "auto", m.as_bool))
                    ui.Label("Auto show eight fallen cases once; each result holds 2 s")
                with ui.HStack(height=24):
                    follow = ui.CheckBox(width=24)
                    follow.model.set_value(True)
                    follow.model.add_value_changed_fn(lambda m: setattr(self, "follow", m.as_bool))
                    ui.Label("Follow camera (untick to orbit freely)")
                ui.Label("Physics 1000 Hz / actor 200 Hz. Playback capped at 0.5x;\nactual speed depends on rendering. Simulated time is shown below.", word_wrap=True, height=40)
                self.state = ui.Label("Initializing native state", word_wrap=True, height=210)
                ui.Label("Candidate demonstration only. Failed attempts remain visible.\nChanging a case or Replay resets the complete physical episode.", word_wrap=True, height=40)

    def choose(self, model, _item):
        selected = model.get_item_value_model().as_int
        if selected != self.selected:
            self.selected = selected
            self.replay()

    def replay(self):
        self.reset_requested, self.paused = True, False

    def pause(self):
        self.paused = not self.paused

    def next(self):
        self.selector.model.get_item_value_model().set_value((self.selected + 1) % len(NAMES))

    def key(self, event, *_):
        if event.type.name == "KEY_PRESS":
            if event.input.name == "P":
                self.pause()
            elif event.input.name == "R":
                self.replay()
            elif event.input.name == "N":
                self.next()
        return True

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=211322)
    parser.add_argument("--case", choices=NAMES, default=NAMES[0])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    cfg, task, manifest, control, mean, info = prepare(args.output, args.seed)
    if args.preflight_only:
        print("GUI118 CPU preflight passed; no simulation launched", flush=True)
        return
    # Do not share a native GPU scene with the held qualification worker.
    boundary = json.loads((REPORTS / "gui118_boundary.json").read_text())
    pid = boundary["holdout_native_pid"]
    if Path(f"/proc/{pid}/stat").exists() and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z":
        raise RuntimeError("The held qualification worker has not finished")
    if json.loads((ROOT / "runs/unified_robust_20261022/composed_static_holdout115/progress.json").read_text())["status"] != "completed":
        raise RuntimeError("Holdout native receipt is not complete")
    from play_flat_base import prepare_flat_gui_experience
    experience = prepare_flat_gui_experience(args.output)
    receipt = dict(schema="pure-rl-gui118-v1", status="starting", pid=os.getpid(),
        checkpoint=str(MODEL), checkpoint_sha256=MODEL_SHA, source_sha256=sha(__file__),
        started=time.time(), run_scope="interactive_unscored", qualification=None,
        policy_hz=200, physics_hz=1000, scripted_recovery=False, optimizer_updates=0,
        state_writes="episode_reset_only", cases=[], render_transport="USD", episode_results=[])
    write(args.output / "report.json", receipt)
    launcher = env = view = writer = None
    interrupted = {"signal": None}
    def stop(sig, _frame):
        interrupted["signal"] = sig
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=False, device="cuda:0", enable_cameras=False,
            experience=experience["file"], visualizer=["kit"]))
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from pxr import UsdGeom, Usd
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.unified_getup import FAMILIES
        from train_robust_getup import fixed_cases
        from omni.kit.viewport.utility import capture_viewport_to_file
        torch.set_num_threads(4)
        mean = mean.cuda().eval()
        bank = REPORTS / "pose_bank002.json"
        env = RobustGetupEnv(cfg, manifest, control, ROOT, stage_name="speed", num_envs=1,
            device="cuda:0", seed=args.seed, level=1., run_role="playback",
            bank=bank, settings=task, render_use_fabric=False)
        assert env.recovery is None and not env.sim.cfg.use_fabric
        templates = {c["name"]: c for c in fixed_cases(env.bank, 35, args.seed)}
        cases = [templates[n] for n in NAMES]
        receipt["cases"] = cases
        receipt["startup_report"] = env.startup_report
        writer = SummaryWriter(str(args.output), filename_suffix=".gui118")
        view = View(env, args)
        ids = torch.arange(1, device=env.device)
        view.reset_requested = True
        completed = False
        cycle = -1
        rows = []
        saved_times = set()
        wall_start = last_render = time.monotonic()
        receipt.update(status="gui_running", native_recovery_is_none=True,
            render_use_fabric=env.sim.cfg.use_fabric, update_to_usd=env.sim.get_setting("/physics/updateToUsd"))
        write(args.output / "report.json", receipt)
        print("GUI118_READY " + str(args.output), flush=True)
        with torch.inference_mode():
            while launcher.app.is_running() and interrupted["signal"] is None:
                if view.reset_requested:
                    if rows:
                        path = args.output / f"episode_{cycle:03d}_{active}.npz"
                        np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
                        receipt["episode_results"].append(dict(case=active, trajectory=str(path), sha256=sha(path),
                            completed=completed, seconds=float(rows[-1]["time"][0])))
                    cycle += 1
                    active = NAMES[view.selected]
                    env.fixed_cases = [deepcopy(cases[view.selected])]
                    env.generator.manual_seed(args.seed)
                    env.domain.rng.manual_seed(args.seed + 83977)
                    env.reset(ids)
                    view.reset_requested = False
                    view.finished_wall = None
                    completed, rows, saved_times = False, [], set()
                    tick, wall_start = 0, time.monotonic()
                    write(args.output / "report.json", receipt)
                    print("GUI118_CASE " + active, flush=True)
                if view.paused or completed:
                    env.sim.render()
                    time.sleep(.02)
                    if completed and view.auto and view.selected < 7 and time.monotonic() - view.finished_wall > 2.:
                        view.next()
                    continue
                obs = env.get_observations()["policy"]
                action = mean(obs)
                _, _, done, extras = env.step(action)
                d = extras["diagnostics"]
                tick += 1
                keys = ("time", "gravity", "height", "motor_position", "motor_velocity", "motor_effort",
                        "actions", "gap", "passive", "stable", "recovered", "followup_success", "terminated", "local_position",
                        "enable_event", "settled_family")
                rows.append({k: d[k].detach().cpu().numpy() for k in keys}
                            | {"observations": obs.cpu().numpy()})
                seconds = float(d["time"][0])
                terminal = bool(done[0]) or seconds >= (30. if active.startswith("stand") else 12.) - .001
                if tick % 8 == 0 or terminal:
                    position = env.robot.data.root_link_pose_w.torch[0, :3].cpu().numpy()
                    if view.follow:
                        env.sim.set_camera_view(tuple(position + [1.05, 1.35, .7]), tuple(position + [0., 0., -.03]))
                    q = d["motor_position"][0].cpu().numpy()
                    legs = np.degrees(q[[0, 1, 3, 4]])
                    wheels = d["motor_velocity"][0, [2, 5]].cpu().tolist()
                    tau = d["motor_effort"][0].cpu().tolist()
                    phase = "PASSIVE: all six motor torques zero" if bool(d["passive"][0]) else "RL: all six motors"
                    if bool(d["terminated"][0]):
                        phase = "TERMINATED: native protection triggered"
                    family_index = int(d["settled_family"][0])
                    settled = FAMILIES[family_index] if family_index >= 0 else "settling"
                    view.state.text = (f"{active} | sim {seconds:.3f} s | {phase}\n"
                        f"Actual settled attitude: {settled}\n"
                        f"Recovered: {bool(d['recovered'][0])} | stable now: {bool(d['stable'][0])}\n"
                        f"Height {float(d['height'][0]):.3f} m | gap {float(d['gap'][0])*1000:.3f} mm\n"
                        f"Actual left hip/aux: {legs[0]:+.1f} / {legs[1]:+.1f} deg\n"
                        f"Actual right hip/aux: {legs[2]:+.1f} / {legs[3]:+.1f} deg\n"
                        f"Wheel speeds: {wheels[0]:+.2f}, {wheels[1]:+.2f} rad/s\n"
                        f"Actual torques [LH,LA,LW,RH,RA,RW] N.m:\n" + ", ".join(f"{x:+.2f}" for x in tau))
                    for tag, value in dict(height_m=float(d["height"][0]), closure_gap_m=float(d["gap"][0]),
                        passive=float(d["passive"][0]), stable=float(d["stable"][0]), recovered=float(d["recovered"][0])).items():
                        writer.add_scalar(f"GetupViewer/{active}/{tag}", value, tick)
                    env.sim.render()
                    # Compare actual link translations to the USD display source.
                    if tick % 200 < 8:
                        paths = ("base_link", "L_link1", "LL_link1", "R_link1", "RR_link1")
                        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                        errors, rotation_errors = [], []
                        for body in paths:
                            prim = env.sim.stage.GetPrimAtPath(env.env_paths[0] + "/Robot/" + body)
                            matrix = cache.GetLocalToWorldTransform(prim)
                            visual = np.asarray(matrix.ExtractTranslation())
                            physical = env.robot.data.body_link_pose_w.torch[0, env.robot.body_names.index(body)].cpu().numpy()
                            errors.append(float(np.max(np.abs(visual - physical[:3]))))
                            rotation = matrix.ExtractRotationQuat()
                            quaternion = np.array([*rotation.GetImaginary(), rotation.GetReal()])
                            dot = abs(float(quaternion @ physical[3:])) / (np.linalg.norm(quaternion) * np.linalg.norm(physical[3:]))
                            rotation_errors.append(float(2. * math.acos(np.clip(dot, -1., 1.))))
                        receipt["display_translation_error_m"] = max(errors)
                        receipt["display_rotation_error_rad"] = max(rotation_errors)
                        if max(errors) > .001 or max(rotation_errors) > .003:
                            raise RuntimeError("Displayed robot links do not match live physics")
                        receipt.update(active_case=active, simulated_seconds=seconds, cycles=cycle+1,
                                       rendered_motor_position_rad=q.tolist())
                        write(args.output / "report.json", receipt)
                    capture_second = next((s for s in (0., .5, 1., 2., 4., 8., 12., 30.)
                                           if s not in saved_times and seconds >= s), None)
                    if capture_second is not None:
                        saved_times.add(capture_second)
                        capture_viewport_to_file(view.viewport, str(args.output / f"episode_{cycle:03d}_{active}_{capture_second:04.1f}s.png"))
                    wait = 8 * env.policy_dt / .5 - (time.monotonic() - last_render)
                    if wait > 0:
                        time.sleep(wait)
                    last_render = time.monotonic()
                if terminal:
                    completed = True
                    view.finished_wall = time.monotonic()
                    view.state.text += "\nRESULT HELD: native termination" if bool(d["terminated"][0]) else "\nRESULT HELD: observation window complete"
        if rows:
            path = args.output / f"episode_{cycle:03d}_{active}.npz"
            np.savez_compressed(path, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            receipt["episode_results"].append(dict(case=active, trajectory=str(path), sha256=sha(path),
                completed=completed, seconds=float(rows[-1]["time"][0])))
        receipt.update(status="interrupted" if interrupted["signal"] else "closed", signal=interrupted["signal"])
    except BaseException:
        receipt.update(status="failed", error=traceback.format_exc())
        print(receipt["error"], file=sys.stderr, flush=True)
        raise
    finally:
        receipt["finished"] = time.time()
        write(args.output / "report.json", receipt)
        if writer is not None:
            writer.close()
        if view is not None:
            view.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
