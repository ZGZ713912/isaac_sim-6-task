"""Responsive Isaac Sim 3D playback of authenticated native115 body-pose traces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(Path(__file__).parent)]
from gui118_getup import MODEL_SHA, NAMES, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-file", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    receipt = dict(schema="native115-independent-replay-v1", pid=os.getpid(), status="starting",
        checkpoint_sha256=MODEL_SHA, source_sha256=sha(__file__), native_physics_in_viewer=False,
        qualification=None, new_ppo_updates=0, hardware_started=False, controls=[], clips={}, started_unix=time.time())
    write(args.output / "report.json", receipt)
    app = writer = None
    stop_requested = False
    def stop(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    try:
        from play_flat_base import prepare_flat_gui_experience
        experience = prepare_flat_gui_experience(args.output)
        from isaaclab.app import AppLauncher
        app = AppLauncher(dict(headless=False, device="cuda:0", enable_cameras=False,
            experience=experience["file"], visualizer=["kit"]))
        import numpy as np
        import carb.input
        import carb.windowing
        import omni.appwindow
        import omni.kit.viewport.utility as viewport_utils
        import omni.ui as ui
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output), filename_suffix=".gui120_replay")
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        native = json.loads(args.recording.read_text())
        if native["checkpoint_sha256"] != MODEL_SHA or not native.get("body_names"):
            raise ValueError("Native recording has not authenticated its body order")
        bodies = native["body_names"]
        asset = Path(native["asset_directory"]) / native["usd"]
        operations = {}
        for body in bodies:
            root = "/World/RecordedRobot/" + body
            operations[body] = UsdGeom.Xform.Define(stage, root).AddTransformOp()
            visual = stage.DefinePrim(root + "/Visual")
            visual.GetReferences().AddReference(str(asset), "/Robot/" + body + "/Visual")
            UsdGeom.Imageable(visual).MakeVisible()
            color = (.4,.44,.49) if body == "base_link" else ((.94,.52,.17) if body.startswith("R") else (.22,.56,.85))
            material = UsdShade.Material.Define(stage, "/World/Materials/" + body)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.6)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/RecordedRobot")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError("Recorded display contains a physical object")
        robot_display = UsdGeom.Imageable(stage.GetPrimAtPath("/World/RecordedRobot"))
        robot_display.MakeInvisible()
        floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
        floor.CreatePointsAttr([(-4,-4,-.001),(4,-4,-.001),(4,4,-.001),(-4,4,-.001)])
        floor.CreateFaceVertexCountsAttr([4]); floor.CreateFaceVertexIndicesAttr([0,1,2,3])
        floor.CreateDisplayColorAttr([(.32,.34,.37)])
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.)
        camera = UsdGeom.Camera.Define(stage, "/World/ReplayCamera")
        camera.CreateFocalLengthAttr(32.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005,100.))
        camera_op = camera.AddTransformOp()
        viewport = viewport_utils.get_active_viewport()
        viewport.camera_path = "/World/ReplayCamera"
        viewport.resolution = (1280,720)
        state = dict(selected=0, paused=False, speed=.5, follow=True, auto=True, playhead=0.,
                     end_wall=None, closed=False, initial_clip_selected=False)
        clips, last_id, last_catalog, last_report, last_frame = {}, None, 0., 0., time.monotonic()
        screenshots = set()
        def select(index):
            state.update(selected=index, playhead=0., end_wall=None, initial_clip_selected=True)
        def control(command):
            action = command["action"]
            if action == "pause": state["paused"] = not state["paused"]
            elif action == "replay": state.update(playhead=0., paused=False, end_wall=None)
            elif action == "next": selector.model.get_item_value_model().set_value((state["selected"]+1)%8)
            elif action == "select": selector.model.get_item_value_model().set_value(NAMES.index(command["case"]))
            elif action == "speed": state["speed"] = float(command["value"])
            elif action == "close": state["closed"] = True
            else: raise ValueError("Unknown replay control")
            receipt["controls"].append(dict(command=command, handled_unix=time.time()))
        def key(event,*_):
            if event.type.name == "KEY_PRESS" and event.input.name in ("P","R","N"):
                control(dict(action={"P":"pause","R":"replay","N":"next"}[event.input.name], source="keyboard"))
            return True
        window = omni.appwindow.get_default_app_window()
        carb.windowing.acquire_windowing_interface().set_window_title(window.get_window(),
            "Getup115 native 3D REPLAY | P pause | R replay | N next")
        input_api = carb.input.acquire_input_interface()
        subscription = input_api.subscribe_to_keyboard_events(window.get_keyboard(), key)
        panel = ui.Window("GETUP115 NATIVE RECORDING", width=490, height=420)
        with panel.frame:
            with ui.VStack(spacing=7):
                ui.Label("RECORDED NATIVE PHYSX / exact115 actor\nIndependent display: no physics or policy is recomputed here.", word_wrap=True, height=42)
                selector = ui.ComboBox(0, *NAMES[:8], height=26)
                selector.model.add_item_changed_fn(lambda model,item: select(model.get_item_value_model().as_int))
                with ui.HStack(height=28):
                    ui.Button("Replay (R)", clicked_fn=lambda: control(dict(action="replay",source="button")))
                    ui.Button("Pause/resume (P)", clicked_fn=lambda: control(dict(action="pause",source="button")))
                    ui.Button("Next (N)", clicked_fn=lambda: control(dict(action="next",source="button")))
                with ui.HStack(height=26):
                    ui.Label("Playback speed", width=130)
                    speed = ui.ComboBox(1,"0.25x","0.5x","1x","2x")
                    speed.model.add_item_changed_fn(lambda model,item: state.update(speed=(.25,.5,1.,2.)[model.get_item_value_model().as_int]))
                with ui.HStack(height=24):
                    follow = ui.CheckBox(width=24); follow.model.set_value(True)
                    follow.model.add_value_changed_fn(lambda m: state.update(follow=m.as_bool))
                    ui.Label("Follow camera (untick to orbit and zoom)")
                with ui.HStack(height=24):
                    auto = ui.CheckBox(width=24); auto.model.set_value(True)
                    auto.model.add_value_changed_fn(lambda m: state.update(auto=m.as_bool))
                    ui.Label("Auto show all eight cases once; hold results2s")
                text = ui.Label("Loading native clips", word_wrap=True, height=190)
                ui.Label("Blue left / orange right. Failures are kept.\n200Hz saved frames; native physics1000Hz. Not a qualification run.", word_wrap=True, height=42)
        target = ui.Workspace.get_window("Stage")
        if target: panel.dock_in(target,ui.DockPosition.SAME)
        receipt.update(status="gui_running", body_names=bodies, native_recording=str(args.recording.resolve()))
        write(args.output / "report.json",receipt)
        print("GUI120_REPLAY_READY",flush=True)
        frames, render_times = 0, []
        while app.app.is_running() and not stop_requested and not state["closed"]:
            now = time.monotonic(); dt = now-last_frame; last_frame = now
            if args.control_file and args.control_file.exists():
                command = json.loads(args.control_file.read_text())
                if command["id"] != last_id:
                    last_id=command["id"]; control(command)
            if now-last_catalog > .5:
                native=json.loads(args.recording.read_text()); last_catalog=now
                for row in native["clips"]:
                    name=row["case"]
                    if name in clips: continue
                    path=Path(row["trajectory"])
                    if sha(path)!=row["sha256"]: raise ValueError("Native clip hash changed")
                    with np.load(path) as data: arrays={k:data[k].copy() for k in data.files}
                    if arrays["body_link_pose_w"].shape[2:]!=(len(bodies),7): raise ValueError("Body order/shape mismatch")
                    if not np.isfinite(arrays["body_link_pose_w"]).all(): raise ValueError("Nonfinite native pose")
                    clips[name]=(row,arrays)
                    receipt["clips"][name]=dict(trajectory_sha256=row["sha256"], seconds=row["seconds"], native_terminated=row["native_terminated"], ever_recovered=row["ever_recovered"])
                if clips and not state["initial_clip_selected"]:
                    index = min(NAMES.index(name) for name in clips)
                    state["initial_clip_selected"] = True
                    selector.model.get_item_value_model().set_value(index)
            name=NAMES[state["selected"]]
            if name in clips:
                robot_display.MakeVisible()
                row,arrays=clips[name]; times=arrays["time"][:,0]
                if not state["paused"]: state["playhead"]=min(float(times[-1]),state["playhead"]+min(dt,.1)*state["speed"])
                index=max(0,int(np.searchsorted(times,state["playhead"],side="right"))-1)
                poses=arrays["body_link_pose_w"][index,0].copy()
                origin=np.asarray(native["origins"][row["lane"]])
                poses[:,:3]-=origin
                for body,pose in zip(bodies,poses):
                    matrix=Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]),Gf.Vec3d(*(float(v) for v in pose[3:6]))))
                    operations[body].Set(matrix.SetTranslateOnly(Gf.Vec3d(*(float(v) for v in pose[:3]))))
                if state["follow"]:
                    center=(poses[:,:3].min(0)+poses[:,:3].max(0))*.5
                    eye=center+[1.4,1.5,.85]
                    camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(float(x) for x in eye)),Gf.Vec3d(*(float(x) for x in center)),Gf.Vec3d(0,0,1)).GetInverse())
                phase="PASSIVE" if bool(arrays["passive"][index,0]) else "SIX-AXIS RL"
                if bool(arrays["terminated"][index,0]): phase="NATIVE PROTECTION / attempt ended"
                q=np.degrees(arrays["motor_position"][index,0,[0,1,3,4]])
                text.text=(f"{name} | RECORDED {times[index]:.3f}/{times[-1]:.3f}s | {state['speed']}x\n"
                    f"{'PAUSED' if state['paused'] else phase}\n"
                    f"Recovered now {bool(arrays['recovered'][index,0])}; ever {row['ever_recovered']}\n"
                    f"Native terminated {row['native_terminated']}\n"
                    f"Left hip/aux {q[0]:+.1f}/{q[1]:+.1f}deg\nRight hip/aux {q[2]:+.1f}/{q[3]:+.1f}deg\n"
                    f"Available clips {len(clips)}/8; recorder {native['status']}")
                if state["playhead"] >= times[-1]:
                    if state["end_wall"] is None: state["end_wall"]=now
                    if state["auto"] and not state["paused"] and state["selected"]<7 and now-state["end_wall"]>2.:
                        selector.model.get_item_value_model().set_value(state["selected"]+1)
                receipt.update(active_case=name, displayed_native_time_s=float(times[index]), displayed_native_frame=index)
            else:
                robot_display.MakeInvisible()
                text.text=f"{name}: native simulation is recording.\nAvailable {len(clips)}/8.\nNative progress {native.get('last_simulated_seconds',0.):.2f}/12s.\nChoose an available case; the GUI remains responsive."
            render_start=time.monotonic(); app.app.update(); render_times.append(time.monotonic()-render_start); frames+=1
            if name in clips and name not in screenshots and state["playhead"]>=min(.5,float(clips[name][1]["time"][-1,0])):
                viewport_utils.capture_viewport_to_file(viewport,str(args.output/(name+".png"))); screenshots.add(name)
            if now-last_report>=1.:
                cache=UsdGeom.XformCache(Usd.TimeCode.Default())
                if name in clips:
                    errors=[np.max(np.abs(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/RecordedRobot/'+body)).ExtractTranslation())-pose[:3])) for body,pose in zip(bodies,poses)]
                    receipt["display_translation_error_m"]=float(max(errors))
                receipt.update(heartbeat_unix=time.time(),paused=state["paused"],rendered_frames=frames,
                    median_app_update_ms=float(np.median(render_times[-120:])*1000.))
                writer.add_scalar("GetupReplay/app_update_ms",receipt["median_app_update_ms"],frames)
                write(args.output/"report.json",receipt); last_report=now
            remaining=1./60.-(time.monotonic()-now)
            if remaining>0: time.sleep(remaining)
        input_api.unsubscribe_to_keyboard_events(window.get_keyboard(),subscription)
        receipt.update(status="closed",finished_unix=time.time());write(args.output/"report.json",receipt)
    except BaseException:
        receipt.update(status="failed",error=traceback.format_exc());write(args.output/"report.json",receipt);raise
    finally:
        if writer is not None: writer.close()
        if app is not None: app.app.close()


if __name__=="__main__":
    main()
