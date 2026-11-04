#!/usr/bin/env python3
"""Read-only real-robot state mirror. GUI renders measured kinematics only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wheeled_algo.wheel_leg_mirror import DEFAULT_MAPPING, MirrorModel, load_telemetry_reader


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Foxglove WebSocket, e.g. ws://192.168.1.10:8765")
    source.add_argument("--replay", type=Path, help="Recorded session.jsonl")
    source.add_argument("--demo", action="store_true", help="SYNTHETIC nominal state; no robot connection")
    parser.add_argument("--record", type=Path, help="New JSONL file; never overwrite")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--rmcs-root", type=Path, default=ROOT.parent / "RMCS")
    parser.add_argument("--fps", type=float, default=60.)
    parser.add_argument("--stale-after", type=float, default=.2, help="Seconds since last received complete snapshot")
    parser.add_argument("--base-position", type=float, nargs=3, default=(0., 0., .65), metavar=("X", "Y", "Z"))
    parser.add_argument("--speed", type=float, default=1., help="Replay speed")
    parser.add_argument("--loop", action="store_true", help="Repeat replay")
    parser.add_argument("--seconds", type=float, default=0., help="Auto-close after wall seconds; 0 waits for GUI close")
    parser.add_argument("--headless", action="store_true", help="Offscreen render, for automated GUI checks")
    parser.add_argument("--preflight-only", action="store_true", help="Check model, mapping and source dependencies without Kit")
    parser.add_argument("--output", type=Path, help="New directory for runtime.json and summary.json")
    parser.add_argument("--screenshot", type=Path, help="Save one viewport PNG after receiving a valid pose")
    parser.add_argument("--python", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("fps", "stale_after", "speed"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.seconds) or args.seconds < 0 or not np.isfinite(args.base_position).all():
        parser.error("invalid seconds/base position")
    if args.url and not args.url.startswith(("ws://", "wss://")):
        parser.error("--url must use ws:// or wss://")
    if args.loop and not args.replay:
        parser.error("--loop requires --replay")
    for path in (args.record, args.output, args.screenshot):
        if path and path.exists():
            parser.error(f"output already exists: {path}")
    return args


class DemoReader(threading.Thread):
    """Clearly labeled synthetic transport fixture, never a physical robot."""
    def __init__(self, protocol, latest, model):
        super().__init__(daemon=True)
        self.stopped = threading.Event()
        self.protocol, self.latest, self.model = protocol, latest, model

    def run(self):
        calibration = self.model.mapping["calibration"]
        by_source = {e["source"]: e for e in self.model.entries}
        entries = [by_source[name] for name in self.protocol.MOTOR_NAMES]
        positions = tuple((self.model.spec["nominal_joint_pos"][e["target"]] - e["offset_rad"]) / e["sign"]
                          for e in entries)
        # These references come from the nominal model solely for the demo.
        # Real sessions use the actual RMCS estimator's transmitted references.
        q = self.model.spec["nominal_joint_pos"]
        frames = self.model.fk(q)
        knees, lengths = [], []
        for side in self.model.mapping["legs"]:
            h, k, w = (frames[n][:3, 3] for n in side["points"])
            a, b = h - k, w - k
            knees.append(math.acos(float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))
            lengths.append(float(np.linalg.norm(w - h)))
        while not self.stopped.is_set():
            snapshot = self.protocol.Snapshot(time.time_ns(), positions, (1., 0., 0., 0.),
                (0., 0., 0.), (0., 0., -1.), (0., 0., 0.), velocities=(0.,) * 6,
                rmcs_knee=tuple(knees), rmcs_length=tuple(lengths),
                rmcs_model_positions=tuple(q[e["target"]] for e in entries),
                model_offsets=tuple(e["offset_rad"] for e in entries[:4]),
                sensor_age_s=(.001,) * 7, diagnostic_status="SYNTHETIC DEMO", **calibration)
            self.latest.put(snapshot, source="SYNTHETIC DEMO")
            self.stopped.wait(.01)

    def close(self):
        self.stopped.set()
        self.join(timeout=5)


def create_scene(model, args):
    import omni.ui as ui
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux
    from omni.kit.viewport.utility import get_active_viewport

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.MakeInvisible()
    operations = {}
    for body in model.spec["bodies"]:
        path = "/World/Robot/" + body["name"]
        operations[body["name"]] = UsdGeom.Xform.Define(stage, path).AddTransformOp()
        visual = stage.DefinePrim(path + "/Visual")
        visual.GetReferences().AddReference(str(model.visuals), "/Visuals/" + body["component"])
        UsdGeom.Xformable(visual).AddTransformOp().Set(Gf.Matrix4d(np.asarray(body["mesh_origin"]).T.tolist()))
    # No articulation, rigid bodies, collision shapes or physics scene.
    floor = UsdGeom.Cube.Define(stage, "/World/DisplayFloor")
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.025))
    floor.AddScaleOp().Set(Gf.Vec3f(3., 3., .025))
    floor.CreateDisplayColorAttr([Gf.Vec3f(.16, .19, .23)])
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(800.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500.)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-30., -20., 35.))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100.))
    camera.CreateFocalLengthAttr(35.)
    p = np.asarray(args.base_position)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(p + [1.1, 1.6, .65])),
        Gf.Vec3d(*p), Gf.Vec3d(0, 0, 1)).GetInverse())
    viewport = get_active_viewport()
    viewport.camera_path = str(camera.GetPath())
    viewport.resolution = (1280, 900)
    axes = UsdGeom.Xform.Define(stage, "/World/BodyAxes").AddTransformOp()
    for axis, color in zip(np.eye(3), ((1., .15, .15), (.15, 1., .15), (.2, .4, 1.))):
        line = UsdGeom.BasisCurves.Define(stage, "/World/BodyAxes/axis" + str(int(np.argmax(axis))))
        line.CreateTypeAttr("linear")
        line.CreateCurveVertexCountsAttr([2])
        line.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(*(axis * .35))])
        line.CreateWidthsAttr([.006])
        line.SetWidthsInterpolation("constant")
        line.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    panel = ui.Window("Wheel-leg measured state", width=720, height=800)
    with panel.frame:
        with ui.VStack(spacing=7):
            label = "SYNTHETIC DEMO" if args.demo else "RECORDED STATE REPLAY" if args.replay else "REAL ROBOT KINEMATIC MIRROR"
            ui.Label(label, height=26)
            status = ui.Label("WAITING", height=48, word_wrap=True)
            ui.Label("XYZ display anchor fixed | gimbal unmeasured: nominal zero\n"
                     "Axes: X forward (red), Y left (green), Z up (blue)", height=44)
            values = ui.Label("Waiting for a complete, valid snapshot", height=430, word_wrap=True)
            ui.Label("API output shaft -> model joint (rad):\n" + "\n".join(
                f"{e['source']} -> {e['target']} : {e['sign']:+d} q + {e['offset_rad']:.9f}"
                for e in model.entries), height=145)
    return robot, operations, axes, status, values, viewport, panel


def json_state(state):
    return {**state, "body_frames": {n: v.tolist() for n, v in state["body_frames"].items()}}


def main():
    args = arguments()
    model = MirrorModel(args.mapping)
    protocol = load_telemetry_reader(args.rmcs_root)
    if args.url:
        import websockets
        from rosbags.typesys import Stores, get_typestore
        get_typestore(Stores.ROS2_JAZZY)
    if args.replay:
        if next(protocol.read_record(args.replay, model.metadata), None) is None:
            raise ValueError("recording contains no snapshots")
    if args.preflight_only:
        print(json.dumps(dict(model.metadata, joints=model.entries, local_physics=False), indent=2))
        return
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    runtime = Path(os.environ["EXP_PATH"]).parent
    app = SimulationApp({"headless": args.headless, "width": 1600, "height": 1000,
                         "renderer": "RaytracedLighting", "multi_gpu": False,
                         "extra_args": ["--ext-folder", str(runtime / "extscache")]},
                        experience=str(ROOT / "scripts/wheel_leg_live.kit"))
    from pxr import Gf
    from omni.kit.viewport.utility import capture_viewport_to_file
    robot, operations, axes, status, values, viewport, panel = create_scene(model, args)
    latest = protocol.LatestSnapshot(args.record, source="replay" if args.replay else "foxglove", metadata=model.metadata)
    reader = (DemoReader(protocol, latest, model) if args.demo else
              protocol.ReplayReader(args.replay, latest, args.speed, args.loop, model.metadata) if args.replay else
              protocol.FoxgloveReader(args.url, latest))
    reader.start()
    started, last_save, last_snapshot = time.monotonic(), 0., None
    state, state_snapshot, error, screenshot = None, None, "", None
    stats = dict(local_physics=False, source="SYNTHETIC DEMO" if args.demo else str(args.replay or args.url),
                 poses_rendered=0, rejected_snapshots=0, **model.metadata)
    try:
        while app.is_running() and (not args.seconds or time.monotonic() - started < args.seconds):
            tick = time.monotonic()
            snapshot, age, transport = latest.get()
            if snapshot is not None and snapshot is not last_snapshot:
                last_snapshot = snapshot
                try:
                    candidate = model.reconstruct(snapshot, args.base_position)
                    for name, pose in candidate["body_frames"].items():
                        operations[name].Set(Gf.Matrix4d(pose.T.tolist()))
                    axes.Set(Gf.Matrix4d(candidate["body_frames"]["base_link"].T.tolist()))
                    robot.MakeVisible()
                    state, state_snapshot, error = candidate, snapshot, ""
                    stats["poses_rendered"] += 1
                except (ValueError, KeyError, IndexError) as exc:
                    error = str(exc)
                    stats["rejected_snapshots"] += 1
            complete = transport == "replay complete"
            healthy = snapshot is not None and not error and age <= args.stale_after and transport == "receiving"
            flag = "LIVE" if healthy else "REPLAY END" if complete and not error else "STALE / INVALID"
            if args.replay and healthy:
                flag = "REPLAY"
            if args.demo:
                flag = "SYNTHETIC DEMO | " + flag
            warnings = state["diagnostic_warnings"] if state else []
            if healthy and warnings:
                flag += " / DIAGNOSTIC"
            detail = error or "; ".join(warnings) or transport
            status.text = f"{flag} | last receive {age:.3f} s ago\n{detail}"
            status.style = {"color": 0xFF66CFFF if healthy and warnings else 0xFF72D98A if healthy else 0xFF6060FF}
            if state is not None:
                s = state_snapshot
                rows = [f"{e['source']:>17}: API {s.positions_by_name[e['source']]:+.4f} -> model {state['joint_positions'][e['target']]:+.4f} rad; dq {s.velocities_by_name[e['source']]:+.3f} rad/s"
                        for e in model.entries]
                enabled = "UNKNOWN" if s.control_enabled is None else "ENABLED" if s.control_enabled else "DISABLED (feedback still live)"
                rows += [f"chassis {enabled}; DM ready: {s.dm_control_ready}; RMCS estimator valid: {s.state_valid}",
                         f"gyro body {np.round(s.gyro_body, 4)} rad/s",
                         f"quaternion wxyz {np.round(s.quaternion_wxyz, 5)}",
                         f"max sensor age {max(s.sensor_age_s) * 1000:.2f} ms | closure {state['closure_error_m'] * 1000:.5f} mm"]
                if state["reference_valid"]:
                    rows += [f"gravity difference {state['gravity_error']:.3g} | model coordinate delta {max(abs(v) for v in state['coordinate_error_rad']):.3g} rad"]
                else:
                    rows += ["RMCS reference deltas unavailable (estimator invalid); measured pose still updates"]
                for i, side in enumerate(("left", "right")):
                    knee = f"{side} inner knee: model {math.degrees(state['knee_rad'][i]):.3f} deg"
                    length = f"{side} leg length: model {state['leg_length_m'][i] * 1000:.2f} mm"
                    if state["reference_valid"]:
                        knee += f", RMCS {math.degrees(s.rmcs_knee[i]):.3f}, delta {math.degrees(state['knee_error_rad'][i]):+.3f} deg"
                        length += f", RMCS {s.rmcs_length[i] * 1000:.2f}, delta {state['leg_length_error_m'][i] * 1000:+.2f} mm"
                    rows += [knee, length,
                             f"{side} wheel center in body: {np.round(state['wheel_centers_body_m'][i], 4)} m"]
                values.text = "\n".join(rows)
            stats.update(status=flag, transport=transport, last_error=error, frames_received=latest.count,
                         receive_age_s=age if snapshot else None, diagnostic_warnings=warnings)
            if args.output and tick - last_save >= .5:
                payload = {**stats, "current": json_state(state) if state else None}
                (args.output / "runtime.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                last_save = tick
            app.update()
            if args.screenshot and screenshot is None and state is not None and time.monotonic() - started > 2.:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                screenshot = capture_viewport_to_file(viewport, str(args.screenshot.resolve()), is_hdr=False)
            time.sleep(max(0., 1. / args.fps - (time.monotonic() - tick)))
    finally:
        reader.close()
        latest.close()
        if args.output:
            (args.output / "summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
        app.close()


if __name__ == "__main__":
    main()
